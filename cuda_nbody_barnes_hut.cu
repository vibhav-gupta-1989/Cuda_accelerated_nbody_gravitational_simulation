#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define N 50000
#define WIDTH 800
#define HEIGHT 800
#define STEPS 20
#define MIN_CELL_SIZE 1e-5f

#define DT 0.001f
#define G 1.0f
#define THETA 0.85f
#define SOFTENING 1e-9f

#define BLOCK 256

struct Bodies{
    float *x;
    float *y;

    float *vx;
    float *vy;

    float *mass;
};

////////////////////////////////////////////////////////////
// Barnes-Hut Tree Node
////////////////////////////////////////////////////////////

struct QuadNode{
    float cx,cy;
    float half;

    float mass;
    float com_x,com_y;

    int body;

    int children[4];
    bool leaf;
};

////////////////////////////////////////////////////////////
// Insert Body Into Tree
////////////////////////////////////////////////////////////

void insert_body(std::vector<QuadNode>& nodes,
                 Bodies bodies,
                 int nodeIndex,
                 int bodyIndex)
{
    // Empty leaf
    if(nodes[nodeIndex].leaf && nodes[nodeIndex].body == -1)
    {
        nodes[nodeIndex].body = bodyIndex;
        return;
    }

    // Leaf already contains a body → split
    if(nodes[nodeIndex].leaf)
    {
        // Prevent infinite subdivision
        if(nodes[nodeIndex].half < MIN_CELL_SIZE)
        {
            // Just accumulate mass without further splitting
            return;
        }
        int oldBody = nodes[nodeIndex].body;

        float cx = nodes[nodeIndex].cx;
        float cy = nodes[nodeIndex].cy;
        float half = nodes[nodeIndex].half;

        nodes[nodeIndex].body = -1;
        nodes[nodeIndex].leaf = false;

        // Create children
        for(int i=0;i<4;i++)
        {
            QuadNode child;

            child.half = half / 2.0f;

            child.cx = cx + ((i % 2) ? child.half : -child.half);
            child.cy = cy + ((i / 2) ? child.half : -child.half);

            child.mass = 0;
            child.com_x = 0;
            child.com_y = 0;
            child.body = -1;
            child.leaf = true;

            for(int j=0;j<4;j++) child.children[j] = -1;

            nodes.push_back(child);

            nodes[nodeIndex].children[i] = nodes.size() - 1;
        }

        // Reinsert old body
        int oldQuad =
            (bodies.x[oldBody] > cx) +
            2 * (bodies.y[oldBody] > cy);

        insert_body(nodes,
                    bodies,
                    nodes[nodeIndex].children[oldQuad],
                    oldBody);
    }

    // Insert new body
    int quad =
        (bodies.x[bodyIndex] > nodes[nodeIndex].cx) +
        2 * (bodies.y[bodyIndex] > nodes[nodeIndex].cy);

    insert_body(nodes,
                bodies,
                nodes[nodeIndex].children[quad],
                bodyIndex);
}


////////////////////////////////////////////////////////////
// Compute Center of Mass
////////////////////////////////////////////////////////////

void compute_mass(std::vector<QuadNode>& nodes,
                  Bodies bodies,
                  int nodeIndex)
{
    QuadNode &node=nodes[nodeIndex];

    if(node.leaf)
    {
        if(node.body!=-1)
        {
            node.mass = bodies.mass[node.body];
            node.com_x = bodies.x[node.body];
            node.com_y = bodies.y[node.body];
        }
        return;
    }

    float mass=0;
    float cx=0;
    float cy=0;

    for(int i=0;i<4;i++)
    {
        int child=node.children[i];
        if(child==-1) continue;

        compute_mass(nodes,bodies,child);

        QuadNode &c=nodes[child];

        mass+=c.mass;
        cx+=c.com_x*c.mass;
        cy+=c.com_y*c.mass;
    }

    node.mass=mass;

    if(mass>0)
    {
        node.com_x=cx/mass;
        node.com_y=cy/mass;
    }
}

////////////////////////////////////////////////////////////
// GPU Barnes-Hut Force Kernel
////////////////////////////////////////////////////////////

__global__ void compute_forces_bh(
        Bodies bodies,
        QuadNode *nodes)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;

    float my_x = bodies.x[i];
    float my_y = bodies.y[i];
    float my_vx = bodies.vx[i];
    float my_vy = bodies.vy[i];
    float my_mass = bodies.mass[i];

    float fx=0;
    float fy=0;

    int stack[64];
    int top=0;

    stack[top++]=0;

    while(top>0)
    {
        int nodeIndex=stack[--top];
        QuadNode node=nodes[nodeIndex];

        float dx=node.com_x-my_x;
        float dy=node.com_y-my_y;

        float dist=sqrtf(dx*dx+dy*dy+SOFTENING*SOFTENING);

        if(node.leaf || node.half/dist<THETA)
        {
            if(node.body == i) continue;

            float force=
                G*my_mass*node.mass/(dist*dist);

            fx+=force*dx/dist;
            fy+=force*dy/dist;
        }
        else
        {
            for(int k=0;k<4;k++)
            {
                int child=node.children[k];
                if(child!=-1)
                    stack[top++]=child;
            }
        }
    }

    my_vx+=DT*fx/my_mass;
    my_vy+=DT*fy/my_mass;

    bodies.vx[i]=my_vx;
    bodies.vy[i]=my_vy;
}

////////////////////////////////////////////////////////////
// Update Positions
////////////////////////////////////////////////////////////

__global__ void update_positions(Bodies bodies)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;

    bodies.x[i]+=bodies.vx[i]*DT;
    bodies.y[i]+=bodies.vy[i]*DT;

    if(bodies.x[i] < 0 || bodies.x[i] > 1){
        bodies.vx[i] *= -1.0;
    }

    if(bodies.y[i] < 0 || bodies.y[i] > 1){
        bodies.vy[i] *= -1.0;
    }
}

////////////////////////////////////////////////////////////
// Save Frame
////////////////////////////////////////////////////////////

void save_frame(Bodies bodies,int frame)
{
    std::vector<int> image(WIDTH*HEIGHT*3,0);

    for(int i=0;i<N;i++)
    {
        int x=bodies.x[i]*WIDTH;
        int y=bodies.y[i]*HEIGHT;

        if(x>=0 && x<WIDTH && y>=0 && y<HEIGHT)
        {
            int index=(y*WIDTH+x)*3;
            image[index]=255;
            image[index+1]=255;
            image[index+2]=255;
        }
    }

    char filename[50];
    sprintf(filename,"frame_%04d.ppm",frame);

    std::ofstream file(filename);

    file<<"P3\n"<<WIDTH<<" "<<HEIGHT<<"\n255\n";

    for(int j=0;j<HEIGHT;j++)
        for(int i=0;i<WIDTH;i++)
        {
            int index=(j*WIDTH+i)*3;

            file<<image[index]<<" "
                <<image[index+1]<<" "
                <<image[index+2]<<"\n";
        }

    file.close();
}

////////////////////////////////////////////////////////////
// CPU Simulation (baseline O(N^2))
////////////////////////////////////////////////////////////

void cpu_simulate(Bodies bodies)
{
    for(int step=0;step<STEPS;step++)
    {
        for(int i=0;i<N;i++)
        {
            float fx=0,fy=0;

            for(int j=0;j<N;j++)
            {
                if(i == j) continue;

                float dx=bodies.x[j]-bodies.x[i];
                float dy=bodies.y[j]-bodies.y[i];

                float dist=sqrt(dx*dx+dy*dy+SOFTENING*SOFTENING);

                float force=
                    G*bodies.mass[i]*bodies.mass[j]/(dist*dist);

                fx+=force*dx/dist;
                fy+=force*dy/dist;
            }

            bodies.vx[i]+=DT*fx/bodies.mass[i];
            bodies.vy[i]+=DT*fy/bodies.mass[i];
        }

        for(int i=0;i<N;i++)
        {
            bodies.x[i]+=bodies.vx[i]*DT;
            bodies.y[i]+=bodies.vy[i]*DT;

            if(bodies.x[i] < 0 || bodies.x[i] > 1){
                bodies.vx[i] *= -1.0;
            }

            if(bodies.y[i] < 0 || bodies.y[i] > 1){
                bodies.vy[i] *= -1.0;
            }
        }
    }
}

////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////

int main()
{
    Bodies cpu_bodies;
    Bodies gpu_bodies;

    cpu_bodies.x = new float[N];
    cpu_bodies.y = new float[N];
    cpu_bodies.vx = new float[N];
    cpu_bodies.vy = new float[N];
    cpu_bodies.mass = new float[N];

    cudaMallocManaged(&gpu_bodies.x, N*sizeof(float));
    cudaMallocManaged(&gpu_bodies.y, N*sizeof(float));
    cudaMallocManaged(&gpu_bodies.vx, N*sizeof(float));
    cudaMallocManaged(&gpu_bodies.vy, N*sizeof(float));
    cudaMallocManaged(&gpu_bodies.mass, N*sizeof(float));

    for(int i=0;i<N;i++)
    {
        cpu_bodies.x[i] = rand()/float(RAND_MAX);
        cpu_bodies.y[i] = rand()/float(RAND_MAX);

        cpu_bodies.vx[i] = 0;
        cpu_bodies.vy[i] = 0;
        cpu_bodies.mass[i] = 1;

        gpu_bodies.x[i] = cpu_bodies.x[i];
        gpu_bodies.y[i] = cpu_bodies.y[i];
        gpu_bodies.vx[i] = cpu_bodies.vx[i];
        gpu_bodies.vy[i] = cpu_bodies.vy[i];
        gpu_bodies.mass[i] = cpu_bodies.mass[i];
    }

    ////////////////////////////////////////////////////////////
    // CPU timing
    ////////////////////////////////////////////////////////////

    auto cpu_start=std::chrono::high_resolution_clock::now();

    cpu_simulate(cpu_bodies);

    auto cpu_end=std::chrono::high_resolution_clock::now();

    double cpu_time=
        std::chrono::duration<double>(cpu_end-cpu_start).count();

    ////////////////////////////////////////////////////////////
    // GPU timing
    ////////////////////////////////////////////////////////////

    int threads=BLOCK;
    int blocks=(N+threads-1)/threads;

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    std::vector<QuadNode> nodes;
    nodes.reserve(4*N);

    QuadNode *d_nodes;

    cudaMalloc(&d_nodes,
        4*N*sizeof(QuadNode));

    for(int step=0;step<STEPS;step++)
    {
        QuadNode root;

        root.cx=0.5f;
        root.cy=0.5f;
        root.half=0.5f;

        root.mass=0;
        root.com_x=0;
        root.com_y=0;

        root.body=-1;
        root.leaf=true;

        for(int i=0;i<4;i++) root.children[i]=-1;

        
        nodes.push_back(root);
        
        for(int i=0;i<N;i++){
            //std::cout << "Reaching here " << i << std::endl;
            insert_body(nodes,gpu_bodies,0,i);
            //std::cout << "Tree nodes: " << nodes.size() << std::endl;
        }

        
        
        compute_mass(nodes,gpu_bodies,0);

        cudaMemcpy(d_nodes,
                   nodes.data(),
                   nodes.size()*sizeof(QuadNode),
                   cudaMemcpyHostToDevice);
        compute_forces_bh<<<blocks,threads>>>(
                gpu_bodies,d_nodes);

        update_positions<<<blocks,threads>>>(gpu_bodies);

        cudaDeviceSynchronize();

        
        //save_frame(gpu_bodies,step);

        nodes.clear();
    }

    cudaFree(d_nodes);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms,start,stop);

    double gpu_time=gpu_ms/1000.0;

    ////////////////////////////////////////////////////////////
    // Results
    ////////////////////////////////////////////////////////////

    std::cout<<"CPU Time: "<<cpu_time<<" seconds\n";
    std::cout<<"GPU Time: "<<gpu_time<<" seconds\n";
    std::cout<<"Speedup: "<<cpu_time/gpu_time<<"x\n";

    cudaFree(gpu_bodies.x);
    cudaFree(gpu_bodies.y);
    cudaFree(gpu_bodies.vx);
    cudaFree(gpu_bodies.vy);
    cudaFree(gpu_bodies.mass);
    
    delete[] cpu_bodies.x;
    delete[] cpu_bodies.y;
    delete[] cpu_bodies.vx;
    delete[] cpu_bodies.vy;
    delete[] cpu_bodies.mass;
}