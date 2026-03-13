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

struct Body{
    float x,y;
    float vx,vy;
    float mass;
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
                 Body* bodies,
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
            (bodies[oldBody].x > cx) +
            2 * (bodies[oldBody].y > cy);

        insert_body(nodes,
                    bodies,
                    nodes[nodeIndex].children[oldQuad],
                    oldBody);
    }

    // Insert new body
    int quad =
        (bodies[bodyIndex].x > nodes[nodeIndex].cx) +
        2 * (bodies[bodyIndex].y > nodes[nodeIndex].cy);

    insert_body(nodes,
                bodies,
                nodes[nodeIndex].children[quad],
                bodyIndex);
}


////////////////////////////////////////////////////////////
// Compute Center of Mass
////////////////////////////////////////////////////////////

void compute_mass(std::vector<QuadNode>& nodes,
                  Body* bodies,
                  int nodeIndex)
{
    QuadNode &node=nodes[nodeIndex];

    if(node.leaf)
    {
        if(node.body!=-1)
        {
            Body &b=bodies[node.body];

            node.mass=b.mass;
            node.com_x=b.x;
            node.com_y=b.y;
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
        Body *bodies,
        QuadNode *nodes)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;

    Body my=bodies[i];

    float fx=0;
    float fy=0;

    int stack[64];
    int top=0;

    stack[top++]=0;

    while(top>0)
    {
        int nodeIndex=stack[--top];
        QuadNode node=nodes[nodeIndex];

        float dx=node.com_x-my.x;
        float dy=node.com_y-my.y;

        float dist=sqrtf(dx*dx+dy*dy+SOFTENING*SOFTENING);

        if(node.leaf || node.half/dist<THETA)
        {
            if(node.body == i) continue;

            float force=
                G*my.mass*node.mass/(dist*dist);

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

    my.vx+=DT*fx/my.mass;
    my.vy+=DT*fy/my.mass;

    bodies[i]=my;
}

////////////////////////////////////////////////////////////
// Update Positions
////////////////////////////////////////////////////////////

__global__ void update_positions(Body *bodies)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;

    bodies[i].x+=bodies[i].vx*DT;
    bodies[i].y+=bodies[i].vy*DT;

    if(bodies[i].x < 0 || bodies[i].x > 1){
        bodies[i].vx *= -1.0;
    }

    if(bodies[i].y < 0 || bodies[i].y > 1){
        bodies[i].vy *= -1.0;
    }
}

////////////////////////////////////////////////////////////
// Save Frame
////////////////////////////////////////////////////////////

void save_frame(Body *bodies,int frame)
{
    std::vector<int> image(WIDTH*HEIGHT*3,0);

    for(int i=0;i<N;i++)
    {
        int x=bodies[i].x*WIDTH;
        int y=bodies[i].y*HEIGHT;

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

void cpu_simulate(Body *bodies)
{
    for(int step=0;step<STEPS;step++)
    {
        for(int i=0;i<N;i++)
        {
            float fx=0,fy=0;

            for(int j=0;j<N;j++)
            {
                if(i == j) continue;

                float dx=bodies[j].x-bodies[i].x;
                float dy=bodies[j].y-bodies[i].y;

                float dist=sqrt(dx*dx+dy*dy+SOFTENING*SOFTENING);

                float force=
                    G*bodies[i].mass*bodies[j].mass/(dist*dist);

                fx+=force*dx/dist;
                fy+=force*dy/dist;
            }

            bodies[i].vx+=DT*fx/bodies[i].mass;
            bodies[i].vy+=DT*fy/bodies[i].mass;
        }

        for(int i=0;i<N;i++)
        {
            bodies[i].x+=bodies[i].vx*DT;
            bodies[i].y+=bodies[i].vy*DT;

            if(bodies[i].x < 0 || bodies[i].x > 1){
                bodies[i].vx *= -1.0;
            }

            if(bodies[i].y < 0 || bodies[i].y > 1){
                bodies[i].vy *= -1.0;
            }
        }
    }
}

////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////

int main()
{
    Body *cpu_bodies=new Body[N];
    Body *gpu_bodies;

    cudaMallocManaged(&gpu_bodies,N*sizeof(Body));

    for(int i=0;i<N;i++)
    {
        cpu_bodies[i].x=rand()/float(RAND_MAX);
        cpu_bodies[i].y=rand()/float(RAND_MAX);

        cpu_bodies[i].vx=0;
        cpu_bodies[i].vy=0;
        cpu_bodies[i].mass=1;

        gpu_bodies[i]=cpu_bodies[i];
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

        QuadNode *d_nodes;

        cudaMalloc(&d_nodes,
                   nodes.size()*sizeof(QuadNode));

        cudaMemcpy(d_nodes,
                   nodes.data(),
                   nodes.size()*sizeof(QuadNode),
                   cudaMemcpyHostToDevice);
        compute_forces_bh<<<blocks,threads>>>(
                gpu_bodies,d_nodes);

        update_positions<<<blocks,threads>>>(gpu_bodies);

        cudaDeviceSynchronize();

        
        //save_frame(gpu_bodies,step);

        cudaFree(d_nodes);

        nodes.clear();
    }

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

    cudaFree(gpu_bodies);
    delete[] cpu_bodies;
}