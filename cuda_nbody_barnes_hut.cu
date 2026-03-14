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

#define MAX_NODES (N*4)


////////////////////////////////////////////////////////////
// Barnes-Hut Tree Node
////////////////////////////////////////////////////////////

struct QuadTree{
    float *cx;
    float *cy;
    float *half;

    float *mass;
    float *com_x;
    float *com_y;

    int *body;

    int *child0;
    int *child1;
    int *child2;
    int *child3;

    bool *leaf;
};

////////////////////////////////////////////////////////////
// Insert Body Into Tree
////////////////////////////////////////////////////////////

void insert_body(
    QuadTree &tree,
    Bodies &bodies,
    int node,
    int bodyIndex,
    int &nodeCount)
{
    float x = bodies.x[bodyIndex];
    float y = bodies.y[bodyIndex];

    // If node is empty leaf
    if(tree.leaf[node] && tree.body[node] == -1)
    {
        tree.body[node] = bodyIndex;
        return;
    }

    if(tree.half[node] < MIN_CELL_SIZE)
    {
        tree.body[node] = bodyIndex;
        return;
    }

    // If node is leaf but already contains a body
    if(tree.leaf[node])
    {
        int oldBody = tree.body[node];
        tree.body[node] = -1;
        tree.leaf[node] = 0;

        float cx = tree.cx[node];
        float cy = tree.cy[node];
        float half = tree.half[node] * 0.5f;

        if(nodeCount + 4 >= MAX_NODES)
        {
            std::cerr << "Tree overflow\n";
            return;
        }

        // Create children
        tree.child0[node] = nodeCount++;
        tree.child1[node] = nodeCount++;
        tree.child2[node] = nodeCount++;
        tree.child3[node] = nodeCount++;

        int c0 = tree.child0[node];
        int c1 = tree.child1[node];
        int c2 = tree.child2[node];
        int c3 = tree.child3[node];

        // Initialize children
        tree.cx[c0] = cx - half;
        tree.cy[c0] = cy - half;
        tree.half[c0] = half;

        tree.cx[c1] = cx + half;
        tree.cy[c1] = cy - half;
        tree.half[c1] = half;

        tree.cx[c2] = cx - half;
        tree.cy[c2] = cy + half;
        tree.half[c2] = half;

        tree.cx[c3] = cx + half;
        tree.cy[c3] = cy + half;
        tree.half[c3] = half;

        for(int c : {c0,c1,c2,c3})
        {
            tree.leaf[c] = 1;
            tree.body[c] = -1;
            tree.child0[c] = tree.child1[c] = tree.child2[c] = tree.child3[c] = -1;
        }

        // determine quadrant of old body
        float ox = bodies.x[oldBody];
        float oy = bodies.y[oldBody];

        int oldQuadrant = 0;
        if(ox > cx) oldQuadrant += 1;
        if(oy > cy) oldQuadrant += 2;

        int oldChild;

        if(oldQuadrant == 0) oldChild = tree.child0[node];
        else if(oldQuadrant == 1) oldChild = tree.child1[node];
        else if(oldQuadrant == 2) oldChild = tree.child2[node];
        else oldChild = tree.child3[node];

        // insert old body into child
        insert_body(tree, bodies, oldChild, oldBody, nodeCount);
    }

    // Determine quadrant
    float cx = tree.cx[node];
    float cy = tree.cy[node];

    int quadrant = 0;

    if(x > cx) quadrant += 1;
    if(y > cy) quadrant += 2;

    int child;

    if(quadrant == 0) child = tree.child0[node];
    else if(quadrant == 1) child = tree.child1[node];
    else if(quadrant == 2) child = tree.child2[node];
    else child = tree.child3[node];

    // Recurse
    insert_body(tree, bodies, child, bodyIndex, nodeCount);
}


////////////////////////////////////////////////////////////
// Compute Center of Mass
////////////////////////////////////////////////////////////

void compute_mass(QuadTree &tree, Bodies &bodies, int node)
{
    // If this is a leaf node
    if(tree.leaf[node])
    {
        int b = tree.body[node];

        if(b != -1)
        {
            tree.mass[node]  = bodies.mass[b];
            tree.com_x[node] = bodies.x[b];
            tree.com_y[node] = bodies.y[b];
        }
        else
        {
            tree.mass[node] = 0.0f;
        }

        return;
    }

    // Otherwise compute from children
    float mass = 0.0f;
    float com_x = 0.0f;
    float com_y = 0.0f;

    int children[4] = {
        tree.child0[node],
        tree.child1[node],
        tree.child2[node],
        tree.child3[node]
    };

    for(int i = 0; i < 4; i++)
    {
        int c = children[i];

        if(c == -1) continue;

        compute_mass(tree, bodies, c);

        float m = tree.mass[c];

        mass += m;
        com_x += tree.com_x[c] * m;
        com_y += tree.com_y[c] * m;
    }

    tree.mass[node] = mass;

    if(mass > 0.0f)
    {
        tree.com_x[node] = com_x / mass;
        tree.com_y[node] = com_y / mass;
    }
}

////////////////////////////////////////////////////////////
// GPU Barnes-Hut Force Kernel
////////////////////////////////////////////////////////////

__global__ void compute_forces_bh(Bodies bodies, QuadTree tree)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;

    float my_x = bodies.x[i];
    float my_y = bodies.y[i];
    float my_mass = bodies.mass[i];

    float fx = 0.0f;
    float fy = 0.0f;

    int stack[64];
    int top = 0;

    stack[top++] = 0;

    while(top > 0)
    {
        int nodeIndex = stack[--top];

        float dx = tree.com_x[nodeIndex] - my_x;
        float dy = tree.com_y[nodeIndex] - my_y;

        float dist2 = dx*dx + dy*dy + SOFTENING*SOFTENING;

        float half = tree.half[nodeIndex];
        float half2 = half * half;

        if(tree.leaf[nodeIndex] || half2 < THETA*THETA * dist2)
        {
            if(tree.body[nodeIndex] == i) continue;

            float invDist = rsqrtf(dist2);

            float force =
                G * my_mass * tree.mass[nodeIndex] *
                invDist * invDist;

            fx += force * dx * invDist;
            fy += force * dy * invDist;
        }
        else
        {
            int c0 = tree.child0[nodeIndex];
            int c1 = tree.child1[nodeIndex];
            int c2 = tree.child2[nodeIndex];
            int c3 = tree.child3[nodeIndex];

            if(c0 != -1) stack[top++] = c0;
            if(c1 != -1) stack[top++] = c1;
            if(c2 != -1) stack[top++] = c2;
            if(c3 != -1) stack[top++] = c3;
        }
    }

    bodies.vx[i] += DT * fx / my_mass;
    bodies.vy[i] += DT * fy / my_mass;
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

    //std::vector<QuadNode> nodes;
    //nodes.reserve(4*N);

    //QuadNode *d_nodes;

    QuadTree tree;
    int nodeCount;

    tree.cx = (float*)malloc(MAX_NODES*sizeof(float));
    tree.cy = (float*)malloc(MAX_NODES*sizeof(float));
    tree.half = (float*)malloc(MAX_NODES*sizeof(float));

    tree.mass = (float*)malloc(MAX_NODES*sizeof(float));
    tree.com_x = (float*)malloc(MAX_NODES*sizeof(float));
    tree.com_y = (float*)malloc(MAX_NODES*sizeof(float));

    tree.body = (int*)malloc(MAX_NODES*sizeof(int));

    tree.child0 = (int*)malloc(MAX_NODES*sizeof(int));
    tree.child1 = (int*)malloc(MAX_NODES*sizeof(int));
    tree.child2 = (int*)malloc(MAX_NODES*sizeof(int));
    tree.child3 = (int*)malloc(MAX_NODES*sizeof(int));

    tree.leaf = (bool*)malloc(MAX_NODES*sizeof(bool));

    QuadTree d_tree;

    cudaMalloc(&d_tree.cx, MAX_NODES*sizeof(float));
    cudaMalloc(&d_tree.cy, MAX_NODES*sizeof(float));
    cudaMalloc(&d_tree.half, MAX_NODES*sizeof(float));

    cudaMalloc(&d_tree.mass, MAX_NODES*sizeof(float));
    cudaMalloc(&d_tree.com_x, MAX_NODES*sizeof(float));
    cudaMalloc(&d_tree.com_y, MAX_NODES*sizeof(float));

    cudaMalloc(&d_tree.body, MAX_NODES*sizeof(int));

    cudaMalloc(&d_tree.child0, MAX_NODES*sizeof(int));
    cudaMalloc(&d_tree.child1, MAX_NODES*sizeof(int));
    cudaMalloc(&d_tree.child2, MAX_NODES*sizeof(int));
    cudaMalloc(&d_tree.child3, MAX_NODES*sizeof(int));

    cudaMalloc(&d_tree.leaf, MAX_NODES*sizeof(bool));

    for(int step=0;step<STEPS;step++)
    {
        //QuadNode root;

        tree.cx[0]=0.5f;
        tree.cy[0]=0.5f;
        tree.half[0]=0.5f;

        tree.mass[0]=0;
        tree.com_x[0]=0;
        tree.com_y[0]=0;

        tree.body[0]=-1;
        tree.leaf[0]=true;

        tree.child0[0]=-1;
        tree.child1[0]=-1;
        tree.child2[0]=-1;
        tree.child3[0]=-1;

        
        //nodes.push_back(root);
        nodeCount = 1;
        
        for(int i=0;i<N;i++){
            //std::cout << "Reaching here " << i << std::endl;
            insert_body(tree, gpu_bodies,0,i, nodeCount);
            //std::cout << "Tree nodes: " << nodes.size() << std::endl;
        }

        
        compute_mass(tree,gpu_bodies,0);

        //cudaMemcpy(d_nodes,
        //           nodes.data(),
        //           nodes.size()*sizeof(QuadNode),
        //           cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_tree.cx, tree.cx, nodeCount*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tree.cy, tree.cy, nodeCount*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tree.half, tree.half, nodeCount*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tree.mass, tree.mass, nodeCount*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tree.com_x, tree.com_x, nodeCount*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tree.com_y, tree.com_y, nodeCount*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tree.body, tree.body, nodeCount*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tree.child0, tree.child0, nodeCount*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tree.child1, tree.child1, nodeCount*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tree.child2, tree.child2, nodeCount*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tree.child3, tree.child3, nodeCount*sizeof(int), cudaMemcpyHostToDevice);
        
        cudaMemcpy(d_tree.leaf, tree.leaf, nodeCount*sizeof(bool), cudaMemcpyHostToDevice);
        
        
        compute_forces_bh<<<blocks,threads>>>(
                gpu_bodies,d_tree);

        update_positions<<<blocks,threads>>>(gpu_bodies);

        cudaDeviceSynchronize();

        //std::cout << gpu_bodies.x[0] << " " << gpu_bodies.y[0] << std::endl;

        
        //save_frame(gpu_bodies,step);

        //nodes.clear();
    }

    //cudaFree(d_nodes);

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