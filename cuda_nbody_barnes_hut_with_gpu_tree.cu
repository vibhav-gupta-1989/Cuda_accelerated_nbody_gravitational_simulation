#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define N 200000
#define BLOCK 256
#define STEPS 20

#define WIDTH 800
#define HEIGHT 800

#define DT 0.001f
#define G 1.0f
#define SOFTENING 1e-9f

////////////////////////////////////////////////////////////
// Bodies
////////////////////////////////////////////////////////////

struct Bodies{
    float *x;
    float *y;
    float *vx;
    float *vy;
    float *mass;
};

////////////////////////////////////////////////////////////
// Node
////////////////////////////////////////////////////////////

struct Node{
    float mass;
    float com_x;
    float com_y;

    int left;
    int right;
    int parent;

    int body;
};

////////////////////////////////////////////////////////////
// Morton utilities
////////////////////////////////////////////////////////////

__device__ unsigned int expandBits(unsigned int v)
{
    v=(v*0x00010001u)&0xFF0000FFu;
    v=(v*0x00000101u)&0x0F00F00Fu;
    v=(v*0x00000011u)&0xC30C30C3u;
    v=(v*0x00000005u)&0x49249249u;
    return v;
}

__device__ unsigned int morton2D(float x,float y)
{
    x=fminf(fmaxf(x*1024.0f,0.0f),1023.0f);
    y=fminf(fmaxf(y*1024.0f,0.0f),1023.0f);

    unsigned int xx=expandBits((unsigned int)x);
    unsigned int yy=expandBits((unsigned int)y);

    return (xx<<1)|yy;
}

////////////////////////////////////////////////////////////
// Morton kernel
////////////////////////////////////////////////////////////

__global__ void computeMorton(
    float *x,
    float *y,
    unsigned int *codes,
    int *indices)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;

    codes[i]=morton2D(x[i],y[i]);
    indices[i]=i;
}

////////////////////////////////////////////////////////////
// Build leaves
////////////////////////////////////////////////////////////

__global__ void buildLeaves(
    Node *nodes,
    int *indices,
    float *x,
    float *y,
    float *mass)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;

    int id=indices[i];

    nodes[i].body=id;
    nodes[i].left=-1;
    nodes[i].right=-1;
    nodes[i].parent=-1;

    nodes[i].mass=mass[id];
    nodes[i].com_x=x[id];
    nodes[i].com_y=y[id];
}

////////////////////////////////////////////////////////////
// Internal nodes
////////////////////////////////////////////////////////////

__global__ void buildInternal(
    Node *nodes)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N-1) return;

    int left=i;
    int right=i+1;

    nodes[N+i].left=left;
    nodes[N+i].right=right;

    nodes[left].parent=N+i;
    nodes[right].parent=N+i;

    nodes[N+i].body=-1;
}

////////////////////////////////////////////////////////////
// Center of mass
////////////////////////////////////////////////////////////

__global__ void computeMass(Node *nodes)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N-1) return;

    int idx=N+i;

    Node left=nodes[nodes[idx].left];
    Node right=nodes[nodes[idx].right];

    float m=left.mass+right.mass;

    nodes[idx].mass=m;

    nodes[idx].com_x=
        (left.com_x*left.mass+
         right.com_x*right.mass)/m;

    nodes[idx].com_y=
        (left.com_y*left.mass+
         right.com_y*right.mass)/m;
}

////////////////////////////////////////////////////////////
// Barnes Hut force kernel
////////////////////////////////////////////////////////////

__global__ void computeForces(
    Bodies bodies,
    Node *nodes)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;

    float x=bodies.x[i];
    float y=bodies.y[i];
    float m=bodies.mass[i];

    float fx=0;
    float fy=0;

    int stack[64];
    int top=0;

    stack[top++]=2*N-2;

    while(top>0)
    {
        int node=stack[--top];

        if(nodes[node].body==i) continue;

        float dx=nodes[node].com_x-x;
        float dy=nodes[node].com_y-y;

        float dist2=dx*dx+dy*dy+SOFTENING;

        if(nodes[node].left==-1)
        {
            float invDist=rsqrtf(dist2);

            float force=
                G*m*nodes[node].mass*
                invDist*invDist;

            fx+=force*dx*invDist;
            fy+=force*dy*invDist;
        }
        else
        {
            stack[top++]=nodes[node].left;
            stack[top++]=nodes[node].right;
        }
    }

    bodies.vx[i]+=DT*fx/m;
    bodies.vy[i]+=DT*fy/m;
}

////////////////////////////////////////////////////////////
// Update positions
////////////////////////////////////////////////////////////

__global__ void update(Bodies bodies)
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
// Frame output
////////////////////////////////////////////////////////////

void save_frame(float *x,float *y,int frame)
{
    std::vector<int> img(WIDTH*HEIGHT*3,0);

    for(int i=0;i<N;i++)
    {
        int px=x[i]*WIDTH;
        int py=y[i]*HEIGHT;

        if(px>=0 && px<WIDTH && py>=0 && py<HEIGHT)
        {
            int idx=(py*WIDTH+px)*3;

            img[idx]=255;
            img[idx+1]=255;
            img[idx+2]=255;
        }
    }

    char name[64];
    sprintf(name,"frame_%04d.ppm",frame);

    std::ofstream f(name);

    f<<"P3\n"<<WIDTH<<" "<<HEIGHT<<"\n255\n";

    for(int i=0;i<WIDTH*HEIGHT*3;i+=3)
        f<<img[i]<<" "<<img[i+1]<<" "<<img[i+2]<<"\n";
}

////////////////////////////////////////////////////////////
// CPU brute force
////////////////////////////////////////////////////////////

void cpu_simulate(
    float *x,float *y,
    float *vx,float *vy,
    float *mass)
{
    for(int step=0;step<STEPS;step++)
    {
        for(int i=0;i<N;i++)
        {
            float fx=0;
            float fy=0;

            for(int j=0;j<N;j++)
            {
                if(i==j) continue;

                float dx=x[j]-x[i];
                float dy=y[j]-y[i];

                float dist2=dx*dx+dy*dy+SOFTENING;

                float invDist=1.0f/sqrt(dist2);

                float force=
                    G*mass[i]*mass[j]*
                    invDist*invDist;

                fx+=force*dx*invDist;
                fy+=force*dy*invDist;
            }

            vx[i]+=DT*fx/mass[i];
            vy[i]+=DT*fy/mass[i];
        }

        for(int i=0;i<N;i++)
        {
            x[i]+=vx[i]*DT;
            y[i]+=vy[i]*DT;

            if(x[i] < 0 || x[i] > 1){
                vx[i] *= -1.0;
            }

            if(y[i] < 0 || y[i] > 1){
                vy[i] *= -1.0;
            }
        }
    }
}

////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////

int main()
{
    float *hx=new float[N];
    float *hy=new float[N];
    float *hm=new float[N];

    float *cpu_x=new float[N];
    float *cpu_y=new float[N];
    float *cpu_vx=new float[N];
    float *cpu_vy=new float[N];
    float *cpu_mass=new float[N];

    for(int i=0;i<N;i++)
    {
        hx[i]=rand()/float(RAND_MAX);
        hy[i]=rand()/float(RAND_MAX);
        hm[i]=1;

        cpu_x[i]=hx[i];
        cpu_y[i]=hy[i];
        cpu_vx[i]=0;
        cpu_vy[i]=0;
        cpu_mass[i]=1;
    }

    ////////////////////////////////////////////////////////////
    // CPU timing
    ////////////////////////////////////////////////////////////

    auto cpu_start=
        std::chrono::high_resolution_clock::now();

    //cpu_simulate(cpu_x,cpu_y,cpu_vx,cpu_vy,cpu_mass);

    auto cpu_end=
        std::chrono::high_resolution_clock::now();

    double cpu_time=
        std::chrono::duration<double>(cpu_end-cpu_start).count();

    ////////////////////////////////////////////////////////////
    // GPU memory
    ////////////////////////////////////////////////////////////

    Bodies bodies;

    cudaMalloc(&bodies.x,N*sizeof(float));
    cudaMalloc(&bodies.y,N*sizeof(float));
    cudaMalloc(&bodies.vx,N*sizeof(float));
    cudaMalloc(&bodies.vy,N*sizeof(float));
    cudaMalloc(&bodies.mass,N*sizeof(float));

    cudaMemcpy(bodies.x,hx,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(bodies.y,hy,N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(bodies.mass,hm,N*sizeof(float),cudaMemcpyHostToDevice);

    unsigned int *d_morton;
    int *d_indices;

    cudaMalloc(&d_morton,N*sizeof(unsigned int));
    cudaMalloc(&d_indices,N*sizeof(int));

    Node *d_nodes;
    cudaMalloc(&d_nodes,(2*N)*sizeof(Node));

    int blocks=(N+BLOCK-1)/BLOCK;

    ////////////////////////////////////////////////////////////
    // GPU timing
    ////////////////////////////////////////////////////////////

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int step=0;step<STEPS;step++)
    {
        computeMorton<<<blocks,BLOCK>>>(
            bodies.x,bodies.y,
            d_morton,d_indices);

        thrust::device_ptr<unsigned int> m(d_morton);
        thrust::device_ptr<int> idx(d_indices);

        thrust::sort_by_key(m,m+N,idx);

        buildLeaves<<<blocks,BLOCK>>>(
            d_nodes,d_indices,
            bodies.x,bodies.y,
            bodies.mass);

        buildInternal<<<blocks,BLOCK>>>(d_nodes);

        computeMass<<<blocks,BLOCK>>>(d_nodes);

        computeForces<<<blocks,BLOCK>>>(
            bodies,d_nodes);

        update<<<blocks,BLOCK>>>(bodies);

        cudaDeviceSynchronize();

        cudaMemcpy(hx,bodies.x,
                   N*sizeof(float),
                   cudaMemcpyDeviceToHost);

        cudaMemcpy(hy,bodies.y,
                   N*sizeof(float),
                   cudaMemcpyDeviceToHost);

        save_frame(hx,hy,step);
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

    std::cout<<"\nFrames generated.\n";
    std::cout<<"Create video with:\n";
    std::cout<<"ffmpeg -framerate 30 -i frame_%04d.ppm -pix_fmt yuv420p nbody.mp4\n";
}
