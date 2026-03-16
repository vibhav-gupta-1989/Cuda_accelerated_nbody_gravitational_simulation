#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <fstream>

#define N 200000
#define BLOCK 256

#define WIDTH 800
#define HEIGHT 800

#define DT 0.001f
#define G 0.0001f
#define SOFTENING 1e-6f
#define THETA 0.5f

#define STEPS 5

////////////////////////////////////////////////////////////
struct Bodies{
    float *x;
    float *y;
    float *vx;
    float *vy;
    float *mass;
};

struct Node{
    float mass;
    float com_x;
    float com_y;

    int left;
    int right;
    int parent;

    int body;
};

Bodies bodies;

GLuint vbo;
cudaGraphicsResource* cuda_vbo_resource;

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
__global__ void buildInternal(Node *nodes)
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

        float size=0.01f;

        if(nodes[node].left==-1 || size*size/dist2 < THETA*THETA)
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
__global__ void update(Bodies bodies)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;

    bodies.x[i]+=bodies.vx[i]*DT;
    bodies.y[i]+=bodies.vy[i]*DT;

    if(bodies.x[i]<0){ bodies.x[i]=0; bodies.vx[i]*=-1; }
    if(bodies.x[i]>1){ bodies.x[i]=1; bodies.vx[i]*=-1; }

    if(bodies.y[i]<0){ bodies.y[i]=0; bodies.vy[i]*=-1; }
    if(bodies.y[i]>1){ bodies.y[i]=1; bodies.vy[i]*=-1; }
}

////////////////////////////////////////////////////////////
__global__ void copyToVBO(
float *x,
float *y,
float2 *pos)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=N) return;

    pos[i]=make_float2(
        x[i]*2.0f-1.0f,
        y[i]*2.0f-1.0f);
}

////////////////////////////////////////////////////////////
void createVBO()
{
    glGenBuffers(1,&vbo);

    glBindBuffer(GL_ARRAY_BUFFER,vbo);

    glBufferData(
        GL_ARRAY_BUFFER,
        N*sizeof(float2),
        0,
        GL_DYNAMIC_DRAW);

    cudaGraphicsGLRegisterBuffer(
        &cuda_vbo_resource,
        vbo,
        cudaGraphicsMapFlagsWriteDiscard);
}

////////////////////////////////////////////////////////////
void render()
{
    glClear(GL_COLOR_BUFFER_BIT);

    glBindBuffer(GL_ARRAY_BUFFER,vbo);

    glEnableClientState(GL_VERTEX_ARRAY);

    glVertexPointer(2,GL_FLOAT,0,0);

    glDrawArrays(GL_POINTS,0,N);

    glDisableClientState(GL_VERTEX_ARRAY);
}

////////////////////////////////////////////////////////////
void stepSimulation(
Node *d_nodes,
unsigned int *d_morton,
int *d_indices)
{
    int blocks=(N+BLOCK-1)/BLOCK;

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
}

void cudaGraphicsWork(){
    int blocks=(N+BLOCK-1)/BLOCK;
    cudaGraphicsMapResources(1,&cuda_vbo_resource);

    float2 *dptr;
    size_t num_bytes;

    cudaGraphicsResourceGetMappedPointer(
        (void**)&dptr,
        &num_bytes,
        cuda_vbo_resource);

    copyToVBO<<<blocks,BLOCK>>>(
        bodies.x,
        bodies.y,
        dptr);

    cudaGraphicsUnmapResources(
        1,
        &cuda_vbo_resource);
}

void cpuStep(
std::vector<float>& x,
std::vector<float>& y,
std::vector<float>& vx,
std::vector<float>& vy,
std::vector<float>& mass)
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

            float dist2=
            dx*dx+dy*dy+SOFTENING;

            float inv=sqrt(1.0f/dist2);

            float force=
            G*mass[i]*mass[j]/dist2;

            fx+=force*dx*inv;
            fy+=force*dy*inv;
        }

        vx[i]+=DT*fx/mass[i];
        vy[i]+=DT*fy/mass[i];
    }

    for(int i=0;i<N;i++)
    {
        x[i]+=vx[i]*DT;
        y[i]+=vy[i]*DT;

        if(x[i]<0){ x[i]=0; vx[i]*=-1; }
        if(x[i]>1){ x[i]=1; vx[i]*=-1; }

        if(y[i]<0){ y[i]=0; vy[i]*=-1; }
        if(y[i]>1){ y[i]=1; vy[i]*=-1; }
    }

    
}

////////////////////////////////////////////////////////////
int main()
{
    glfwInit();

    GLFWwindow* window=
        glfwCreateWindow(
            WIDTH,
            HEIGHT,
            "CUDA Barnes-Hut NBody",
            NULL,
            NULL);

    glfwMakeContextCurrent(window);

    glewInit();

    glPointSize(1);

    std::vector<float> hx(N);
    std::vector<float> hy(N);
    std::vector<float> hvx(N);
    std::vector<float> hvy(N);
    std::vector<float> hm(N,1);

    std::vector<float> cpu_x(N);
    std::vector<float> cpu_y(N);
    std::vector<float> cpu_vx(N);
    std::vector<float> cpu_vy(N);
    std::vector<float> cpu_mass(N,1);

    srand(time(NULL));

    for(int i=0;i<N;i++)
    {
        hx[i]=rand()/float(RAND_MAX);
        hy[i]=rand()/float(RAND_MAX);

        hvx[i]=(rand()/float(RAND_MAX)-0.5f)*0.01f;
        hvy[i]=(rand()/float(RAND_MAX)-0.5f)*0.01f;

        cpu_x[i] = hx[i];
        cpu_y[i] = hy[i];
        cpu_vx[i] = hvx[i];
        cpu_vy[i] = hvy[i];
    }

    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0; i<STEPS; i++)
        cpuStep(cpu_x,cpu_y,cpu_vx,cpu_vy,cpu_mass);

    auto end = std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double,std::milli>(end-start).count();

    cudaMalloc(&bodies.x,N*sizeof(float));
    cudaMalloc(&bodies.y,N*sizeof(float));
    cudaMalloc(&bodies.vx,N*sizeof(float));
    cudaMalloc(&bodies.vy,N*sizeof(float));
    cudaMalloc(&bodies.mass,N*sizeof(float));

    cudaMemcpy(bodies.x,hx.data(),N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(bodies.y,hy.data(),N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(bodies.vx,hvx.data(),N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(bodies.vy,hvy.data(),N*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(bodies.mass,hm.data(),N*sizeof(float),cudaMemcpyHostToDevice);

    Node *d_nodes;
    cudaMalloc(&d_nodes,(2*N)*sizeof(Node));

    unsigned int *d_morton;
    int *d_indices;

    cudaMalloc(&d_morton,N*sizeof(unsigned int));
    cudaMalloc(&d_indices,N*sizeof(int));

    createVBO();

    double ms = 0;

    for(int i=0; i<STEPS; i++){
        
        start = std::chrono::high_resolution_clock::now();

        stepSimulation(
            d_nodes,
            d_morton,
            d_indices);
        
        cudaDeviceSynchronize();

        end = std::chrono::high_resolution_clock::now();
        
        ms += std::chrono::duration<double,std::milli>(end-start).count();
        
        cudaGraphicsWork();
        
        cudaDeviceSynchronize();
    }
    
    double speedup = cpu_ms / ms;
    std::cout << speedup << std::endl;

    while(!glfwWindowShouldClose(window))
    {
        stepSimulation(
            d_nodes,
            d_morton,
            d_indices);
        
        cudaGraphicsWork();

        render();

        glfwSwapBuffers(window);

        glfwPollEvents();
    }

    glfwTerminate();
}