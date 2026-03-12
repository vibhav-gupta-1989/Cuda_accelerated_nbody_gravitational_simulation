#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>

#define N 10000
#define WIDTH 800
#define HEIGHT 800
#define STEPS 100
#define DT 0.001f
#define G 1.0f

struct Body{
    float x,y;
    float vx,vy;
    float mass;
};

////////////////////////////////////////////////////////////
// CPU Simulation
////////////////////////////////////////////////////////////

void cpu_simulate(Body *bodies)
{
    for(int step=0; step<STEPS; step++)
    {
        for(int i=0;i<N;i++)
        {
            float fx=0, fy=0;

            for(int j=0;j<N;j++)
            {
                if(i==j) continue;

                float dx = bodies[j].x - bodies[i].x;
                float dy = bodies[j].y - bodies[i].y;

                float dist = sqrt(dx*dx + dy*dy + 1e-9f);
                float force = G * bodies[i].mass * bodies[j].mass /(dist*dist);

                fx += force * dx/dist;
                fy += force * dy/dist;
            }

            bodies[i].vx += DT * fx / bodies[i].mass;
            bodies[i].vy += DT * fy / bodies[i].mass;
        }

        for(int i=0;i<N;i++)
        {
            bodies[i].x += bodies[i].vx * DT;
            bodies[i].y += bodies[i].vy * DT;

            if(bodies[i].x < 0 || bodies[i].x > 1)
            {
                bodies[i].vx *= -1;
            }

            if(bodies[i].y < 0 || bodies[i].y > 1)
            {
                bodies[i].vy *= -1;
            }

        }
    }
}

////////////////////////////////////////////////////////////
// CUDA Force Kernel
////////////////////////////////////////////////////////////

__global__ void compute_forces(Body *bodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;

    float fx=0, fy=0;

    for(int j=0;j<N;j++)
    {
        if(i==j) continue;

        float dx = bodies[j].x - bodies[i].x;
        float dy = bodies[j].y - bodies[i].y;

        float dist = sqrtf(dx*dx + dy*dy + 1e-9f);

        float force = G * bodies[i].mass * bodies[j].mass /(dist*dist);

        fx += force * dx/dist;
        fy += force * dy/dist;
    }

    bodies[i].vx += DT * fx / bodies[i].mass;
    bodies[i].vy += DT * fy / bodies[i].mass;
}

////////////////////////////////////////////////////////////

__global__ void update_positions(Body *bodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;

    bodies[i].x += bodies[i].vx * DT;
    bodies[i].y += bodies[i].vy * DT;

    if(bodies[i].x < 0 || bodies[i].x > 1) bodies[i].vx *= -1;
    if(bodies[i].y < 0 || bodies[i].y > 1) bodies[i].vy *= -1;
}

////////////////////////////////////////////////////////////
// Save Frame as Image
////////////////////////////////////////////////////////////

void save_frame(Body *bodies, int frame)
{
    //std::cout << "Reaching here" << std::endl;
    
    std::vector<int> image(WIDTH*HEIGHT*3,0);

    for(int i=0;i<N;i++)
    {
        int x = bodies[i].x * WIDTH;
        int y = bodies[i].y * HEIGHT;

        if(x>=0 && x<WIDTH && y>=0 && y<HEIGHT)
        {
            int index = (y*WIDTH + x)*3;
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
    {
        for(int i=0;i<WIDTH;i++)
        {
            int index = (j*WIDTH + i)*3;
            file<<image[index]<<" "
                <<image[index+1]<<" "
                <<image[index+2]<<"\n";
        }
    }

    file.close();
}

////////////////////////////////////////////////////////////

int main()
{
    
    Body *cpu_bodies = new Body[N];
    Body *gpu_bodies;

    cudaMallocManaged(&gpu_bodies, N*sizeof(Body));

    // Initialize particles
    for(int i=0;i<N;i++)
    {
        cpu_bodies[i].x = rand()/float(RAND_MAX);
        cpu_bodies[i].y = rand()/float(RAND_MAX);

        cpu_bodies[i].vx = cpu_bodies[i].vy = 0;
        cpu_bodies[i].mass = 1;

        gpu_bodies[i] = cpu_bodies[i];
    }

    //////////////////////////////////////////
    // CPU Timing
    //////////////////////////////////////////

    auto cpu_start = std::chrono::high_resolution_clock::now();

    cpu_simulate(cpu_bodies);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    double cpu_time =
        std::chrono::duration<double>(cpu_end-cpu_start).count();

    //////////////////////////////////////////
    // GPU Timing
    //////////////////////////////////////////

    int threads = 256;
    int blocks = (N+threads-1)/threads;

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(int step=0;step<STEPS;step++)
    {
        compute_forces<<<blocks,threads>>>(gpu_bodies);
        cudaDeviceSynchronize();
	
        update_positions<<<blocks,threads>>>(gpu_bodies);
        cudaDeviceSynchronize();
	
        
        save_frame(gpu_bodies,step);
	//std::cout << "Reaching here" << std::endl;
    }

    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time_ms;
    cudaEventElapsedTime(&gpu_time_ms,start,stop);

    double gpu_time = gpu_time_ms/1000.0;

    //////////////////////////////////////////
    // Speedup
    //////////////////////////////////////////

    std::cout<<"CPU Time: "<<cpu_time<<" seconds\n";
    std::cout<<"GPU Time: "<<gpu_time<<" seconds\n";

    std::cout<<"Speedup: "<<cpu_time/gpu_time<<"x\n";

    cudaFree(gpu_bodies);
    delete[] cpu_bodies;
}