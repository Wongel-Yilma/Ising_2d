#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <cassert>
#include <random>
#include <iostream>

__global__ void mcMoveKernel(int *input_config_d, int * output_config_d, int* final_config_d,float * rand_nums_d, int N,float temp, int phase){

    // Each thread handles one lattice site
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    // Create a shared memory 
    int spin;
    float rn;
    float neigh_sum;

    if (row< N && col <N){  // Guard to prevent memory access that is out of bounds
        // Phase 1
        if (((row+col)%2==phase)){
            spin = input_config_d[row*N+col];
            rn  = rand_nums_d[row*N+col];
            neigh_sum = input_config_d[((row-1+N)%N)*N+col] + 
                        input_config_d[N*((row+1)%N)+col] + 
                        input_config_d[row*N+(col-1+N)%N] + 
                        input_config_d[row*N+(col+1)%N];
            float ediff = 2.0f*spin*neigh_sum;
            if (ediff<0.0|| rn< expf(-ediff/temp)  ){
                spin*=-1;
            }
            output_config_d[row*N+col]= spin;
            final_config_d[row*N+col]= spin;

        }
    }
}
inline cudaError_t checkCuda(cudaError_t result){
    if (result !=cudaSuccess){
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result==cudaSuccess);
    }
    return result;
}

extern "C" void launch_kernel_ising(int * input_config_h, int * output_config_h, int N, float temp, int nsteps, std::mt19937 &gen){
    
    int threadsPerBlock = 16;
    int config_size = N*N*sizeof(int);
    int rn_size = N*N*sizeof(float);
    float *rand_nums_h=NULL; 
    float *rand_nums_d=NULL;

    // Creating device variables

    int *input_config_d, *output_config_d, *temp_d;

    rand_nums_h = (float *)malloc(rn_size);

    // std::cout<<"Random numbers created on the host"<<std::endl;
    // Allocating memory for the device variables
    checkCuda(cudaMalloc((void **)&input_config_d, config_size));
    checkCuda(cudaMalloc((void **)&output_config_d, config_size));
    checkCuda(cudaMalloc((void **)&temp_d, config_size));
    checkCuda(cudaMalloc((void **)&rand_nums_d, rn_size));
    // std::cout<<"Memory allocated on the device"<<std::endl;
    // Copying data from host to device
    checkCuda(cudaMemcpy(input_config_d, input_config_h, config_size, cudaMemcpyHostToDevice));
    
    
    // Calculating the block and grid sizes
    dim3 DimGrid(ceil(float(N)/threadsPerBlock),ceil(float(N)/threadsPerBlock),1 );
    dim3 DimBlock(threadsPerBlock, threadsPerBlock, 1);

    // Executing with 2 phases --> Checkerboard scheme

    std::uniform_real_distribution<float> real_distr(0.0, 1.0);
    for (int s=0; s<nsteps; s++){
        checkCuda(cudaMemcpy(temp_d, input_config_d, config_size, cudaMemcpyDeviceToDevice));
        for (int i=0; i<N*N; i++) rand_nums_h[i] = real_distr(gen);
        checkCuda(cudaMemcpy(rand_nums_d, rand_nums_h, rn_size, cudaMemcpyHostToDevice));
        /*
            Phase 0:
                input_config_d ->input  
                temp_config_d -> temporary output
                output_config_d -> FINAL output
        */
        mcMoveKernel<<<DimGrid, DimBlock>>>(input_config_d, temp_d, output_config_d, rand_nums_d, N, temp, 0);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        /*
            Phase 1:
                temp_config_d ->input
                output_config_d -> output
                output_config_d -> FINAL output
        */
        mcMoveKernel<<<DimGrid, DimBlock>>>(temp_d, output_config_d,output_config_d, rand_nums_d, N, temp, 1);
        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaMemcpy(output_config_h, output_config_d, config_size, cudaMemcpyDeviceToHost));
        std::swap(input_config_d, output_config_d);

    }
    // int sum = 0;
    // for (int i=0; i< N*N; i++) sum+= output_config_h[i];
    // printf("Total sum of the output in the cuda file %d\n", sum);
    cudaFree(input_config_d);
    cudaFree(output_config_d);
    cudaFree(rand_nums_d);
    cudaFree(temp_d);
    free(rand_nums_h);

}