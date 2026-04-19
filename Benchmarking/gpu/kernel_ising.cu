#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <cassert>
#include <random>
#include <iostream>
#include <curand_kernel.h>

#define TILE_SIZE 4

#define BLOCK_SIZE (TILE_SIZE+2)

// __global__ void mcMoveKernel(int *input_config_d, int * output_config_d,float * rand_nums_d, int N,float temp, int phase){
__global__ void mcMoveKernel(int *input_config_d, int * output_config_d, curandState *d_states,  int N,float temp, int phase){

    // Each thread loads spin for one lattice site and its random number into a shared memory

    __shared__  int input_ds[BLOCK_SIZE][BLOCK_SIZE];

    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global lattice site indices
    int row = by*TILE_SIZE + ty -1;
    int col = bx*TILE_SIZE + tx -1;

    // Load the spin and random number into the shared memory
    if (row>=0 && row<N && col>=0 && col<N){   // Reads from the global memory into the shared memory for inner sites
        input_ds[ty][tx] = input_config_d[((row+N)%N)*N+((col+N)%N) ];
    }  
    else if(row==-1){  //  Halo sites at the upper boundary (row=-1)
        input_ds[ty][tx] = input_config_d[((N+row)%N)*N+((col+N)%N)];
    }
    else if(row==N){  //  Halo sites at the lower boundary (row=N)
        input_ds[ty][tx] = input_config_d[(col+N)%N];
    }
    else if(col==-1){  //  Halo sites at the left boundary (col=-1)
        input_ds[ty][tx] = input_config_d[((row+N)%N)*N+(N+col)%N];
    }
    else if(col==N){  //  Halo sites at the right boundary (col=N)
        input_ds[ty][tx] = input_config_d[((row+N)%N)*N];
    }
    else{ // Halo sites at the corners
        input_ds[ty][tx] = 0;
    }
    __syncthreads();

    // Phase 0
    if (tx > 0 && tx <=  TILE_SIZE && ty>0 &&ty <=TILE_SIZE){  // Threads reading Halo sites do not participate in spin calculations
        if (row< N && col <N && (row+col)%2==phase){  // Guard to prevent memory access that is out of bounds
            // float rn = rand_nums_d[row*N+col];  
            curandState localState = d_states[row*N+col];
            float rn = curand_uniform(&localState);
            d_states[row*N+col] = localState;
            int spin = input_ds[ty][tx];
            int neigh_sum = input_ds[ty-1][tx] + input_ds[ty+1][tx] + input_ds[ty][tx-1] + input_ds[ty][tx+1];
            float ediff = 2.0f*spin*neigh_sum;
            if (ediff<0.0|| rn< expf(-ediff/temp)  ){
                spin*=-1;
            }
            output_config_d[row*N+col]= spin;
        }
    }
    

}

__global__ void initRNG(curandState *states, int N, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;

    if (id < total) {
        curand_init(seed, id, 0, &states[id]);
    }
}


inline cudaError_t checkCuda(cudaError_t result){
    if (result !=cudaSuccess){
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result==cudaSuccess);
    }
    return result;
}



extern "C" void launch_kernel_ising(int * input_config_h, int * output_config_h, int N, float temp, int nsteps, std::mt19937 *gen_ptr){
    
    int threadsPerBlock = BLOCK_SIZE;
    int config_size = N*N*sizeof(int);
    // int rn_size = N*N*sizeof(float);
    // float *rand_nums_h=NULL; 
    // float *rand_nums_d=NULL;
    curandState *d_states;
    checkCuda(cudaMalloc(&d_states, N * N * sizeof(curandState)));
    // Creating device variables

    int *input_config_d, *output_config_d;

    dim3 RnDimGrid(ceil(float(N*N)/256),1,1 );
    dim3 RnDimBlock(256, 1, 1);

    initRNG<<<RnDimGrid, RnDimBlock>>>(d_states, N, 1234UL);
    checkCuda(cudaDeviceSynchronize());

    // rand_nums_h = (float *)malloc(rn_size);

    // std::cout<<"Random numbers created on the host"<<std::endl;
    // Allocating memory for the device variables
    checkCuda(cudaMalloc((void **)&input_config_d, config_size));
    checkCuda(cudaMalloc((void **)&output_config_d, config_size));
    // checkCuda(cudaMalloc((void **)&rand_nums_d, rn_size));
    // std::cout<<"Memory allocated on the device"<<std::endl;
    // Copying data from host to device
    checkCuda(cudaMemcpy(input_config_d, input_config_h, config_size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(output_config_d, input_config_d, config_size, cudaMemcpyDeviceToDevice));
    
    
    // Calculating the block and grid sizes
    dim3 DimGrid(ceil(float(N)/TILE_SIZE),ceil(float(N)/TILE_SIZE),1 );
    dim3 DimBlock(threadsPerBlock, threadsPerBlock, 1);

    // Executing with 2 phases --> Checkerboard scheme

    // std::uniform_real_distribution<float> real_distr(0.0, 1.0);

    for (int s=0; s<nsteps; s++){
        // for (int i=0; i<N*N; i++) rand_nums_h[i] = real_distr(*gen_ptr); 
        // checkCuda(cudaMemcpy(rand_nums_d, rand_nums_h, rn_size, cudaMemcpyHostToDevice));
        /*
            Phase 0
        */
        // mcMoveKernel<<<DimGrid, DimBlock>>>(input_config_d, output_config_d, rand_nums_d, N, temp, 0);
        mcMoveKernel<<<DimGrid, DimBlock>>>(input_config_d, output_config_d,d_states, N, temp, 0);

        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        // std::swap(input_config_d, output_config_d);
        checkCuda(cudaMemcpy(input_config_d, output_config_d, config_size, cudaMemcpyDeviceToDevice));
        /*
            Phase 1
        */
        // mcMoveKernel<<<DimGrid, DimBlock>>>(input_config_d, output_config_d, rand_nums_d, N, temp, 1);
        mcMoveKernel<<<DimGrid, DimBlock>>>(input_config_d, output_config_d, d_states, N, temp, 1);

        checkCuda(cudaGetLastError());
        checkCuda(cudaDeviceSynchronize());
        // std::swap(input_config_d, output_config_d);
        checkCuda(cudaMemcpy(input_config_d, output_config_d, config_size, cudaMemcpyDeviceToDevice));

    }
    checkCuda(cudaMemcpy(output_config_h, input_config_d, config_size, cudaMemcpyDeviceToHost));
    // int sum = 0;
    // for (int i=0; i< N*N; i++) sum+= output_config_h[i];
    // printf("Total sum of the output in the cuda file %d\n", sum);
    checkCuda(cudaFree(input_config_d));
    checkCuda(cudaFree(output_config_d));
    // checkCuda(cudaFree(rand_nums_d));
    // free(rand_nums_h);
    cudaFree(d_states);

}
