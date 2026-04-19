#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <cassert>
#include <random>
#include <iostream>
#include "kernel_ising_mpi.cuh"
#include <curand_kernel.h>


__global__ void mcMoveKernel(int *input_config_d, int * output_config_d,
                int *upper_ghost_d, int *lower_ghost_d,
                curandState * d_states, int N, int local_N,float temp, int phase, int rank)  {

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
    if (row>=0 && row<local_N && col>=0 && col<N){   // Reads from the global memory into the shared memory for inner sites
        input_ds[ty][tx] = input_config_d[row*N+((col+N)%N)];
    }  
    else if(row==-1){  //  Halo sites at the upper boundary (row=-1)
        input_ds[ty][tx] = upper_ghost_d[((col+N)%N)];
    }
    else if(row==local_N){  //  Halo sites at the lower boundary (row=local_N)
        input_ds[ty][tx] = lower_ghost_d[((col+N)%N)];
    }
    else if(col==-1){  //  Halo sites at the left boundary (col=-1)
        input_ds[ty][tx] = input_config_d[row*N+(N+col)%N];
    }
    else if(col==N){  //  Halo sites at the right boundary (col=N)
        input_ds[ty][tx] = input_config_d[row*N];
    }
    else{ // Halo sites at the corners
        input_ds[ty][tx] = 0;
    }
    __syncthreads();

    // Phase 0
    if (tx > 0 && tx <=  TILE_SIZE && ty>0 && ty <=TILE_SIZE){  // Threads reading Halo and ghost sites do not participate in spin calculations
        if (row< local_N && col <N && (row+col+rank*local_N)%2==phase){  // Guard to prevent memory access that is out of bounds
            // float rn = rand_nums_d[row*N+col];  
            /*
                Random number generation on cuda device
            */
            curandState localState = d_states[row*N+col];
            float rn = curand_uniform(&localState);
            d_states[row*N+col] = localState;
            /* 
                Spin calculation for ising model.    
            */
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

/*
    Kernel function to initialize the random states for each active
    thread, to be used for random number generation
*/

__global__ void initRNG(curandState *states, int local_N, int N, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = local_N * N;
    // Instantiaing the random states for each thread.
    if (id < total) {
        curand_init(seed, id, 0, &states[id]);
    }
}