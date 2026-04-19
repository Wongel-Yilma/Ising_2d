#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cassert>
#include <math.h>
#include "kernel_ising_mpi.cuh"
#include "gpu_ising.h"
#include <curand_kernel.h>


IsingGPU::IsingGPU(int N_, int local_N_): N(N_), local_N(local_N_) {

    config_size = N * local_N * sizeof(int);
    ghost_size  = N * sizeof(int);
    rn_size     = N * local_N *  sizeof(curandState);

    checkCuda(cudaMalloc((void **)&input_config_d, config_size));
    checkCuda(cudaMalloc((void **)&output_config_d, config_size));
    checkCuda(cudaMalloc((void **)&upper_ghost_d, ghost_size));
    checkCuda(cudaMalloc((void **)&lower_ghost_d, ghost_size));
    // checkCuda(cudaMalloc((void **)&rand_nums_d, rn_size));
    checkCuda(cudaMalloc(&d_states, rn_size));
}
IsingGPU::~IsingGPU() {
    checkCuda(cudaFree(input_config_d));
    checkCuda(cudaFree(output_config_d));
    checkCuda(cudaFree(upper_ghost_d));
    checkCuda(cudaFree(lower_ghost_d));
    // checkCuda(cudaFree(rand_nums_d));
    checkCuda(cudaFree(d_states));
}

void IsingGPU::initialize_random_states(int N, int local_N){
    dim3 RnDimGrid(ceil(float(local_N*N)/256),1,1 );
    dim3 RnDimBlock(256, 1, 1);

    initRNG<<<RnDimGrid, RnDimBlock>>>(d_states,local_N, N, 1234UL);
    checkCuda(cudaDeviceSynchronize());
}
        

/*
    Launching the kernel function to perform ising model update per phase
*/
void IsingGPU::launch_kernel_ising(int phase, float temp, int rank){
    dim3 DimGrid(ceil(float(N)/TILE_SIZE),ceil(float(local_N)/TILE_SIZE),1 );
    dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    mcMoveKernel<<<DimGrid, DimBlock>>>(input_config_d, output_config_d, upper_ghost_d, lower_ghost_d, d_states, N, local_N, temp, phase, rank);
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}