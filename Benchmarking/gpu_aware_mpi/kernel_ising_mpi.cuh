// kernel_ising_mpi.cuh
#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdio>
#include <cassert>

#define TILE_SIZE 16
#define BLOCK_SIZE (TILE_SIZE + 2)

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

// Kernel declaration (definition stays in kernel_ising_mpi.cu)
__global__ void mcMoveKernel(
    int* input_config_d,
    int* output_config_d,
    int* upper_ghost_d,
    int* lower_ghost_d,
    // float* rand_nums_d,
    curandState * d_states,
    int N,
    int local_N,
    float temp,
    int phase, 
    int rank
);

// Kernel declaration for random state initialization

__global__ void initRNG(curandState *states, int local_N, int N,  unsigned long seed);