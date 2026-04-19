// gpu_ising.h
#pragma once

#include "kernel_ising_mpi.cuh"  // TILE_SIZE, BLOCK_SIZE, checkCuda, mcMoveKernel
#include <curand_kernel.h>

class IsingGPU {
public:
    int *input_config_d = nullptr;
    int *output_config_d = nullptr;
    int *upper_ghost_d = nullptr;
    int *lower_ghost_d = nullptr;
    // float *rand_nums_d = nullptr;
    curandState * d_states;

    int N = 0;
    int local_N = 0;
    int config_size = 0;
    int ghost_size = 0;
    int rn_size = 0;

    explicit IsingGPU(int N_, int local_N_);
    ~IsingGPU();
    void initialize_random_states(int N, int local_N);
    void launch_kernel_ising(int phase, float temp, int rank);
};