#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <cassert>

__global__ void mcMoveKernel(int *input_config_d, int * output_config_d, float * rand_nums_d, int N, int phase, float temp){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int spin;
    float rn, ediff;
    float neigh_sum;

    if (row< N && col <N){  // Guard to prevent memory access that is out of bounds
        if (((phase+row+col)%2==0)){
            spin = input_config_d[row*N+col];
            rn  = rand_nums_d[row*N+col];
            neigh_sum = input_config_d[((row-1+N)%N)*N+col] + 
                        input_config_d[N*((row+1)%N)+col] + 
                        input_config_d[row*N+(col-1+N)%N] + 
                        input_config_d[row*N+(col+1)%N];
            ediff = 2*spin*neigh_sum;
            if (ediff<0.0|| rn< exp(-ediff/temp)  ){
                spin*=-1;
            }
            output_config_d[row*N+col]= spin;
        }
        else  {
            output_config_d[row*N+col]= 0;
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

extern "C" void launch_kernel_ising(int * input_config_h, int * output_config_h, float *rand_nums_h, int N, float temp){
    
    int threadsPerBlock = 16;
    int size_config = N*N*sizeof(int);
    int size_rn = N*N*sizeof(float);

    
    // Creating device variables
    int *input_config_d, *output_config_d;
    float *rand_nums_d;

    // Allocating memory for the device variables
    checkCuda(cudaMalloc((void **)&input_config_d, size_config));
    checkCuda(cudaMalloc((void **)&output_config_d, size_config));
    checkCuda(cudaMalloc((void **)&rand_nums_d, size_rn));
    // Copying data from host to device
    checkCuda(cudaMemcpy(input_config_d, input_config_h, size_config, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(rand_nums_d, rand_nums_h, size_rn, cudaMemcpyHostToDevice));
    
    // Calculating the block and grid sizes
    dim3 DimGrid((N+threadsPerBlock-1)/threadsPerBlock,(N+threadsPerBlock-1)/threadsPerBlock,1 );
    dim3 DimBlock(threadsPerBlock, threadsPerBlock, 1);

    // Executing with 2 phases --> Checkerboard scheme
    int phase;
    // First phase execution
    phase = 0;


    mcMoveKernel<<<DimGrid, DimBlock>>>(input_config_d, output_config_d, rand_nums_d, N, phase, temp);
    cudaDeviceSynchronize();
    // Copying the result of the first phase to perform the second phase;
    checkCuda(cudaMemcpy(input_config_d, output_config_d,size_config , cudaMemcpyDeviceToDevice));
    // Second phase execution
    phase = 1;
    mcMoveKernel<<<DimGrid, DimBlock>>>(input_config_d, output_config_d, rand_nums_d, N, phase, temp);
    cudaDeviceSynchronize();
    checkCuda(cudaMemcpy(output_config_h, output_config_d, size_config, cudaMemcpyDeviceToHost));

    int sum = 0;
    for (int i=0; i< N*N; i++) sum+= output_config_h[i];
    printf("Total sum of the output in the cuda file %d\n", sum);
    cudaFree(input_config_d);
    cudaFree(output_config_d);
    cudaFree(rand_nums_d);

}