cmake_minimum_required(VERSION 3.18)
project(Ising2DMpiGpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set to your GPU (e.g. 75 for Turing, 86 for Ampere) or use -DCMAKE_CUDA_ARCHITECTURES=native
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

# MPI + GPU: kernel launch lives in gpu_ising.cpp → compile as CUDA
set(GPU_ISING_SRC gpu_ising.cpp)
set_source_files_properties(${GPU_ISING_SRC} PROPERTIES LANGUAGE CUDA)

add_executable(ising_2d_mpi_gpu
  ising_2d_mpi_gpu.cpp
  ${GPU_ISING_SRC}
  kernel_ising_mpi.cu
)

target_link_libraries(ising_2d_mpi_gpu PRIVATE
  MPI::MPI_CXX
  CUDA::cudart
)

target_compile_options(ising_2d_mpi_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O2>
  $<$<COMPILE_LANGUAGE:CUDA>:-O2>
)

set_target_properties(ising_2d_mpi_gpu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)