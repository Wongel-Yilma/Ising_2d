cmake_minimum_required(VERSION 3.18)  # minimum for CUDA language support

project(Ising2DGPU LANGUAGES CXX CUDA)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify the CUDA architecture (sm_75)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Optionally, use separate compilation for .cu files
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add executable
add_executable(ising_2d_gpu ising_2d_gpu.cpp kernel_ising.cu)

# Link CUDA libraries automatically
set_target_properties(ising_2d_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Optional: enable all warnings
if(MSVC)
    target_compile_options(ising_2d_gpu PRIVATE /W4)
else()
    target_compile_options(ising_2d_gpu PRIVATE -Wall -Wextra -O2)
endif()