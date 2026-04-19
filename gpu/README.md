# This is the GPU version of the 2D Ising model.

How to run it on CHPC:

1. Load the appropraite modules:
```bash
    module load gcc/8.5.0 cuda/11.6.2 cmake/3.26.0
```
2. Compile it with cmake --> Makes the compilation simple. cuda architecture definiion in the CMakeLists.txt file will vary
from GPU to GPU. Make sure the arch number defined in line 10 matches the hardware.
Here we used 86 for NVIDIA GPU 3090.
3. create a build directory and enter it:
```bash
    mkdir build 
    cd build 
```
4. Setup the compilation:
```bash
    cmake ..
```
5. Compile it
```bash
    cmake --build .
```
6. Copy the compiled executable to the main folder.
```bash
    cp ./ising_2d_gpu ..
    cd ..
```
7. Run it with an input file, and thread-count. box size N should be evenly divisible by SLURM_NTASKS
```bash
    ./ising_2d_gpu in.input 
```