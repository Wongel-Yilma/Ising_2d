# This is the OpenMP version of the Ising model.

How to compile and run it on CHPC:

1. Load the appropraite modules:
```bash
    $ module load gcc/8.5.0
```
2. Compile it with g++
```bash
    $ g++ -o ising_2d_omp ising_2d_omp.cpp -fopenmp
```
3. Run it with an input file, and thread-count. box size N should be evenly divisible by THREAD_COUNT 
```bash
    $ ./ising_2d_omp in.input $THREAD_COUNT
```
