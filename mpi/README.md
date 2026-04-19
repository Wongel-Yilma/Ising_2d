# This is the MPI version of the Ising model.

## How to compile and run it on CHPC

1. Load the appropriate modules:
```bash
   module load gcc/8.5.0 openmpi/4.1.4
   ```
2. Compile with mpicxx:
```bash
   mpicxx -o ising_2d_mpi ising_2d_mpi.cpp
```
3. Run with an input file. The box size N should be evenly divisible by SLURM_NTASKS:
```bash
   mpirun -np $SLURM_NTASKS ./ising_2d_mpi in.input
```