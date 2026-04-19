# Description and Instructions

To run each implementation on CHPC, you must either create an ssh connection through VS Code or open a secure shell in the browser. If you choose the latter, you must make sure that the file that you want to run has been uploaded to the home directory of the files section on the CHPC website.

Also, only use the CHPC login shell for loading libraries and compiling code. After that, you can request to enter an interactive shell for an available cluster like this:

```
salloc -n <number of tasks> -N <number of nodes> -t <time> -p <partition> -A <Account>
```

# ising_2d (serial)

The serial implementation of this project acts as the groundwork for any kind of parallelization in subsequent implementations. The method that we chose is commonly known as the mote carlo method. Therefore, the foundational logic outlined in this implementation also applies to the parallelized versions. This implementation is simply a barebones approach to work off of. In this implementation, the program simply loops over every other lattice site. If the difference in energy before and after a flip is less than 0, then we flip. if it is greater than 0, then we flip only if a random number r is less than e^-EDiff/T. We then need to do this to the other portion of lattice sites that were left out.

**execution flow**

The program begins execution in main, where it first checks that an input file has been provided as a command-line argument. It then creates an instance of the Ising class, passing the input file name to the constructor. Inside the constructor, the program reads the input file and extracts simulation parameters such as the lattice size N, temperature, number of steps, output frequency, and random seed. After parsing these values, it calls a setup function that allocates memory for the spin configuration and random number arrays, initializes the random number generator with the given seed, and opens the output files for logging and data dumping.

Once the object is constructed, the Run function is called to start the simulation. The first step in this process is initialization, where each site in the N×N lattice is assigned a random spin value of either −1 or +1, representing a disordered, high-temperature state. The initial magnetization is computed and printed. The program then enters the main simulation loop, which continues until the specified number of steps is reached. At each step, a Monte Carlo update is performed in which the program iterates over the lattice and attempts to flip spins based on the Metropolis acceptance criterion. For each spin, it calculates the sum of its four nearest neighbors using periodic boundary conditions, computes the energy change that would result from flipping the spin, and then decides whether to accept the flip. If the energy decreases, the flip is always accepted; otherwise, it is accepted with a probability that depends on the temperature and the energy increase.

After each Monte Carlo step, the program increments the step counter and, at specified intervals, computes the total energy and magnetization of the system. It then writes the current lattice configuration to a dump file for visualization and logs the energy and magnetization values to a separate log file. Progress is also printed to the console. This loop continues until all simulation steps are completed. Finally, control returns to main, where the total execution time is calculated and printed. The simulation object is then deleted, which triggers the destructor to close the output files and free allocated memory, and the program terminates.

**Running on CHPC**
in CHPC if gcc is not already loaded, you can do so with:

```
module load gcc
```

At which point you can run

```
g++ ising2d.cpp -o executablename
```

# ising_2d_omp (shared memory implimentation)

When parallelizing this project, we need to worry about race conditions. each flip determination relies on its neighbors, if their neighbors are being changed by other threads working concurrently, then this could lead to race conditions. So, for this openmp implementation, we use a checkerboard pattern. We start by only updating every other index, this forms a sort of checkerboard pattern where the neighbors of each index that are getting updated will not be touched by other threads. We can then update the remaining squares. This solution avoids race conditions.

**execution flow**

The overall execution flow of this OpenMP implementation follows the same high-level structure as the serial version (input parsing, setup, initialization, Monte Carlo updates, and periodic output), but the key difference is that the main simulation is executed inside a parallel region. After main reads the input file and the desired number of threads, it constructs the Ising object exactly as in the serial implementation. When Run() is called, the program enters a #pragma omp parallel region where multiple threads cooperate on the simulation. Within this region, initialization is performed once collectively: a single directive ensures that only one thread generates the random numbers and prints output, while a parallel for loop distributes the work of assigning spin values across threads. The simulation loop then proceeds similarly to the serial case, but each Monte Carlo step (MC_Move) is parallelized so that different threads update different portions of the lattice simultaneously using OpenMP for directives. The checkerboard update pattern is preserved to avoid race conditions between neighboring spins, ensuring correctness despite parallel updates. The global step counter is incremented inside a single block to prevent conflicts, while energy and magnetization calculations use OpenMP reduction clauses so that each thread computes partial sums that are safely combined into global values. Output operations such as dumping configurations, logging, and printing progress are also restricted to a single thread to avoid file corruption or duplicate output. This loop continues until the total number of steps is reached, after which control exits the parallel region, timing information is printed, and cleanup occurs as in the serial implementation.

**Running in CHPC**

When Running this program on CHPC, you just need to make sure that during compilation, you include openMP like this:

```
g++ ising_2d_omp.cpp -o <ExecutableName> -fopenmp
```

# ising_2d_gpu (gpu implimentation)

The Cuda implementation of this project is very similar to our openMP solution. Just like before, we are using the checkerboard pattern to avoid race conditions. In this solution, we use cuda to take advantage of the GPU to speed up the process. The Cuda kernel is located in kernel_ising.cu

**execution flow**
The full execution flow of the CUDA implementation combines the same high-level structure as the serial version of the Ising model with a GPU-accelerated update step. As before, the program begins on the host (CPU): main constructs the Ising object, the input file is parsed, memory is allocated, and the lattice is initialized with random spins. The simulation loop in Run() proceeds similarly to the serial version, but each call to MC_Move() offloads the spin updates to the GPU via launch_kernel_ising. Inside this function, host memory for the lattice is copied to device memory (input_config_d), and additional device memory is allocated for the output configuration and random numbers. The kernel execution is then organized over a 2D grid of thread blocks, where each block processes a tile of the lattice.

For each group of Monte Carlo steps (equal to output_freq), the host first generates random numbers and copies them to the GPU. The simulation then performs two kernel launches per step to implement the checkerboard update scheme. Inside the mcMoveKernel, each thread corresponds to a lattice site and first loads its spin (and neighboring halo values) into shared memory, which significantly reduces global memory access. After synchronization, threads compute the neighbor sum, evaluate the energy difference, and apply the Metropolis acceptance criterion to decide whether to flip the spin. Only threads corresponding to valid lattice sites and matching the current checkerboard phase perform updates, ensuring no race conditions occur. The updated spins are written to the output array in global memory. After each phase, the device arrays are copied or swapped so that the updated configuration becomes the input for the next phase.

Once all kernel iterations are complete, the final lattice configuration is copied back from device memory to the host (output_config_h). Control returns to the CPU, where energy and magnetization are computed exactly as in the serial implementation, and results are written to output files. This process repeats until all simulation steps are completed. Finally, device memory is freed, host memory is cleaned up, and the program terminates.

**Running in CHPC**

When you run the cuda implimentation in CHPC, you have to make sure that you choose a cluster that gives you access to a gpu. Before you compile this program, make sure you load in cuda and cmake along with gcc:

```
module load cuda gcc cmake
```

And to compile cuda, use nvcc:

```
nvcc ising_2d_gpu.cpp -o <ExecutableName>
```

When you request time to allocate computing power, you need to make sure that it knows you will be using a GPU with --gres=gpu:

example:

```
salloc -n 1 -N 1 -t 0:15:00 -p notchpeak-gpu -A notchpeak-gpu --gres=gpu

```

# ising_2d_mpi (MPI implimentation)

For the MPI implimentation, we decided to split up the lattice horizontally so that each process gets a set number of rows. We us the message passing interface when a process needs to know the state of a lattice site that is either above or below its domain, messages get passed.

**Execution flow**

The execution flow of this MPI implementation follows the same overall structure as the serial version of the Ising model, but the lattice is distributed across multiple processes and coordinated through message passing. The program begins with MPI_Init, after which each process determines its rank and the total number of processes. Rank 0 reads the input file and initializes global simulation parameters, then broadcasts these values (such as N, temperature, number of steps, and seed) to all other processes using MPI_Bcast. The global lattice is conceptually divided row-wise among processes, so each process is responsible for a contiguous block of rows of size local_N=N/size. During initialization, rank 0 generates the full set of random spins and distributes portions of this array to each process using MPI_Scatter, allowing each process to independently construct its local spin configuration.

Once initialized, the simulation loop proceeds similarly to the serial case but operates on local subdomains. At each step, rank 0 generates random numbers for the entire lattice and scatters them so that each process receives the portion corresponding to its local rows. Each process then performs the Monte Carlo update (MC_Move) on its local lattice using the same checkerboard scheme as in the serial implementation. However, because each process only holds part of the lattice, it must exchange boundary rows with its neighboring processes to correctly compute interactions across subdomain boundaries. This is done using MPI_Sendrecv to update “ghost” rows (top and bottom neighbors) before and during the checkerboard phases. After completing the updates for a step, each process computes its local contribution to the energy and magnetization. These local values are then combined using MPI_Reduce to obtain global totals on rank 0.

At specified output intervals, all processes send their local lattice segments back to rank 0 using MPI_Gather, reconstructing the full lattice. Rank 0 alone handles output operations, writing the configuration to a dump file, logging energy and magnetization, and printing progress. This loop continues until all simulation steps are completed. Finally, memory is cleaned up on each process, MPI_Finalize is called to terminate the MPI environment, and the program exits.

**Running in CHPC**

Make sure you load MPI:

```
module load intel-mpi
```

You also need to make your final build using mpicc:

```
mpicc program.c -o executablename
```

# ising_2d_mpi_gpu.cpp

This implimentation takes advantage of the GPU just like our previous GPU implimentation, but, it also involves the message passing interface. The host initializes MPI and each MPI process uses the GPU. This implimentation involves ising_2d_mpi_gpu.cpp, kernel_ising_mpi.cu, and kernel_ising_mpi.cuh

**Execution flow**
The program begins in the host main function, where MPI is initialized and each process (rank) is created. Each rank determines its own ID and the total number of processes. The root process (rank 0) reads the input file, sets global simulation parameters, and broadcasts them to all other ranks so every process has identical configuration data.

Next, the lattice is partitioned across MPI processes by rows, so each rank owns a contiguous block of size local_N × N. Rank 0 generates the full initial random configuration and distributes it using MPI_Scatter, giving each process its local portion of spins. Each rank then converts these values into ±1 spin states and allocates memory for ghost rows (upper_ghost and lower_ghost), which will store boundary data from neighboring ranks.

Before simulation begins, each process performs an initial ghost-cell exchange using MPI_Sendrecv, ensuring boundary consistency between neighboring subdomains.

The simulation then enters the main loop. In each iteration, every MPI process independently:

Generates random numbers (rank 0 and then scattered)
Copies data to the GPU
Launches the CUDA kernel twice (checkerboard phase 0 and phase 1)

Inside the CUDA kernel, each GPU thread loads a tile of the local lattice into shared memory, including halo cells from ghost rows. The kernel then performs a checkerboard Monte Carlo update, where each spin is evaluated independently using the Metropolis criterion and updated into an output buffer.

After each kernel phase, the host copies updated boundary rows back from the GPU and performs MPI ghost exchanges, ensuring neighboring processes have updated edge information before the next phase.

After completing the Monte Carlo updates for the iteration, each MPI rank computes:

local energy
local magnetization

These are then combined using MPI_Reduce to produce global energy and magnetization on rank 0.

At output intervals, all local lattices are collected using MPI_Gather, and rank 0 writes the full system state to file and prints progress.

Finally, after all time steps are completed, memory is freed, MPI is finalized, and the program terminates.

**Running in CHPC**

In CHPC you start by loading modules:

```
module load gcc
module load cuda
module load openmpi
```

You can then create 1 build using both nvcc and mpicxx:

```
nvcc -ccbin mpicxx -O3 \
    ising_2d_mpi_gpu.cpp \
    kernel_ising_mpi.cu \
    -o ising_gpu
```
