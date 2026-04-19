# 🧲 2D Ising Model: Serial, OpenMP, MPI, and GPU Implementations

## 📌 Introduction

This project models the equilibrium behavior of the two-dimensional Ising system using **Monte Carlo simulation techniques**. The primary focus is on **thermalization**, where the lattice evolves toward equilibrium at a given temperature.

We consider a square lattice with:
- Periodic boundary conditions  
- Variable lattice sizes  
- The standard Ising Hamiltonian  

---

## 🚀 Implementations

This repository contains **six C++ implementations** of the 2D Ising model:

- ✅ Serial  
- ✅ Shared Memory (OpenMP)  
- ✅ GPU (CUDA)  
- ✅ Distributed Memory (MPI)  
- ✅ Distributed Memory + GPU (MPI + CUDA)  
- ✅ GPU-aware MPI  

---

# 1. Serial Implementation

The serial version serves as the **baseline** for all parallel implementations.

## 🔬 Method

We use the **Metropolis Monte Carlo algorithm**:
- Compute energy difference ΔE for a spin flip  
- If ΔE < 0 → accept flip  
- Else → accept with probability:  
  e^(−ΔE / T)

---

## ⚙️ Execution Flow

1. Read input parameters (N, T, steps, output frequency, seed)
2. Initialize lattice with random spins (±1)
3. Run Monte Carlo loop:
   - Compute neighbor interactions
   - Apply Metropolis criterion
4. Periodically:
   - Compute energy and magnetization
   - Write outputs (dump + log)

---

## ♟️ Checkerboard Update

To avoid dependency conflicts:

- Split lattice into **black and white sites**
- Update in two phases:
  1. Even (white) sites  
  2. Odd (black) sites  

**Benefits:**
- Deterministic update order  
- Enables safe parallelization  

---

# 2. Shared Memory (OpenMP)

Parallelization introduces **race conditions** due to neighbor dependencies.

## ✅ Solution

Use the **checkerboard scheme**:
- Ensures no two adjacent spins are updated simultaneously

## ⚙️ Key Features

- `#pragma omp parallel` for main simulation  
- Work-sharing with `omp for`  
- Reductions for:
  - Energy  
  - Magnetization  
- Single-threaded I/O to avoid conflicts  

---

# 3. GPU Implementation (CUDA)

This version accelerates computation using the GPU.

## ⚡ Key Ideas

- Each thread handles one lattice site  
- Uses **shared memory** for neighbor access  
- Two kernel launches per step (checkerboard phases)  

## ⚙️ Execution Flow

1. Copy lattice to GPU  
2. Generate random numbers on host  
3. Launch CUDA kernels:
   - Load tiles into shared memory  
   - Compute ΔE and apply Metropolis rule  
4. Copy results back to CPU  
5. Compute observables on host  

---

# 4. Distributed Memory (MPI)

The lattice is decomposed **row-wise across MPI ranks**.

## 🔁 Communication Pattern

- Each rank exchanges boundary rows with neighbors  
- Ring topology:
  - `up` neighbor  
  - `down` neighbor  

## ⚙️ Key Steps

- `MPI_Bcast` → distribute parameters  
- `MPI_Scatter` → distribute lattice  
- `MPI