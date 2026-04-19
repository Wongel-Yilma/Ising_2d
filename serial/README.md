# This is the serial version of the Ising model.

How to run it on CHPC:

1. Load the appropraite modules:
```bash
    $ module load gcc/8.5.0
```
2. Compile it with g++
```bash
    $ g++ -o ising_2d ising_2d.cpp
```
3. Run it with an input file
```bash
    $ ./ising_2d in.input
```