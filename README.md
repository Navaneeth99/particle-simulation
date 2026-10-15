# Parallel Particle Simulation (OpenMP)

A 2D particle simulation where particles repel each other within a cutoff distance, sped up from O(n²) to O(n) and parallelized with OpenMP

![Particle simulation](particles.gif)

## How it works

- **Spatial grid:** the box is split into cells one cutoff wide, so each particle only checks the 3×3 block of cells around it instead of every other particle. That makes each timestep O(n).
- **OpenMP:** binning, force computation, movement, and grid clearing each run as a parallel loop. A lock per cell keeps concurrent inserts into the grid safe.

## Build and run

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make

./serial -n 10000 -s 1                       # serial O(n) version
OMP_NUM_THREADS=8 ./openmp -n 100000 -s 1    # OpenMP version
```

Options: `-n` number of particles, `-s` random seed, `-o` output file of particle positions.

## Files

| File | Description |
|---|---|
| `serial.cpp` | Linear-time serial implementation |
| `openmp.cpp` | OpenMP parallel implementation |
| `common.h` | Particle struct and simulation constants |
| `main.cpp` | Driver (provided starter code) |

## Acknowledgments

Starter code and assignment adapted from UC Berkeley CS267 and MIT 6.172.
