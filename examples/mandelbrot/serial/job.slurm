#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --tasks-per-node=36
#SBATCH --time=00:30:00
#SBATCH --cluster=genius
#SBATCH --partition=batch_debug

module load LinaroForge
module load foss/2023a

export N=17280
gcc -march=native -O3 -g1 -fno-inline -fno-optimize-sibling-calls mandelbrot.c -o mandelbrot
gcc -DUSE_AVX2 -march=native -O3 -g1 -fno-inline -fno-optimize-sibling-calls mandelbrot.c -o mandelbrot_avx2

for suffix in "" "_avx2"
do
    ./mandelbrot${suffix} $N
    map --profile ./mandelbrot${suffix} $N
done
