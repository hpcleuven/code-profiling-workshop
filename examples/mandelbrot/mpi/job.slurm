#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --tasks-per-node=36
#SBATCH --time=00:30:00
#SBATCH --cluster=genius
#SBATCH --partition=batch_debug

module load LinaroForge
module load foss/2023a

# Compilation
mpicc -march=native -O3 -g1 -fno-inline -fno-optimize-sibling-calls mandelbrot.c -o mandelbrot
mpicc -DUSE_AVX2 -march=native -O3 -g1 -fno-inline -fno-optimize-sibling-calls mandelbrot.c -o mandelbrot_avx2

do_runs() {
    for suffix in "" "_avx2"
    do
        executable=${SLURM_SUBMIT_DIR}/mandelbrot${suffix}
        rm output.bin
        mpirun ${executable} $N
        rm output.bin
        map --profile mpirun ${executable} $N
    done
}

export N=17280
# Base run on NFS
cd $VSC_DATA
do_runs
# Write output to Lustre
cd $VSC_SCRATCH
do_runs
