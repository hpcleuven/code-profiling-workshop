#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --tasks-per-node=9
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --cluster=genius
#SBATCH --partition=gpu_p100_debug

module load LinaroForge
module load CUDA/12.4.0

export N=17280
export blocksize=32
nvcc -O3 -g mandelbrot.cu -o mandelbrot
executable=${SLURM_SUBMIT_DIR}/mandelbrot
cd $VSC_SCRATCH
${executable} $N ${blocksize}
map --profile ${executable} $N ${blocksize}
