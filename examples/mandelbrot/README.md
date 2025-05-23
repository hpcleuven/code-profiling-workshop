This directory contains several code examples that compute the
[Mandelbrot set](https://en.wikipedia.org/wiki/Mandelbrot_set). All programs
need a command-line argument `N` to specify the dimensions of the grid on
which the Mandelbrot set is computed. The actual dimensions are `3N` (from
-2 to 1) in the x-direction and `2N` (from -1 to +1) in the y-direction. For
each example, a Slurm jobscript is provided that shows how the code can be
compiled and profiled with LinaroForge MAP. The jobscripts have been tested
on the KU Leuven/UHasselt Tier-2 infrastructure and will require modifications
on other infrastructure. When submitting on the KU Leuven/UHasselt cluster,
you need to include a valid credit account
(`sbatch --account=<your_account> job.slurm`).

## serial

The file `serial/mandelbrot.c` contains a serial C implementation. The code
contains a variant (which can be selected by compiling with `-DUSE_AVX2`) that
uses vector intrinsics to treat multiple x-coordinates in a SIMD fashion
(specifically using AVX2). Compilers will (normally) not be able automatically
vectorize the code, since the number of iterations of the inner loop is
different for each coordinate. Interestingly, the total number of iterations
will change slightly when using AVX2 instead of the version without vectorization.

## mpi

The file `mpi/mandelbrot.c` contains a version that runs in parallel by
decomposing the domain into rectangles using MPI. To keep the code simple, it
is required that the number of grid points along x and y are divisible by the
number of domains in the x- and y-direction respectively, the latter being
determined automatically using `MPI_Cart_create`. You need to choose the 
command-line argument `N` appropriately or the program will exit.
The example job script will do runs in both `$VSC_DATA` and `$VSC_SCRATCH` to
demonstrate the influence of the filesystem on MPI-IO performance. As a
consequence, the LinaroForge MAP output files will not be located in the job
submission directory in this case.

## cuda

The file `cuda/mandelbrot.cu` contains a CUDA version of the serial code. This
program requires two command-line arguments: `N` to indicate the grid size as
before and the [CUDA blocksize](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/).

## postprocessing

All programs above output a binary file `output.bin` containing the number of
iterations for each grid point. The `plot_mandelbrot.py` script generates an
image from such a file using matplotlib.
