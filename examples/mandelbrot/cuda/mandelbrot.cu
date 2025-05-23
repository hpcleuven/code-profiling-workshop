#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <stdint.h>
#include <cuda.h>

#define MAXITER 255

int get_N(int argc, char *argv[], int* blocksize) {
    /* Read an integer as the (only) command-line argument */
    if (argc != 3) {
        printf("Expected exactly 2 arguments: N, blocksize\n");
        exit(EXIT_FAILURE);
    }

    int N = atoi(argv[1]);

    *blocksize = atoi(argv[2]);

    if (N % *blocksize != 0) {
        printf("blocksize should evenly divide N\n");
        exit(EXIT_FAILURE);
    }

    return N;
}

__global__ void mandelbrot_kernel(int N, int Nx, int Ny, int *arr) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    double x = (1.0 * i - 2.0 * N) / N;
    double y = (1.0 * j - 1.0 * N) / N;

    double wx = 0;
    double wy = 0;
    double v = 0;
    double xx = 0;
    int k;

    for (k = 0; k < MAXITER; k++){
        xx = wx*wx - wy*wy;
        wy = 2.0*wx*wy;
        wx = xx + x;
        wy = wy + y;
        v = wx*wx + wy*wy;
        if (v >= 4.0) break;
    }
    arr[i*Ny + j] = k;
}

int main(int argc, char *argv[]) {
    // Start a timer
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // Check input parameters
    int blocksize;
    int N = get_N(argc, argv, &blocksize);

    // Compute local size and offset
    int myNx = 3 * N;
    int myNy = 2 * N;

    // Initialize result array
    int *arr_dev = NULL;
    int err = cudaMalloc(&arr_dev, sizeof(int) * myNx * myNy);
    if (err != 0) {
        printf("cudaMalloc failed!\n");
        return err;
    }

    // Do the actual work
    dim3 dimBlock(blocksize, blocksize);
    dim3 dimGrid(myNx/blocksize, myNy/blocksize);
    mandelbrot_kernel<<<dimGrid, dimBlock>>>( N, myNx, myNy, arr_dev);

    // Copy data from device to host for postprocessing
    int *arr = (int *)malloc(sizeof(int) * myNx * myNy);
    cudaMemcpy(arr, arr_dev, sizeof(int) * myNx * myNy, cudaMemcpyDeviceToHost);

    long long int niter = 0;
    for (int k=0; k < myNx * myNy; k++) niter += arr[k];

    // Write output to file
    FILE *fh = fopen("output.bin", "wb");
    fwrite(&arr[0], sizeof(int), myNx * myNy, fh);
    fclose(fh);

    // Stop measuring time and calculate the elapsed time
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;    
    printf("Performed %16lld iterations in %8.3fs\n", niter, elapsed);

    // Finalize
    free(arr);
    return EXIT_SUCCESS;
}
