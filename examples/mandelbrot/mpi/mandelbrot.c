#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <stdint.h>
#include <mpi.h>
#include <immintrin.h>

#define MAXITER 255

int get_N(int argc, char *argv[], int Px, int Py) {
    /* Read an integer as the (only) command-line argument and check that an
    evenly distributed domain decomposition is possible */
    if (argc != 2) {
        printf("Expected exactly 1 argument: N\n");
        exit(EXIT_FAILURE);
    }

    int N = atoi(argv[1]);

    if (N % Px != 0) {
        printf("Px should evenly divide N\n");
        exit(EXIT_FAILURE);
    }
    if (N % Py != 0) {
        printf("Py should evenly divide N\n");
        exit(EXIT_FAILURE);
    }
    return N;
}

MPI_Comm get_domain_decomposition(int size, int* dims, int *my_rank, int* my_coords) {
    /* Let the MPI library create a 2D cartesian domain decomposition */
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {false, false};
    int reorder = true;
    MPI_Comm new_communicator;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &new_communicator);
    MPI_Comm_rank(new_communicator, my_rank);
    MPI_Cart_coords(new_communicator, *my_rank, 2, my_coords);
    return new_communicator;
}

long long int mandelbrot(int N, int myNx, int myNy, int myOx, int myOy, int (*arr)[myNy]) {
    long long int niter = 0;
    double x, y, wx,wy,v,xx;
    int k;

    for (int i=0; i < myNx; i++) {
        x = (1.0 * (i + myOx) - 2.0 * N) / N;
        for (int j=0; j < myNy; j++) {
            y = (1.0 * (j + myOy) - 1.0 * N) / N;
            wx = 0.0; wy = 0.0; v = 0.0;
            k = 0;
            while ((v < 4) && (k++ < MAXITER))
            {
                xx = wx*wx - wy*wy;
                wy = 2.0*wx*wy;
                wx = xx + x;
                wy = wy + y;
                v = wx*wx + wy*wy;
            }
            arr[i][j] = k - 1;
            niter += k - 1;
        }
    }
    return niter;
}

long long int mandelbrot_avx2(int N, int myNx, int myNy, int myOx, int myOy, int (*arr)[myNy]) {
    long long int niter = 0;

    __m256d wx, wy, v, xx, x, y;
    __m256i k;
    __m256d four = _mm256_set1_pd(4.0);
    __m256d two = _mm256_set1_pd(2.0);
    __m256i one = _mm256_set1_epi64x(1);

    for (int i=0; i < myNx; i+=4) {
       __m256d x = _mm256_set_pd(
            (1.0 * (i+myOx+3) - 2.0 * N) / N,
            (1.0 * (i+myOx+2) - 2.0 * N) / N,
            (1.0 * (i+myOx+1) - 2.0 * N) / N,
            (1.0 * (i+myOx+0) - 2.0 * N) / N 
        );
        for (int j=0; j < myNy; j++) {
            double yscalar = (1.0 * (j + myOy) - 1.0 * N) / N;
            y = _mm256_set1_pd(yscalar);
            wx = _mm256_setzero_pd();
            wy = _mm256_setzero_pd();
            v = _mm256_setzero_pd();
            k = _mm256_setzero_si256();

            // Iterate until divergence or max iterations
            for (int iter = 0; iter < MAXITER; iter++) {
                // Compute new values
                xx = _mm256_sub_pd(_mm256_mul_pd(wx, wx), _mm256_mul_pd(wy, wy));
                wy = _mm256_mul_pd(_mm256_mul_pd(two, wx), wy);
                wx = _mm256_add_pd(xx, x);
                wy = _mm256_add_pd(wy, y);
                v = _mm256_add_pd(_mm256_mul_pd(wx, wx), _mm256_mul_pd(wy, wy));

                // Check if all lanes are converged
                __m256d mask = _mm256_cmp_pd(v, four, _CMP_LT_OQ);
                int bitmask = _mm256_movemask_pd(mask);
                if (bitmask == 0) break;

                // Update number of iterations
                k = _mm256_add_epi64(k, _mm256_castpd_si256(mask));
            }

            // Store number of iterations; the minus sign in front of k has
            // something to do with alignment
            long long int temp[4];
            _mm256_store_si256((__m256i*)temp, -k);
            niter += temp[0] + temp[1] + temp[2] + temp[3];
            arr[i][j] = temp[0];
            arr[i+1][j] = temp[1];
            arr[i+2][j] = temp[2];
            arr[i+3][j] = temp[3];
        }
    }
    return niter;
}

void write_output_mpiio(int N, int myNx, int myNy, int myOx, int myOy, MPI_Comm cart_comm, int (*arr)[myNy]) {
    /* Create derived type for file view */
    MPI_Datatype view;
    int startV[2] = { myOx, myOy };
    int arrsizeV[2] = { 3 * N, 2 * N };
    int gridsizeV[2] = { myNx, myNy };
    MPI_Type_create_subarray(2, arrsizeV, gridsizeV,
                             startV, MPI_ORDER_C, MPI_INT, &view);
    MPI_Type_commit(&view);

    /* Create derived datatype for interior grid (output grid) */
    MPI_Datatype grid;
    int start[2] = {0, 0};
    int arrsize[2] = {myNx, myNy};
    int gridsize[2] = {myNx, myNy};
    MPI_Type_create_subarray(2, arrsize, gridsize,
                            start, MPI_ORDER_C, MPI_FLOAT, &grid);
    MPI_Type_commit(&grid);

    /* MPI IO */
    MPI_File fh;
    MPI_File_open(cart_comm, "output.bin", MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &fh);
    MPI_File_set_view(fh, 0, MPI_INT, view, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, &arr[0][0], 1, grid, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    MPI_Type_free(&view);
}

int main(int argc, char *argv[]) {
    // Start a timer
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // Initialize domain decomposition
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int coords[2];
    int dims[2] = {0, 0};
    MPI_Comm cart_comm;
    cart_comm = get_domain_decomposition(size, dims, &rank, coords);
    int Px = dims[0];
    int Py = dims[1];
    if (rank == 0) {
        printf("Domain decomposition: Px = %d Py = %d\n", dims[0], dims[1]);
    }

    // Check input parameters
    int N = get_N(argc, argv, Px, Py);

    // Compute local size and offset
    int myNx = 3 * N / Px;
    int myOx = coords[0] * myNx;
    int myNy = 2 * N / Py;
    int myOy = coords[1] * myNy;
 
    // Initialize result array
    int (*arr)[myNy] = malloc(sizeof(int[myNx][myNy]));

    // Do the actual work
#ifdef USE_AVX2
    long long int niter = mandelbrot_avx2(N, myNx, myNy, myOx, myOy, arr);
#else
    long long int niter = mandelbrot(N, myNx, myNy, myOx, myOy, arr);
#endif
    // Synchronize after workload; not strictly necessary, but allows
    // separating actual workload from I/O in profiler
    long long int total_niter;
    MPI_Allreduce(&niter, &total_niter, 1, MPI_LONG_LONG_INT, MPI_SUM, cart_comm);

    // Write output to file
    write_output_mpiio(N, myNx, myNy, myOx, myOy, cart_comm, arr);

    // Stop measuring time and calculate the elapsed time
    MPI_Barrier(cart_comm);    
    gettimeofday(&end, 0);
    long seconds = end.tv_sec - begin.tv_sec;
    long microseconds = end.tv_usec - begin.tv_usec;
    double elapsed = seconds + microseconds*1e-6;    
    printf("Rank %5d performed %16lld iterations in %8.3fs\n", rank, niter, elapsed);
    if (rank == 0) printf("Total number of iterations is %16lld\n", total_niter);

    // Finalize
    free(arr);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
