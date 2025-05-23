#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <stdint.h>
#include <immintrin.h>

#define MAXITER 255

int get_N(int argc, char *argv[]) {
    /* Read an integer as the (only) command-line argument */
    if (argc != 2) {
        printf("Expected exactly 1 argument: N\n");
        exit(EXIT_FAILURE);
    }

    int N = atoi(argv[1]);

    return N;
}

long long int mandelbrot(int N, int myNx, int myNy, int (*arr)[myNy]) {
    long long int niter = 0;
    double x, y, wx,wy,v,xx;
    int k;

    for (int i=0; i < myNx; i++) {
        x = (1.0 * i - 2.0 * N) / N;
        for (int j=0; j < myNy; j++) {
            y = (1.0 * j - 1.0 * N) / N;
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

long long int mandelbrot_avx2(int N, int myNx, int myNy, int (*arr)[myNy]) {
    long long int niter = 0;

    __m256d wx, wy, v, xx, x, y;
    __m256i k;
    __m256d four = _mm256_set1_pd(4.0);
    __m256d two = _mm256_set1_pd(2.0);
    __m256i one = _mm256_set1_epi64x(1);

    for (int i=0; i < myNx; i+=4) {
       __m256d x = _mm256_set_pd(
            (1.0 * (i+3) - 2.0 * N) / N,
            (1.0 * (i+2) - 2.0 * N) / N,
            (1.0 * (i+1) - 2.0 * N) / N,
            (1.0 * (i+0) - 2.0 * N) / N 
        );
        for (int j=0; j < myNy; j++) {
            double yscalar = (1.0 * j - 1.0 * N) / N;
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

int main(int argc, char *argv[]) {
    // Start a timer
    struct timeval begin, end;
    gettimeofday(&begin, 0);

    // Check input parameters
    int N = get_N(argc, argv);

    // Compute local size and offset
    int myNx = 3 * N;
    int myNy = 2 * N;

    // Initialize result array
    int (*arr)[myNy] = malloc(sizeof(int[myNx][myNy]));

    // Do the actual work
#ifdef USE_AVX2
    long long int niter = mandelbrot_avx2(N, myNx, myNy, arr);
#else
    long long int niter = mandelbrot(N, myNx, myNy, arr);
#endif

    // Write output to file
    FILE *fh = fopen("output.bin", "wb");
    fwrite(&arr[0][0], sizeof(int), myNx * myNy, fh);
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
