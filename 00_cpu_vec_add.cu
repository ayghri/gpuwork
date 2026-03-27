// Exercise 0: CPU Vector Addition (Baseline)
// =============================================
// Before writing GPU code, let's see what vector addition
// looks like on CPU -- both sequential and with OpenMP.
//
// This gives us a baseline to compare against the GPU version.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void vec_add_sequential(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

void vec_add_openmp(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1 << 24;  // 16M elements
    size_t bytes = N * sizeof(float);

    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);

    // Initialize
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(2 * i);
    }

    // Sequential
    double start = omp_get_wtime();
    vec_add_sequential(A, B, C, N);
    double end = omp_get_wtime();
    printf("Sequential: %.3f ms\n", (end - start) * 1000.0);

    // OpenMP
    start = omp_get_wtime();
    vec_add_openmp(A, B, C, N);
    end = omp_get_wtime();
    printf("OpenMP (%d threads): %.3f ms\n", omp_get_max_threads(), (end - start) * 1000.0);

    // Verify
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (C[i] != A[i] + B[i]) errors++;
    }
    printf("Errors: %d\n", errors);

    free(A);
    free(B);
    free(C);
    return 0;
}

// Compile: gcc -O2 -fopenmp 00_cpu_vec_add.c -o cpu_vec_add
// Run:     ./cpu_vec_add
//
// Questions to think about:
//   - How does sequential compare to OpenMP?
//   - How many threads does OpenMP use? (hint: OMP_NUM_THREADS)
//   - After Exercise 1, compare these numbers to the GPU version.
//     For vec_add, is the GPU actually faster? Why or why not?
