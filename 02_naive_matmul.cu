// Exercise 2: Naive Matrix Multiplication
// ========================================
// Implement C = A * B where A is (M x K) and B is (K x N).
// Each thread computes one element of C.
//
// Concepts: 2D grid/blocks, row-major indexing, dot product loop
//
// Compile: nvcc 02_naive_matmul.cu -o naive_matmul && ./naive_matmul

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    // TODO: compute row and col from 2D block/thread indices
    int row = 0; // fix this
    int col = 0; // fix this

    if (row < M && col < N) {
        float sum = 0.0f;
        // TODO: loop over K and accumulate the dot product
        // A is row-major: A[row][k] = A[row * K + k]
        // B is row-major: B[k][col] = B[k * N + col]

        C[row * N + col] = sum;
    }
}

int main() {
    int M = 512, N = 512, K = 512;

    float *h_a = (float*)malloc(M * K * sizeof(float));
    float *h_b = (float*)malloc(K * N * sizeof(float));
    float *h_c = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++) h_a[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) h_b[i] = (float)(rand() % 100) / 100.0f;

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M * K * sizeof(float));
    cudaMalloc(&d_b, K * N * sizeof(float));
    cudaMalloc(&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify a few elements against CPU
    int errors = 0;
    for (int i = 0; i < M && errors < 10; i++) {
        for (int j = 0; j < N && errors < 10; j++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++) expected += h_a[i * K + k] * h_b[k * N + j];
            if (fabsf(h_c[i * N + j] - expected) > 1e-2) {
                printf("Mismatch at (%d,%d): got %f, expected %f\n",
                       i, j, h_c[i * N + j], expected);
                errors++;
            }
        }
    }
    printf(errors == 0 ? "PASSED\n" : "FAILED with %d errors\n", errors);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
