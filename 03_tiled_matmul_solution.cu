// Exercise 3: Tiled Matrix Multiplication -- Solution
// ====================================================
// Compile: nvcc 03_tiled_matmul_solution.cu -o tiled_matmul && ./tiled_matmul

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE 16

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;

        int b_row = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N) {
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

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

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
