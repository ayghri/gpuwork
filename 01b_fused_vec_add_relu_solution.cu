// Exercise 1b: Fused Vector Addition + ReLU -- Solution
// ======================================================
// Compile: nvcc 01b_fused_vec_add_relu_solution.cu -o vec_add_relu && ./vec_add_relu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void fused_vec_add_relu_kernel(const float* A, const float* B,
                                          float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = fmaxf(A[i] + B[i], 0.0f);
    }
}

int main() {
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i - N / 2;
        h_b[i] = (float)(i % 100);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    fused_vec_add_relu_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = fmaxf(h_a[i] + h_b[i], 0.0f);
        if (fabsf(h_c[i] - expected) > 1e-5) errors++;
    }
    printf("Errors: %d / %d\n", errors, N);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
