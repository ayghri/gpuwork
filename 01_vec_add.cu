// Exercise 1: Vector Addition
// ===========================
// Implement a CUDA kernel that adds two vectors element-wise:
//   C[i] = A[i] + B[i]
//
// Concepts: __global__, blockIdx, blockDim, threadIdx, boundary check
//
// Compile: nvcc 01_vec_add.cu -o vec_add && ./vec_add

#include <stdio.h>
#include <stdlib.h>

__global__ void vec_add_kernel(const float* A, const float* B, float* C, int N) {
    // TODO: compute the global thread index
    int i = 0; // fix this

    // TODO: boundary check and perform the addition
}

int main() {
    int N = 1 << 20;  // 1M elements
    size_t bytes = N * sizeof(float);

    // Host arrays
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(2 * i);
    }

    // Device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    vec_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, N);

    // Copy back and verify
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) errors++;
    }
    printf("Errors: %d / %d\n", errors, N);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
