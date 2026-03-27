// Exercise 1b: Fused Vector Addition + ReLU -- Solution
// ======================================================
// Compile: nvcc 01b_fused_vec_add_relu_solution.cu -o vec_add_relu && ./vec_add_relu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void vec_add_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

__global__ void relu_kernel(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

__global__ void fused_vec_add_relu_kernel(const float* A, const float* B,
                                          float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = fmaxf(A[i] + B[i], 0.0f);
    }
}

int main() {
    int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i - N / 2;
        h_b[i] = (float)(i % 100);
    }

    float *d_a, *d_b, *d_tmp, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_tmp, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    int warmup = 5, repeat = 20;

    // Unfused
    for (int i = 0; i < warmup; i++) {
        vec_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_tmp, N);
        relu_kernel<<<grid_size, block_size>>>(d_tmp, d_c, N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        vec_add_kernel<<<grid_size, block_size>>>(d_a, d_b, d_tmp, N);
        relu_kernel<<<grid_size, block_size>>>(d_tmp, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float unfused_ms = ms / repeat;

    // Fused
    for (int i = 0; i < warmup; i++) {
        fused_vec_add_relu_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        fused_vec_add_relu_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float fused_ms = ms / repeat;

    // Verify
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = fmaxf(h_a[i] + h_b[i], 0.0f);
        if (fabsf(h_c[i] - expected) > 1e-3) errors++;
    }

    printf("N = %d (%.1f MB)\n", N, (float)bytes / 1e6);
    printf("\n");
    printf("Unfused (2 kernels): %.3f ms\n", unfused_ms);
    printf("Fused   (1 kernel):  %.3f ms\n", fused_ms);
    printf("Speedup: %.2fx\n", unfused_ms / fused_ms);
    printf("Errors:  %d\n", errors);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_tmp); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
