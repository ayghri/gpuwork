// Exercise 1b: Fused Vector Addition + ReLU
// ============================================
// Compute C[i] = max(A[i] + B[i], 0) in a SINGLE kernel.
//
// Why fuse? Without fusion you need two kernels:
//   1) vec_add:  tmp[i] = A[i] + B[i]   (read A,B from HBM, write tmp to HBM)
//   2) relu:     C[i] = max(tmp[i], 0)   (read tmp from HBM, write C to HBM)
// That's 4 HBM accesses per element. Fused = only 2 (read A,B, write C).
//
// This file benchmarks both versions so you can see the difference.
//
// Compile: nvcc 01b_fused_vec_add_relu.cu -o vec_add_relu && ./vec_add_relu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ---------- UNFUSED VERSION (two kernels, slow) ----------

__global__ void vec_add_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

__global__ void relu_kernel(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

// ---------- FUSED VERSION (one kernel, fast) ----------

__global__ void fused_vec_add_relu_kernel(const float* A, const float* B,
                                          float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // TODO: compute the addition AND the relu in one step
        // Hint: use fmaxf(value, 0.0f)
        C[i] = 0.0f; // fix this
    }
}

int main() {
    int N = 1 << 24;  // 16M elements -- large enough to see timing differences
    size_t bytes = N * sizeof(float);

    // Host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i - N / 2;  // mix of positive and negative
        h_b[i] = (float)(i % 100);
    }

    // Device memory
    float *d_a, *d_b, *d_tmp, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_tmp, bytes);  // intermediate buffer for unfused version
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    // CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;
    int warmup = 5, repeat = 20;

    // ---- Benchmark UNFUSED: vec_add then relu (two kernels) ----
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

    // Verify unfused result
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    int unfused_errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = fmaxf(h_a[i] + h_b[i], 0.0f);
        if (fabsf(h_c[i] - expected) > 1e-3) unfused_errors++;
    }

    // ---- Benchmark FUSED: single kernel ----
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

    // Verify fused result
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    int fused_errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = fmaxf(h_a[i] + h_b[i], 0.0f);
        if (fabsf(h_c[i] - expected) > 1e-3) fused_errors++;
    }

    // ---- Results ----
    printf("N = %d (%.1f MB)\n", N, (float)bytes / 1e6);
    printf("\n");
    printf("Unfused (2 kernels): %.3f ms  [errors: %d]\n", unfused_ms, unfused_errors);
    printf("Fused   (1 kernel):  %.3f ms  [errors: %d]\n", fused_ms, fused_errors);
    printf("\n");
    if (fused_ms > 0 && fused_errors == 0) {
        printf("Speedup: %.2fx\n", unfused_ms / fused_ms);
    } else if (fused_errors > 0) {
        printf("Fused kernel has errors -- fix the TODO!\n");
    }
    printf("\n");
    printf("Why? Unfused: 4 HBM accesses/element (read A,B -> write tmp -> read tmp -> write C)\n");
    printf("     Fused:   2 HBM accesses/element (read A,B -> write C)\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_tmp); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
