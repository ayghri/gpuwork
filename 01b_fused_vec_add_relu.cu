// Exercise 1b: Fused Vector Addition + ReLU
// ============================================
// Compute C[i] = max(A[i] + B[i], 0) in a SINGLE kernel.
//
// Why fuse? Without fusion you need two kernels:
//   1) vec_add:  tmp[i] = A[i] + B[i]   (read A,B from HBM, write tmp to HBM)
//   2) relu:     C[i] = max(tmp[i], 0)   (read tmp from HBM, write C to HBM)
// That's 4 HBM accesses per element. Fused = only 2 (read A,B, write C).
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
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i - N / 2;  // mix of positive and negative
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
