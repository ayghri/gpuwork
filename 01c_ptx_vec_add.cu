// Exercise 1c: Vector Addition using inline PTX
// ===============================================
// Same vec_add, but the addition is done with inline PTX assembly.
// This shows you exactly what instructions the GPU executes.
//
// NOTE: Inline PTX requires nvcc. It does NOT work on LeetGPU (which
//       uses a simulator). Use godbolt.org or compile locally with nvcc.
//
// PTX reference: docs.nvidia.com/cuda/parallel-thread-execution
// Explore interactively: godbolt.org (select CUDA as language)
//
// Compile: nvcc 01c_ptx_vec_add.cu -o ptx_vec_add && ./ptx_vec_add

#include <stdio.h>
#include <stdlib.h>

__global__ void vec_add_ptx_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float a = A[i];
        float b = B[i];
        float c;

        // Inline PTX: add two floats
        // "add.f32 %0, %1, %2" means: %0 = %1 + %2
        //   %0 is output (=c), %1 is input (=a), %2 is input (=b)
        asm("add.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));

        C[i] = c;
    }
}

// For comparison: the normal CUDA version compiles to the same PTX,
// but writing it explicitly helps you understand what's happening.
__global__ void vec_add_normal_kernel(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];  // nvcc generates: add.f32 %fN, %fM, %fK
    }
}

int main() {
    int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(2 * i);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    vec_add_ptx_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) errors++;
    }
    printf("PTX vec_add -- Errors: %d / %d\n", errors, N);

    // Try generating the full PTX file:
    //   nvcc -ptx 01c_ptx_vec_add.cu -o 01c_ptx_vec_add.ptx
    // Then open the .ptx file and search for "add.f32"

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    return 0;
}
