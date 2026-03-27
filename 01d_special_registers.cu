// Exercise 1d: Thread Identity and Special Registers
// ====================================================
// Every CUDA thread can query its own identity: block, thread,
// warp, and lane. This exercise explores these built-in variables
// and shows how they map to PTX special registers.
//
// This version works on LeetGPU. Inline PTX examples are at the
// bottom (nvcc-only, use godbolt.org or compile locally).
//
// Compile: nvcc 01d_special_registers.cu -o special_regs && ./special_regs

#include <stdio.h>

// ============================================================
// PART 1: CUDA built-ins (works everywhere, including LeetGPU)
// ============================================================

__global__ void identify_kernel() {
    // Standard CUDA built-ins
    unsigned int tid_x  = threadIdx.x;   // PTX: %tid.x
    unsigned int tid_y  = threadIdx.y;   // PTX: %tid.y
    unsigned int bid_x  = blockIdx.x;    // PTX: %ctaid.x
    unsigned int bdim_x = blockDim.x;    // PTX: %ntid.x

    // Derived identities (no direct CUDA built-in, but easy to compute)
    unsigned int laneid = threadIdx.x % 32;      // PTX: %laneid
    unsigned int warpid = threadIdx.x / 32;      // PTX: %warpid
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Print from thread 0 of each block
    if (tid_x == 0) {
        printf("Block %u: bdim=%u, global_id=%u, warp=%u, lane=%u\n",
               bid_x, bdim_x, global_id, warpid, laneid);
    }

    // Print from thread 33 to show it's in warp 1, lane 1
    if (tid_x == 33) {
        printf("  Block %u, Thread 33: warp=%u, lane=%u\n",
               bid_x, warpid, laneid);
    }

    // Print from the last thread in warp 0 to show lane 31
    if (tid_x == 31) {
        printf("  Block %u, Thread 31: warp=%u, lane=%u\n",
               bid_x, warpid, laneid);
    }
}

__global__ void verify_identities(int* errors) {
    // Verify derived values match expectations
    unsigned int laneid = threadIdx.x % 32;
    unsigned int warpid = threadIdx.x / 32;

    // warpSize is a built-in constant (always 32 on NVIDIA GPUs)
    if (laneid != threadIdx.x % warpSize) atomicAdd(errors, 1);
    if (warpid != threadIdx.x / warpSize) atomicAdd(errors, 1);

    // Global ID formula
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int expected = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id != expected) atomicAdd(errors, 1);
}

int main() {
    printf("=== Thread Identity Demo ===\n\n");

    // Part 1: Print thread identities
    printf("--- 4 blocks x 64 threads ---\n");
    identify_kernel<<<4, 64>>>();
    cudaDeviceSynchronize();

    // Part 2: Verify
    printf("\n--- Verifying identities ---\n");
    int *d_errors;
    cudaMalloc(&d_errors, sizeof(int));
    cudaMemset(d_errors, 0, sizeof(int));
    verify_identities<<<16, 256>>>(d_errors);
    int h_errors;
    cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Mismatches: %d (should be 0)\n", h_errors);
    cudaFree(d_errors);

    // Part 3: Quick reference
    printf("\n--- CUDA Built-in to PTX Register Mapping ---\n");
    printf("threadIdx.x/y/z  <-->  %%tid.x/y/z     thread index in block\n");
    printf("blockDim.x/y/z   <-->  %%ntid.x/y/z    block dimensions\n");
    printf("blockIdx.x/y/z   <-->  %%ctaid.x/y/z   block index in grid\n");
    printf("gridDim.x/y/z    <-->  %%nctaid.x/y/z  grid dimensions\n");
    printf("threadIdx.x %% 32 <-->  %%laneid         lane within warp (0-31)\n");
    printf("threadIdx.x / 32 <-->  %%warpid         warp within block\n");
    printf("(no built-in)    <-->  %%smid           which SM runs this block\n");
    printf("clock()          <-->  %%clock          cycle counter\n");

    return 0;
}

// ============================================================
// PART 2: Inline PTX (nvcc only -- does NOT work on LeetGPU)
//
// To try these, compile locally with nvcc or paste into godbolt.org.
// LeetGPU's simulator does not support inline PTX asm().
//
// The inline PTX syntax:
//   asm("instruction %0, %1;" : "=r"(output) : "r"(input));
//
//   "=r" means: output, unsigned 32-bit register
//   "=f" means: output, 32-bit float register
//   %%  means: literal % (PTX register prefix)
//
// Examples (uncomment and compile with nvcc):
//
// __device__ unsigned int ptx_get_tid_x() {
//     unsigned int r;
//     asm("mov.u32 %0, %%tid.x;" : "=r"(r));
//     return r;  // same as threadIdx.x
// }
//
// __device__ unsigned int ptx_get_smid() {
//     unsigned int r;
//     asm("mov.u32 %0, %%smid;" : "=r"(r));
//     return r;  // which SM -- no CUDA equivalent!
// }
//
// __device__ unsigned int ptx_get_laneid() {
//     unsigned int r;
//     asm("mov.u32 %0, %%laneid;" : "=r"(r));
//     return r;  // lane in warp -- exact HW register
// }
//
// __device__ unsigned int ptx_get_clock() {
//     unsigned int r;
//     asm volatile("mov.u32 %0, %%clock;" : "=r"(r));
//     return r;  // cycle counter
// }
// ============================================================
