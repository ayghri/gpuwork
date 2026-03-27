// Exercise 1d: PTX Special Registers
// ====================================
// CUDA provides built-in variables like blockIdx.x and threadIdx.x,
// but under the hood these are PTX special registers. You can read
// them directly with inline PTX assembly.
//
// This exercise shows how to access these registers and introduces
// others that CUDA doesn't expose directly (like %smid and %laneid).
//
// PTX special registers reference:
//   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers
//
// Compile: nvcc 01d_special_registers.cu -o special_regs && ./special_regs

#include <stdio.h>

__device__ unsigned int get_tid_x() {
    unsigned int tid;
    asm("mov.u32 %0, %%tid.x;" : "=r"(tid));
    return tid;
}

__device__ unsigned int get_tid_y() {
    unsigned int tid;
    asm("mov.u32 %0, %%tid.y;" : "=r"(tid));
    return tid;
}

__device__ unsigned int get_ctaid_x() {
    unsigned int ctaid;
    asm("mov.u32 %0, %%ctaid.x;" : "=r"(ctaid));
    return ctaid;
}

__device__ unsigned int get_ctaid_y() {
    unsigned int ctaid;
    asm("mov.u32 %0, %%ctaid.y;" : "=r"(ctaid));
    return ctaid;
}

__device__ unsigned int get_ntid_x() {
    unsigned int ntid;
    asm("mov.u32 %0, %%ntid.x;" : "=r"(ntid));
    return ntid;
}

__device__ unsigned int get_smid() {
    unsigned int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ unsigned int get_nsmid() {
    unsigned int nsmid;
    asm("mov.u32 %0, %%nsmid;" : "=r"(nsmid));
    return nsmid;
}

__device__ unsigned int get_laneid() {
    unsigned int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

__device__ unsigned int get_warpid() {
    unsigned int warpid;
    asm("mov.u32 %0, %%warpid;" : "=r"(warpid));
    return warpid;
}

__device__ unsigned int get_clock() {
    unsigned int clock_val;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(clock_val));
    return clock_val;
}

// Kernel that prints info about each thread's identity
__global__ void identify_kernel() {
    // Only print from a few threads to avoid flooding the output
    unsigned int ptx_tid_x   = get_tid_x();      // same as threadIdx.x
    unsigned int ptx_ctaid_x = get_ctaid_x();     // same as blockIdx.x
    unsigned int ptx_ntid_x  = get_ntid_x();      // same as blockDim.x
    unsigned int smid        = get_smid();         // which SM is this running on?
    unsigned int laneid      = get_laneid();       // position within the warp (0-31)
    unsigned int warpid      = get_warpid();       // warp index within the block

    // Compute global index using PTX registers (same as blockIdx.x * blockDim.x + threadIdx.x)
    unsigned int global_id = ptx_ctaid_x * ptx_ntid_x + ptx_tid_x;

    // Print from thread 0 of each block
    if (ptx_tid_x == 0) {
        printf("Block %u: SM=%u, nSM=%u, global_id=%u, warp=%u, lane=%u\n",
               ptx_ctaid_x, smid, get_nsmid(), global_id, warpid, laneid);
    }

    // Also print from thread 33 to show it's in warp 1, lane 1
    if (ptx_tid_x == 33) {
        printf("  Block %u, Thread 33: warp=%u, lane=%u (warp 1, lane 1)\n",
               ptx_ctaid_x, warpid, laneid);
    }
}

// Kernel that verifies PTX registers match CUDA built-ins
__global__ void verify_kernel(int* errors) {
    unsigned int ptx_tid   = get_tid_x();
    unsigned int ptx_ctaid = get_ctaid_x();
    unsigned int ptx_ntid  = get_ntid_x();

    // These should be identical
    if (ptx_tid != threadIdx.x)   atomicAdd(errors, 1);
    if (ptx_ctaid != blockIdx.x)  atomicAdd(errors, 1);
    if (ptx_ntid != blockDim.x)   atomicAdd(errors, 1);

    // Verify lane ID is threadIdx.x % 32
    unsigned int laneid = get_laneid();
    if (laneid != threadIdx.x % 32) atomicAdd(errors, 1);

    // Verify warp ID is threadIdx.x / 32
    unsigned int warpid = get_warpid();
    if (warpid != threadIdx.x / 32) atomicAdd(errors, 1);
}

int main() {
    printf("=== PTX Special Registers Demo ===\n\n");

    // Part 1: Print thread identities
    printf("--- Thread identity (4 blocks x 64 threads) ---\n");
    identify_kernel<<<4, 64>>>();
    cudaDeviceSynchronize();

    // Part 2: Verify PTX registers match CUDA built-ins
    printf("\n--- Verifying PTX registers match CUDA built-ins ---\n");
    int *d_errors;
    cudaMalloc(&d_errors, sizeof(int));
    cudaMemset(d_errors, 0, sizeof(int));

    verify_kernel<<<16, 256>>>(d_errors);

    int h_errors;
    cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Mismatches: %d (should be 0)\n", h_errors);
    cudaFree(d_errors);

    // Part 3: Quick reference
    printf("\n--- PTX Special Register Quick Reference ---\n");
    printf("%%tid.{x,y,z}    = threadIdx.{x,y,z}   (thread index in block)\n");
    printf("%%ntid.{x,y,z}   = blockDim.{x,y,z}    (block dimensions)\n");
    printf("%%ctaid.{x,y,z}  = blockIdx.{x,y,z}    (block index in grid)\n");
    printf("%%nctaid.{x,y,z} = gridDim.{x,y,z}     (grid dimensions)\n");
    printf("%%laneid          (no CUDA equivalent)   lane within warp (0-31)\n");
    printf("%%warpid          (no CUDA equivalent)   warp within block\n");
    printf("%%smid            (no CUDA equivalent)   which SM is running this\n");
    printf("%%nsmid           (no CUDA equivalent)   number of SMs on the GPU\n");
    printf("%%clock           (use clock() in CUDA)  cycle counter\n");
    printf("%%clock64         (use clock64())        64-bit cycle counter\n");

    return 0;
}

// Key takeaways:
//
// 1. CUDA built-ins (threadIdx, blockIdx, etc.) are syntactic sugar
//    for PTX special registers (%%tid, %%ctaid, etc.)
//
// 2. PTX exposes registers that CUDA doesn't:
//    - %%smid: which SM your block landed on (useful for debugging scheduling)
//    - %%laneid: your position in the warp (0-31), useful for warp-level tricks
//    - %%warpid: which warp you're in within the block
//    - %%clock/%%clock64: cycle counters for micro-benchmarking
//
// 3. The "=r" constraint means unsigned 32-bit register (vs "=f" for float)
//
// 4. Use %% (double percent) for PTX registers in inline asm
//    (single % is used for operand placeholders like %0, %1)
