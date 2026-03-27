# GPU Workshop Exercises

**Workshop repository**: https://github.com/ayghri/gpuwork

## Platforms

**LeetGPU** -- browser-based CUDA playground, no setup needed
- https://leetgpu.com
- Sign in with Google or GitHub
- Paste any `.cu` file, submit, and see results + generated PTX
- Free tier: 5 submissions/day

**Compiler Explorer (Godbolt)** -- interactive CUDA to PTX/SASS viewer
- https://godbolt.org
- Select CUDA as language, pick a GPU target (e.g. sm_80 for A100)
- Type code on the left, see PTX/SASS on the right in real time
- Great for understanding what the compiler does with your code

**PTX ISA Reference** -- the full instruction set documentation
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

**CUDA Programming Guide**
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

## Local setup (optional)

If you have an NVIDIA GPU and `nvcc` installed:

```bash
# Build a single exercise
nvcc -O2 01_vec_add_solution.cu -o vec_add && ./vec_add

# Generate PTX output
nvcc -O2 -ptx 01_vec_add_solution.cu -o vec_add.ptx

# Or use the Makefile
make vec_add           # compile
make vec_add_cpu       # CPU baseline (needs gcc + OpenMP)
make ptx_vec_add       # inline PTX exercise
make 01_vec_add_solution.ptx   # generate PTX file from any .cu
```

Triton exercises require `pip install triton torch` and a GPU:
```bash
python3 05_triton_vec_add.py
python3 06_triton_softmax.py
python3 07_triton_matmul.py
```

## Exercise list

### CUDA (paste into LeetGPU)

| # | File | Topic | Concepts |
|---|------|-------|----------|
| 00 | `00_cpu_vec_add.cu` | CPU baseline | Sequential loop, OpenMP `#pragma omp parallel for` |
| 01 | `01_vec_add.cu` | Vector addition | `__global__`, `blockIdx`, `threadIdx`, boundary check |
| 01b | `01b_fused_vec_add_relu.cu` | Fused vec add + ReLU | Kernel fusion, memory wall, `fmaxf` |
| 01c | `01c_ptx_vec_add.cu` | Inline PTX vec add | `asm()`, PTX `add.f32`, special registers |
| 01d | `01d_special_registers.cu` | PTX special registers | `%smid`, `%laneid`, `%ctaid`, `%tid`, `%nsmid` |
| 02 | `02_naive_matmul.cu` | Naive matrix multiply | 2D grid/blocks, dot product, row-major indexing |
| 03 | `03_tiled_matmul.cu` | Tiled matmul | `__shared__`, `__syncthreads()`, cache reuse |
| 04 | `04_bank_conflict_fix.cu` | Bank conflict padding | Shared memory banks, +1 padding trick |

Solutions have the `_solution.cu` suffix.

### Triton (run locally with GPU + PyTorch)

| # | File | Topic |
|---|------|-------|
| 05 | `05_triton_vec_add.py` | Vector addition in Triton |
| 06 | `06_triton_softmax.py` | Fused row-wise softmax |
| 07 | `07_triton_matmul.py` | Tiled matrix multiplication |

## Tips

- After each LeetGPU submission, click the **PTX tab** to see the generated assembly
- Use `nvcc -ptx file.cu` to generate PTX locally
- On Godbolt, add `-arch=sm_80` to compiler flags for A100-targeted PTX
- Compare naive matmul PTX (all `ld.global`) vs tiled matmul PTX (`ld.shared` in inner loop)
- Look for `mma.sync` in PTX when Tensor Cores are used
