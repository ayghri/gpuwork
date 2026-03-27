# Exercise 7: Triton Matrix Multiplication
# ==========================================
# Implement tiled matmul in Triton. Compare with the CUDA version.
# Triton handles shared memory, coalescing, and Tensor Cores for you.
#
# Run locally: python 07_triton_matmul.py
# Requires: pip install triton torch

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this block's tile
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Pointers to the first tile of A and B
    A_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Tiled loop over K — Triton compiles this to shared memory + Tensor Cores
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptrs,
                     mask=(rm[:, None] < M) & (rk[None, :] + k < K),
                     other=0.0)
        b = tl.load(B_ptrs,
                     mask=(rk[:, None] + k < K) & (rn[None, :] < N),
                     other=0.0)
        acc += tl.dot(a, b)  # compiles to mma instructions
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Store result
    C_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C_ptrs, acc, mask=mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


if __name__ == "__main__":
    M, N, K = 512, 512, 512
    a = torch.randn(M, K, device="cuda", dtype=torch.float32)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32)

    c_triton = matmul(a, b)
    c_torch = a @ b

    print(f"Max error: {(c_triton - c_torch).abs().max().item():.2e}")
    print("PASSED" if torch.allclose(c_triton, c_torch, atol=1e-2) else "FAILED")
