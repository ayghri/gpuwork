# Exercise 5: Triton Vector Addition
# ====================================
# Implement vector addition in Triton.
# Compare the code with the CUDA version from Exercise 1.
#
# Run locally: python 05_triton_vec_add.py
# Requires: pip install triton torch

import torch
import triton
import triton.language as tl


@triton.jit
def vec_add_kernel(a_ptr, b_ptr, c_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(c_ptr + offsets, a + b, mask=mask)


def vec_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape and a.is_cuda
    c = torch.empty_like(a)
    N = a.numel()
    BLOCK = 1024
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    vec_add_kernel[grid](a, b, c, N, BLOCK=BLOCK)
    return c


if __name__ == "__main__":
    N = 1 << 20
    a = torch.randn(N, device="cuda")
    b = torch.randn(N, device="cuda")

    c_triton = vec_add(a, b)
    c_torch = a + b

    print(f"Max error: {(c_triton - c_torch).abs().max().item():.2e}")
    print("PASSED" if torch.allclose(c_triton, c_torch) else "FAILED")
