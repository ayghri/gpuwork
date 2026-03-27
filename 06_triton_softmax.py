# Exercise 6: Triton Fused Softmax
# ==================================
# Implement row-wise softmax in a single Triton kernel.
# This fuses max, subtract, exp, sum, and divide into one pass.
#
# Run locally: python 06_triton_softmax.py
# Requires: pip install triton torch

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(input_ptr, output_ptr, n_cols, stride,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols

    # Load one row from global memory
    row_ptr = input_ptr + row * stride
    x = tl.load(row_ptr + offsets, mask=mask, other=-float("inf"))

    # Compute softmax entirely in SRAM — no intermediate writes to HBM
    x_max = tl.max(x, axis=0)
    numerator = tl.exp(x - x_max)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    # Store result
    out_ptr = output_ptr + row * stride
    tl.store(out_ptr + offsets, result, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2 and x.is_cuda
    rows, cols = x.shape
    # BLOCK must be a power of 2 >= cols
    BLOCK = triton.next_power_of_2(cols)
    out = torch.empty_like(x)
    softmax_kernel[(rows,)](x, out, cols, x.stride(0), BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    x = torch.randn(128, 512, device="cuda")

    out_triton = softmax(x)
    out_torch = torch.softmax(x, dim=-1)

    print(f"Max error: {(out_triton - out_torch).abs().max().item():.2e}")
    print("PASSED" if torch.allclose(out_triton, out_torch, atol=1e-5) else "FAILED")
