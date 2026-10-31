"""Per-token change score for a gate.

    score[i] = || c[i] - ref[i] ||_2                (absolute L2, the gate's default)
    score[i] = || c[i] - ref[i] ||_2 / ||ref[i]||_2  (relative, for statistics and comparison with the prototype)

`c` is the op input of this frame, `ref` the input each token had when it was last recomputed. One Triton
program per row reads both rows in their stored dtype (fp32 or bf16, no .float() copies), accumulates in fp32
and writes one fp32 score: the whole pass reads 2 x N x C x itemsize bytes and writes 4 x N bytes.
`impl="torch"` is the reference implementation (it materialises fp32 copies and the difference).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_change_kernel(c_ptr, r_ptr, out_ptr, C: tl.constexpr, BLOCK: tl.constexpr, RELATIVE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < C
    c = tl.load(c_ptr + row * C + offs, mask=m, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * C + offs, mask=m, other=0.0).to(tl.float32)
    d = c - r
    s = tl.sqrt(tl.sum(d * d, axis=0))
    if RELATIVE:
        s = s / tl.maximum(tl.sqrt(tl.sum(r * r, axis=0)), 1e-6)
    tl.store(out_ptr + row, s)


def gate_score(c, ref, impl="triton", relative=False, out=None):
    """(N,) fp32 scores for rows of c (N, C) against ref (N, C). Both contiguous, same shape."""
    if impl == "torch":
        d = (c.float() - ref.float()).norm(dim=-1)
        return d / ref.float().norm(dim=-1).clamp_min(1e-6) if relative else d
    assert c.is_contiguous() and ref.is_contiguous() and c.shape == ref.shape, (c.shape, ref.shape)
    n, C = c.shape
    if out is None:
        out = torch.empty(n, device=c.device, dtype=torch.float32)
    _l2_change_kernel[(n,)](c, ref, out, C=C, BLOCK=triton.next_power_of_2(C), RELATIVE=relative, num_warps=4)
    return out
