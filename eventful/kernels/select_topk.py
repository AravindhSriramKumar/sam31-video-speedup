"""Exact top-K row selection for a gate in one Triton program.

    idx   = the K rows with the largest score, in ascending row order (ties at the K-th value: lowest rows first)
    count = #{score > tau}   (lagged K's change count, optional; replaces the separate count kernel)

PyTorch's topk for a single 5,184-element row runs its single-block radix select (sbtopk::gatherTopK, 36-42 us on
one SM, several passes through global memory). Here one program holds all N scores in registers:

  1. scores are non-negative fp32, so their int32 bit patterns order exactly like the values;
  2. the K-th largest bit pattern T is found by a 31-step binary search on the bits (each step one in-register count);
  3. rows with bits > T are all selected, rows with bits == T fill the remaining places in row order;
  4. an exclusive prefix sum over the selection mask gives each selected row its output slot.

The output is sorted by row, so the gather and scatter that follow walk memory in order.
"""
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

NUM_WARPS = 8                    # 8: 15.6 us, 16: 16.3 us, 32: 24.4 us (N = 5184, K = 3840, with count)


@triton.jit
def _select_topk_kernel(s_ptr, idx_ptr, cnt_ptr, N, K, tau, BLOCK: tl.constexpr, COUNT: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < N
    s = tl.load(s_ptr + offs, mask=m, other=0.0)
    bits = tl.where(m, s.to(tl.int32, bitcast=True), -1)     # padding sorts below every real score (>= 0)
    # largest T with #{bits >= T} >= K, built from the top bit down; stops as soon as #{bits >= T} == K exactly
    # (then the remaining low bits cannot change the selected set: typically after ~13-16 of the 31 steps)
    t = tl.zeros((), dtype=tl.int32)
    b = 30
    hit = 0
    while (b >= 0) & (hit == 0):
        cand = t | (1 << b)
        c = tl.sum((bits >= cand).to(tl.int32), axis=0)
        t = tl.where(c >= K, cand, t)
        hit = (c == K).to(tl.int32)
        b -= 1
    gt = bits > t
    eq = bits == t
    need = K - tl.sum(gt.to(tl.int32), axis=0)
    eq_i = eq.to(tl.int32)
    sel = gt | (eq & ((tl.cumsum(eq_i, axis=0) - eq_i) < need))
    sel_i = sel.to(tl.int32)
    pos = tl.cumsum(sel_i, axis=0) - sel_i
    tl.store(idx_ptr + pos, offs.to(tl.int64), mask=sel)
    if COUNT:
        tl.store(cnt_ptr, tl.sum((m & (s > tau)).to(tl.int64), axis=0))


# a torch.library op, so torch.compile (Inductor, CUDA graphs) calls the kernel inside the compiled graph
@triton_op("eventful::select_topk", mutates_args=("count",))
def _select_topk(scores: torch.Tensor, count: torch.Tensor, k: int, tau: float, with_count: bool) -> torch.Tensor:
    n = scores.shape[0]
    idx = torch.empty(k, device=scores.device, dtype=torch.int64)
    wrap_triton(_select_topk_kernel)[(1,)](scores, idx, count, n, k, tau, BLOCK=triton.next_power_of_2(n),
                                           COUNT=with_count, num_warps=NUM_WARPS)
    return idx


def select_topk(scores, k, tau=None, count=None):
    """scores: (N,) fp32, contiguous, >= 0. Returns idx (k,) int64 in ascending row order. With tau, also writes
    #{scores > tau} into count (0-d int64, in place)."""
    assert scores.dtype == torch.float32 and scores.dim() == 1 and scores.is_contiguous()
    assert 0 < k <= scores.shape[0] <= 16384, (k, scores.shape)
    if count is None:
        count = torch.empty((), device=scores.device, dtype=torch.int64)
    return _select_topk(scores, count, int(k), float(tau) if tau is not None else 0.0, tau is not None)
