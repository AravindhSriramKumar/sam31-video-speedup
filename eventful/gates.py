"""GatedOp: one Eventful gate and its cache around one per-token op.

    frame after reset ("flush"):  out = op(c) for all N rows;   ref = c
    every later frame:            score = ||c - ref|| per row  (eventful.kernels.gate_score)
                                  idx   = the K rows with the largest score (top-K, no CPU-GPU sync)
                                  out[idx] = op(c[idx]);  ref[idx] = c[idx]
                                  return out  (all N rows: K fresh, N - K reused from earlier frames)

`ref` holds, for every token, the input it had when it was last recomputed ("last-sent" semantics), so a slowly
drifting token accumulates change until it is selected. Rows are tokens in a fixed order chosen by the caller
(window order or image order); the gate never sees spatial structure.

Buffers are allocated at the first flush, from the actual input and output (shape, dtype), and then kept: later
flushes copy into the same storage, so addresses stay fixed (needed for CUDA graphs).

`out` is returned by alias: the caller must consume it within the same frame and must not modify it. In
EventfulBlock every consumer (reshape/permute views, RoPE, SDPA, residual add) creates new tensors.
"""
import torch
import torch.nn as nn

from eventful import nvtx
from eventful.kernels.gate_score import gate_score
from eventful.kernels.select_topk import select_topk


class GatedOp(nn.Module):
    def __init__(self, op, name, k, policy="topk", threshold=1.0, score="triton", stats=None, lag_tau=None,
                 ref_dtype="fp32", select="torch", fused=None):
        super().__init__()
        assert fused in (None, "tuned", "scatter", "full"), fused
        self.fused = fused                             # CUTLASS kernels for the MLP gate (eventful/kernels/fused_gemm.py)
        assert policy in ("topk", "threshold"), policy
        self.op, self.name, self.k = op, name, int(k)
        self.policy, self.threshold, self.score_impl, self.stats = policy, threshold, score, stats
        # lagged K: rank by relative change, count tokens above lag_tau into self.count (a device
        # tensor, read by the controller after the frame, never synchronised here); k is set by the controller
        self.lag_tau = lag_tau
        self.register_buffer("count", None, persistent=False)
        self.count_on_flush = False                    # dense fallback: still count changes against the old ref
        # "fp32": the cache keeps the gate input's own dtype (fp32 in the trunk); "bf16": cache and input cast to bf16
        self.ref_dtype = {"fp32": None, "bf16": torch.bfloat16}[ref_dtype]
        assert select in ("torch", "triton"), select
        self.select = select                           # top-K kernel; "triton" also does the count
        # allocated at the first flush; non-persistent buffers (cache state, not model state, never in a state_dict),
        # so that torch.compile treats them as static tensors with fixed addresses (needed for CUDA graphs)
        self.register_buffer("ref", None, persistent=False)
        self.register_buffer("out", None, persistent=False)
        self.needs_flush = True
        # (No per-call Python counters here: under torch.compile a mutated int attribute is a guarded constant, so
        #  every frame would recompile the graph.)
        # profiler labels for the gate's steps, e.g. "blk3.mlp.score" (eventful/nvtx.py; free when labels are off)
        self._lab = {s: f"{name}.{s}" for s in ("flush", "score", "select", "gather", "op", "scatter")}

    def reset(self):
        """Next frame is a dense flush. Allocations are kept (fixed addresses)."""
        self.needs_flush = True

    @torch.no_grad()
    def flush(self, c):
        with nvtx.rng(self._lab["flush"]):
            return self._flush(c)

    def _flush(self, c):
        if self.ref_dtype is not None:
            c = c.to(self.ref_dtype)                       # the op sees exactly what the steady path will feed it
        if self.count_on_flush and self.ref is not None:   # before ref is overwritten: keeps lagged K tracking
            scores = gate_score(c, self.ref, impl=self.score_impl, relative=True)
            self.count.copy_((scores > self.lag_tau).sum())
        y = self.op(c)
        if self.ref is None:
            self.ref = torch.empty_like(c, memory_format=torch.contiguous_format)
            self.out = torch.empty_like(y, memory_format=torch.contiguous_format)
            if self.lag_tau is not None or self.select == "triton":   # the Triton top-K always writes a count
                self.count = torch.zeros((), dtype=torch.int64, device=c.device)
        self.ref.copy_(c)
        self.out.copy_(y)
        self.needs_flush = False
        if self.stats is not None:
            self.stats.record_flush(self.name)
        return self.out

    @torch.no_grad()
    def forward(self, c):                          # c: (N, C_in) -> (N, C_out), dense
        if self.needs_flush:
            return self.flush(c)
        L = self._lab
        c = c.to(self.ref_dtype).contiguous() if self.ref_dtype is not None else c.contiguous()
        with nvtx.rng(L["score"]):
            scores = gate_score(c, self.ref, impl=self.score_impl, relative=self.lag_tau is not None)
            if self.lag_tau is not None and not self._fused_select():
                self.count.copy_((scores > self.lag_tau).sum())
        with nvtx.rng(L["select"]):
            if self._fused_select():
                idx = select_topk(scores, self.k, self.lag_tau, self.count)
            else:
                idx = self._select(scores)
        if self.fused is not None:
            return self._fused_step(c, idx)
        with nvtx.rng(L["gather"]):
            c_sel = c.index_select(0, idx)
        with nvtx.rng(L["op"]):
            y_sel = self.op(c_sel)
        if self.stats is not None:                 # before ref is updated: compares against the old reference
            self.stats.record(self.name, scores, idx, c, self.ref)
        with nvtx.rng(L["scatter"]):
            self.ref.index_copy_(0, idx, c_sel)
            self.out.index_copy_(0, idx, y_sel.to(self.out.dtype))
        return self.out

    def _fused_step(self, c, idx):
        """Steady frame through the CUTLASS kernels. fused = "tuned": compact gather and scatter around them; "scatter":
        fc2 writes out[idx] directly; "full": fc1 also reads c[idx] directly, so only the ref update stays as a copy."""
        from eventful.kernels.fused_gemm import FC1_SHAPE, fc2_shape, fused_fc1, fused_fc2

        L, m = self._lab, self.op
        full, scatter = self.fused == "full", self.fused in ("full", "scatter")
        k = self.k
        with nvtx.rng(L["gather"]):
            ii = idx.to(torch.int32)
            c_sel = c.index_select(0, idx)
            self.ref.index_copy_(0, idx, c_sel)
        with nvtx.rng(L["op"]):
            W1, b1, W2, b2 = m.fc1.weight, m.fc1.bias, m.fc2.weight, m.fc2.bias
            h = torch.empty(k, W1.shape[0], device=c.device, dtype=c.dtype)
            fused_fc1(W1, b1, c if full else c_sel, ii, h, full, FC1_SHAPE)
            if scatter:
                fused_fc2(W2, b2, h, self.out, ii, True, fc2_shape(k))
                return self.out
            y = torch.empty(k, W2.shape[0], device=c.device, dtype=self.out.dtype)
            fused_fc2(W2, b2, h, y, ii, False, fc2_shape(k))
        with nvtx.rng(L["scatter"]):
            self.out.index_copy_(0, idx, y)
        return self.out

    def _fused_select(self):
        return self.select == "triton" and self.policy == "topk"

    def _select(self, scores):
        if self.policy == "topk":
            return scores.topk(self.k, sorted=False).indices
        return (scores > self.threshold).nonzero().squeeze(1)    # ablation; nonzero() synchronises

    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.ref, self.out) if t is not None)

    def extra_repr(self):
        return f"name={self.name}, k={self.k}, policy={self.policy}, score={self.score_impl}, select={self.select}"
