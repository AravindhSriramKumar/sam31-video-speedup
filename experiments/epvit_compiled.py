"""EP-ViT under torch.compile, with the compile settings of the dense compiled baseline.

Dense B2d is compiled as torch.compile(vit, mode="default", dynamic=False). These subclasses give EP-ViT the same
treatment without touching epvit/: the per-frame GPU work runs in one torch.compile(mode="default", dynamic=False) graph, and
only the host plan, the variable-length warp copies and the output-buffer write stay eager.

  CompiledEPViT       patch embedding, gather of the M plan rows, the 32 blocks (K/V buffer writes, query-subset
                      attention, RoPE) and ln_post in one graph. Flush frames go through the same graph with M = N.
                      Shapes are static per (M, m): M is the budget k (or N on a flush), and the padded per-window
                      query count m is rounded up to a multiple of M_BUCKET (one graph per bucket, all compiled by
                      `warm()` before anything is timed). Padding rows are zero queries whose outputs are dropped, so
                      the M real rows are unchanged; the frame log keeps the plan's true m, so the MAC counter is
                      the same as the eager runs'.
  CompiledEPViTSelective
                      the 32 blocks of EPViTSelective (N-row residual stream, gated ops on the k plan rows) in one
                      graph; flush frames run eager (B1: compiled RoPE).
"""
import dataclasses

import numpy as np
import torch
import torch._dynamo.config as dynamo_config

from epvit import GRID
from epvit.codec.schedule import N_TOK, WIN, FramePlan
from epvit.model.epvit import _WIN_ID, _LOCAL, EPViT
from epvit.model.selective import EPViTSelective
from sam3.model.vitdet import get_abs_pos

M_BUCKET = 64
COMPILE = dict(mode="default", dynamic=False)


def _raise_recompile_limits():
    dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 64)
    dynamo_config.accumulated_recompile_limit = max(getattr(dynamo_config, "accumulated_recompile_limit", 256), 512)


def m_bucket(m):
    return min(WIN * WIN, -(-m // M_BUCKET) * M_BUCKET)


def _embed(vit, x):
    t = vit.patch_embed(x)
    t = t + get_abs_pos(vit.pos_embed, vit.pretrain_use_cls_token, (t.shape[1], t.shape[2]), vit.retain_cls_token,
                        tiling=vit.tile_abs_pos)
    return vit.ln_pre(t)


def _flush_plan():
    e = np.empty(0, np.int64)
    return FramePlan(True, np.arange(N_TOK), e, e, WIN * WIN, {})


def _spread_plan(M, m):
    """A sparse plan with M rows and at most m per window (for compiling the (M, m) graph; values are irrelevant)."""
    per = np.full(9, M // 9)
    per[:M % 9] += 1
    assert per.max() <= m
    idx = np.sort(np.concatenate([np.nonzero(_WIN_ID == w)[0][:c] for w, c in enumerate(per)]))
    e = np.empty(0, np.int64)
    return FramePlan(False, idx.astype(np.int64), e, e, m, {})


class _PerBlock(list):
    """Per-block K or V buffers; `dtype` is what EPViT._block checks before writing into self.K[l]."""

    @property
    def dtype(self):
        return self[0].dtype


class CompiledEPViT(EPViT):
    def __init__(self, vit, cfg, **kw):
        super().__init__(vit, cfg, **kw)
        _raise_recompile_limits()
        self._core_c = torch.compile(self._core, **COMPILE)

    def _core(self, x, P):
        vit = self.vit
        flat = _embed(vit, x).reshape(N_TOK, -1)
        h = flat if P["full"] else flat.index_select(0, P["idx"])
        for l, blk in enumerate(vit.blocks):
            h = self._block(l, blk, h, P)
        return vit.ln_post(h)

    def state_bytes(self):
        bufs = [*(self.K or ()), *(self.V or ()), *(() if self.OUT is None else (self.OUT,))]
        return sum(t.numel() * t.element_size() for t in bufs)

    def _run(self, x, plan):
        if self.K is None or self.OUT is None:                   # first frame of the process: allocates the buffers
            out = super()._run(x, plan)
            # one buffer per block instead of views of one (L, N, nh, hd) tensor: inside a compiled graph, a write
            # into a view of a graph input is applied by copying back the whole base (340 MB per block)
            self.K, self.V = _PerBlock(k.clone() for k in self.K), _PerBlock(v.clone() for v in self.V)
            return out
        dev = x.device
        P = self._device_plan(dataclasses.replace(plan, max_m_win=m_bucket(plan.max_m_win)), dev)
        if not plan.flush and len(plan.warp_dst):
            src, dst = P["warp_src"], P["warp_dst"]
            for b in (*self.K, *self.V):
                b.index_copy_(0, dst, b.index_select(0, src))
            self.OUT.index_copy_(0, dst, self.OUT.index_select(0, src))
        C = self.OUT.shape[1]
        if P["M"] == 0:
            return self.OUT.view(1, GRID, GRID, C).permute(0, 3, 1, 2).clone()
        h = self._core_c(x, P)
        if P["full"]:
            self.OUT.copy_(h)
        else:
            self.OUT.index_copy_(0, P["idx"], h)
        return self.OUT.view(1, GRID, GRID, C).permute(0, 3, 1, 2).clone()

    @torch.inference_mode()
    def warm(self, x):
        """Compile every graph this budget can meet: the flush (M = N) and each m bucket of an M = k frame.
        The first flush runs eager and allocates the buffers. Under bf16 autocast, as in SAM's predictor."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(2):
                self._run(x, _flush_plan())
            k = self.cfg.k
            if k < N_TOK:
                for m in range(m_bucket(-(-k // 9)), WIN * WIN + 1, M_BUCKET):
                    self._run(x, _spread_plan(k, m))
        self.reset()


class CompiledEPViTSelective(EPViTSelective):
    def __init__(self, vit, cfg, **kw):
        super().__init__(vit, cfg, **kw)
        _raise_recompile_limits()
        self._core_c = torch.compile(self._core, **COMPILE)

    def _core(self, x, idx):
        vit = self.vit
        t = _embed(vit, x)
        for l, blk in enumerate(vit.blocks):
            t = self._sel_block(l, blk, t, idx)
        return vit.ln_post(t).permute(0, 3, 1, 2)

    def _run(self, x, plan):
        if plan.flush or len(plan.idx) == N_TOK or not self.buf:  # eager flush (fills the op buffers)
            return super()._run(x, plan)
        P = self._device_plan(plan, x.device)
        if len(plan.warp_dst):
            src, dst = P["warp_src"], P["warp_dst"]
            for b in self.buf.values():
                b.index_copy_(0, dst, b.index_select(0, src))
        out = self._core_c(x, P["idx"])
        self.OUT = out
        return out

    @torch.inference_mode()
    def warm(self, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._run(x, _flush_plan())
            if self.cfg.k < N_TOK:
                self._run(x, _spread_plan(self.cfg.k, WIN * WIN))
        self.reset()
