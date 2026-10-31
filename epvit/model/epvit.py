"""EP-ViT: drop-in replacement of SAM 3.1's ViT trunk that recomputes only the tokens the codec priors mark as changed.

`EPViT.from_vit(vit, cfg)` wraps SAM 3.1's own ViT (`sam3.model.vitdet.ViT`) and uses its weights as they are. It has
the trunk's interface: `forward(tensor_list) -> [features (B, 1024, 72, 72)]`, plus a side channel for the codec
priors (`set_video`) and the frame ids (`set_frame_ids`, filled by `epvit.model.install`).

Per frame (paper Sec. 4.2, mapped to SAM 3.1):
  * patch embedding, the tiled absolute position embedding and ln_pre run on the whole frame (0.1% of the MACs),
    then the M tokens of the frame plan are gathered (Index Selector);
  * each block runs norm1 / qkv / proj / norm2 / MLP on the M rows only. Their K and V (before RoPE) are written into
    the block's N-row buffers (Buffer module, Eq. 7);
  * attention: only the M query rows are needed downstream, so the paper's sparse row/column update of QK^T and
    delta update of AV (Eqs. 8-10) reduce to the M queries attending to all buffered keys/values ("query-subset
    attention"): (1, M) x N in the global blocks, and in each 24 x 24 window that window's selected queries x its
    576 keys in the windowed blocks (queries padded to the frame's largest window count);
  * RoPE is applied to the gathered queries with their own position frequencies and to the full key buffer, so
    a moved (warped) key is rotated for its new position;
  * matchable tokens: before the blocks run, every K/V buffer row and the output row of a warped token are copied
    from its motion-vector source (gather, then scatter);
  * the block-31 output rows are written into the output buffer; the trunk returns a copy of it.
A flush (first frame, I-frame, non-consecutive frame) is the same code path with all N tokens selected, and it is
numerically the original ViT (tests/test_epvit.py checks this).

Repeated frames: SAM 3.1's pipeline encodes frames 0 and 1 for the prompt and then 0, 1, 2, ... again. A frame that
was already encoded in this session is served from a small output cache, without touching the reuse state.
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from epvit import GRID
from epvit.codec.schedule import N_TOK, WIN, ScheduleConfig, Scheduler
from sam3.model.data_misc import NestedTensor
from sam3.model.vitdet import get_abs_pos
from sam3.sam.rope import complex_mult

N_WIN = (GRID // WIN) ** 2
# raster token id of window-major position (window (wr, wc) row-major, then local (r, c) row-major), as window_partition
_r, _c = np.divmod(np.arange(N_TOK), GRID)
_WIN_ID = (_r // WIN) * (GRID // WIN) + _c // WIN
_LOCAL = (_r % WIN) * WIN + _c % WIN
WIN_PERM = np.argsort(_WIN_ID * WIN * WIN + _LOCAL, kind="stable")


def _rope(x, attn, freq_idx):
    """RoPE of x (..., L, hd) with the attention module's own frequencies, gathered at freq_idx (shape broadcastable
    to x[..., 0::2] without the head dim, e.g. (L,) or (W, 1, L)). Same arithmetic as sam3's apply_rotary_enc(_real)."""
    if attn.use_rope_real:
        fr, fi = attn.freqs_cis_real[freq_idx], attn.freqs_cis_imag[freq_idx]
        xr = x.float().reshape(*x.shape[:-1], -1, 2)[..., 0]
        xi = x.float().reshape(*x.shape[:-1], -1, 2)[..., 1]
        return complex_mult(xr, xi, fr, fi).flatten(-2).type_as(x)
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(xc * attn.freqs_cis[freq_idx]).flatten(-2).type_as(x)


class EPViT(nn.Module):
    def __init__(self, vit, cfg: ScheduleConfig = ScheduleConfig(mode="dense"), cache_frames=4):
        super().__init__()
        self.vit = vit
        self.cfg = cfg
        self.channel_list = vit.channel_list
        self.cache_frames = cache_frames
        self.priors = self.sched = None
        self.video = None
        self.K = self.V = self.OUT = None
        self.frame_ids = None
        self.log = []                      # one dict per encoded frame (host-side, no sync)
        self.session = 0                   # incremented by every reset (= every SAM reset_state once installed)
        # the trunk's only output is the last block's (block 31 = the last global block), as in SAM 3.1
        assert not vit.retain_cls_token and not vit.return_interm_layers
        assert vit.full_attn_ids[-1] == len(vit.blocks) - 1
        self._win_perm = None
        self.reset()

    @classmethod
    def from_vit(cls, vit, cfg: ScheduleConfig = ScheduleConfig(mode="dense"), **kw):
        return cls(vit, cfg, **kw)

    # ---------------------------------------------------------------- side channel
    def set_config(self, cfg: ScheduleConfig):
        self.cfg = cfg
        if self.priors is not None:
            self.sched = Scheduler(self.priors, cfg)
        self.reset()

    def set_video(self, name, priors):
        """priors: dict from epvit.codec.schedule.load_priors (or None: always dense)."""
        self.video, self.priors = name, priors
        self.sched = Scheduler(priors, self.cfg) if priors is not None else None
        self.reset()

    def set_frame_ids(self, ids):
        self.frame_ids = list(ids) if ids is not None else None

    def reset(self):
        """New session: forget the reuse state and the output cache (buffers stay allocated)."""
        if self.sched is not None:
            self.sched.reset()
        self.cache = OrderedDict()
        self.session += 1

    def release(self):
        self.reset()
        self.K = self.V = self.OUT = None

    def state_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.K, self.V, self.OUT) if t is not None)

    # ---------------------------------------------------------------- forward
    def forward(self, tensor_list):
        x = getattr(tensor_list, "tensors", tensor_list)
        mask = getattr(tensor_list, "mask", None)
        if mask is not None and mask.any():
            raise NotImplementedError("padding masks are not supported")
        ids, self.frame_ids = self.frame_ids, None
        if ids is not None:
            assert len(ids) == x.shape[0], (ids, x.shape)
        feats = torch.cat([self._frame(x[b:b + 1], None if ids is None else ids[b]) for b in range(x.shape[0])])
        return [NestedTensor(feats, None)] if isinstance(tensor_list, NestedTensor) else [feats]

    def _frame(self, x, fid):
        if self.sched is None or fid is None:
            # no priors / unknown frame: the original trunk, and the reuse state is no longer valid
            if self.sched is not None:
                self.sched.reset()
            self.log.append({"video": self.video, "session": self.session, "fid": fid, "mode": "dense_no_prior",
                             "M": N_TOK})
            return self.vit(x)[-1]
        last = self.sched.last
        if fid in self.cache and (last is None or fid != last + 1):
            self.log.append({"video": self.video, "session": self.session, "fid": fid, "mode": "cached", "M": 0})
            return self.cache[fid]
        plan = self.sched.step(fid)
        if not plan.flush and self.OUT is None:
            raise RuntimeError("sparse frame before any flush")
        out = self._run(x, plan)
        self.cache[fid] = out
        while len(self.cache) > self.cache_frames:
            self.cache.popitem(last=False)
        self.log.append({"video": self.video, "session": self.session, "fid": fid,
                         "mode": "flush" if plan.flush else "sparse",
                         "max_m_win": plan.max_m_win, **plan.stats})
        return out

    def _device_plan(self, plan, device):
        """Host plan -> GPU index tensors, one pinned H2D copy, sizes known on the host (no sync)."""
        idx = plan.idx
        M = len(idx)
        m = plan.max_m_win
        w = _WIN_ID[idx]
        order = np.argsort(w, kind="stable")                     # idx is raster-sorted -> window-local raster order
        rank = np.empty(M, np.int64)
        starts = np.searchsorted(w[order], np.arange(N_WIN))
        rank[order] = np.arange(M) - starts[w[order]]
        slot = w * m + rank                                       # padded window-major slot of every selected row
        qwin = np.full(N_WIN * m, M, np.int64)                    # row M = zero padding
        qwin[slot] = np.arange(M)
        qwin_local = np.zeros(N_WIN * m, np.int64)
        qwin_local[slot] = _LOCAL[idx]
        parts = [idx, plan.warp_src, plan.warp_dst, qwin, qwin_local, slot]
        host = torch.from_numpy(np.concatenate(parts).astype(np.int64)).pin_memory()
        dev = host.to(device, non_blocking=True)
        out, o = {}, 0
        for k, p in zip(("idx", "warp_src", "warp_dst", "qwin", "qwin_local", "slot"), parts):
            out[k] = dev[o:o + len(p)]
            o += len(p)
        out["M"], out["max_m"], out["full"] = M, m, M == N_TOK
        return out

    def _run(self, x, plan):
        vit = self.vit
        dev = x.device
        P = self._device_plan(plan, dev)
        if self._win_perm is None or self._win_perm.device != dev:
            self._win_perm = torch.from_numpy(WIN_PERM).to(dev)
            self._win_local = torch.arange(WIN * WIN, device=dev)          # key positions inside a window
            self._arange = torch.arange(N_TOK, device=dev)

        t = vit.patch_embed(x)
        t = t + get_abs_pos(vit.pos_embed, vit.pretrain_use_cls_token, (t.shape[1], t.shape[2]),
                            vit.retain_cls_token, tiling=vit.tile_abs_pos)
        t = vit.ln_pre(t)
        C = t.shape[-1]
        flat = t.reshape(N_TOK, C)

        if plan.flush:
            assert P["full"]
        elif len(plan.warp_dst):
            src, dst = P["warp_src"], P["warp_dst"]
            self.K.index_copy_(1, dst, self.K.index_select(1, src))
            self.V.index_copy_(1, dst, self.V.index_select(1, src))
            self.OUT.index_copy_(0, dst, self.OUT.index_select(0, src))

        if P["M"] == 0:                                           # every token reused (static or warped)
            return self.OUT.view(1, GRID, GRID, C).permute(0, 3, 1, 2).clone()
        h = flat if P["full"] else flat.index_select(0, P["idx"])
        for l, blk in enumerate(vit.blocks):
            h = self._block(l, blk, h, P)
        h = vit.ln_post(h)                                       # identity in SAM 3.1 (ln_post=False)

        if self.OUT is None or self.OUT.dtype != h.dtype:
            self.OUT = torch.empty(N_TOK, C, dtype=h.dtype, device=dev)
        if P["full"]:
            self.OUT.copy_(h)
        else:
            self.OUT.index_copy_(0, P["idx"], h)
        # the original trunk returns x.permute(0, 3, 1, 2) of a (1, 72, 72, C) tensor: same strides, fresh memory
        return self.OUT.view(1, GRID, GRID, C).permute(0, 3, 1, 2).clone()

    def _block(self, l, blk, h, P):
        attn = blk.attn
        M, C = h.shape
        nh = attn.num_heads
        hd = C // nh
        shortcut = h
        qkv = attn.qkv(blk.norm1(h)).reshape(M, 3, nh, hd)
        q, k, v = qkv.unbind(1)                                  # (M, nh, hd), pre-RoPE

        if self.K is None or self.K.dtype != k.dtype:
            L = len(self.vit.blocks)
            self.K = torch.empty(L, N_TOK, nh, hd, dtype=k.dtype, device=k.device)
            self.V = torch.empty_like(self.K)
        Kl, Vl = self.K[l], self.V[l]
        if P["full"]:
            Kl.copy_(k)
            Vl.copy_(v)
        else:
            Kl.index_copy_(0, P["idx"], k)
            Vl.index_copy_(0, P["idx"], v)

        if blk.window_size > 0:
            W, m, S = N_WIN, P["max_m"], WIN * WIN
            kw = Kl.index_select(0, self._win_perm).view(W, S, nh, hd).transpose(1, 2)
            vw = Vl.index_select(0, self._win_perm).view(W, S, nh, hd).transpose(1, 2)
            qp = q if (P["full"] and m == S) else torch.cat([q, q.new_zeros(1, nh, hd)])
            qw = qp.index_select(0, P["qwin"]).view(W, m, nh, hd).transpose(1, 2)
            qw = _rope(qw, attn, P["qwin_local"].view(W, 1, m))
            kw = _rope(kw, attn, self._win_local)
            o = F.scaled_dot_product_attention(qw, kw, vw)       # (W, nh, m, hd)
            o = o.transpose(1, 2).reshape(W * m, C).index_select(0, P["slot"])
        else:
            qg = _rope(q.unsqueeze(0).transpose(1, 2), attn, P["idx"])          # (1, nh, M, hd)
            kg = _rope(Kl.unsqueeze(0).transpose(1, 2), attn, self._arange)    # (1, nh, N, hd)
            vg = Vl.unsqueeze(0).transpose(1, 2)
            o = F.scaled_dot_product_attention(qg, kg, vg).transpose(1, 2).reshape(M, C)

        h = shortcut + attn.proj(o)
        return h + blk.mlp(blk.norm2(h))


def trunk_macs(log_row, C=1024, mlp_ratio=4.625, depth=32, global_blocks=(7, 15, 23, 31), patch=14):
    """Executed MACs of one encoded frame of EPViT, from its log row (rows actually processed)."""
    if log_row["mode"] == "cached":
        return 0
    M = log_row["M"]
    m = log_row.get("max_m_win", WIN * WIN) if log_row["mode"] != "dense_no_prior" else WIN * WIN
    lin = M * (3 * C * C + C * C + 2 * C * int(C * mlp_ratio))
    attn_g = 2 * M * N_TOK * C
    attn_w = 2 * N_WIN * m * WIN * WIN * C
    attn = len(global_blocks) * attn_g + (depth - len(global_blocks)) * attn_w
    return depth * lin + attn + N_TOK * 3 * patch * patch * C
