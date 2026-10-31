"""Op- and layer-selective EP-ViT: codec-guided reuse applied to chosen operations of chosen blocks.

`EPViT` keeps only the M recomputed rows in the residual stream: every operation of every block (qkv, attention
queries, proj, MLP) runs on the same M rows, and the other tokens take the trunk output of their (warped) source.
That has no notion of reusing one operation but not another, or of running some blocks dense. `EPViTSelective`
keeps the full N-row residual stream instead, and reuses only the outputs of the selected operations:

  * a reused operation of block l keeps an N-row output buffer (qkv before RoPE, proj, MLP). On a sparse frame the
    frame plan's M rows are recomputed and written into it; the other rows keep their cached value, and the rows of
    matchable tokens are first copied from their motion-vector source (the same warp `EPViT` applies to K/V);
  * every other operation, attention (all N queries, RoPE on all rows) and the norms run dense on all N rows.

The frame plan (static / matchable / unmatchable classes, Eq. 11 update link, budget k, warp sources, flushes on
the first frame, I-frames and non-consecutive frames, the output cache for repeated frames) is `EPViT`'s own:
this class only replaces `_run`. A flush, or a plan with every token selected, is the original ViT.

    EPViTSelective(vit, cfg, ops=("qkv",))                    # QKV-only reuse in all 32 blocks
    EPViTSelective(vit, cfg, blocks=range(16, 32))            # blocks 16-31 reuse qkv, proj and MLP; 0-15 dense
"""
import torch
import torch.nn.functional as F

from epvit.codec.schedule import N_TOK, WIN, ScheduleConfig
from epvit.model.epvit import N_WIN, EPViT
from sam3.model.vitdet import get_abs_pos, window_partition, window_unpartition

OPS = ("qkv", "proj", "mlp")


class EPViTSelective(EPViT):
    def __init__(self, vit, cfg: ScheduleConfig = ScheduleConfig(mode="dense"), ops=OPS, blocks=None, **kw):
        assert set(ops) <= set(OPS) and ops, ops
        super().__init__(vit, cfg, **kw)
        self.ops = tuple(op for op in OPS if op in ops)
        self.blocks = tuple(range(len(vit.blocks))) if blocks is None else tuple(sorted(blocks))
        self.buf = {}

    def gated(self, l, op):
        return op in self.ops and l in self.blocks

    def release(self):
        super().release()
        self.buf = {}

    def state_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.buf.values())

    def _run(self, x, plan):
        vit = self.vit
        P = self._device_plan(plan, x.device)
        dense = plan.flush or P["full"]
        if not dense and len(plan.warp_dst):
            src, dst = P["warp_src"], P["warp_dst"]
            for b in self.buf.values():
                b.index_copy_(0, dst, b.index_select(0, src))
        t = vit.patch_embed(x)
        t = t + get_abs_pos(vit.pos_embed, vit.pretrain_use_cls_token, (t.shape[1], t.shape[2]), vit.retain_cls_token,
                            tiling=vit.tile_abs_pos)
        t = vit.ln_pre(t)
        for l, blk in enumerate(vit.blocks):
            t = self._sel_block(l, blk, t, None if dense else P["idx"])
        out = vit.ln_post(t).permute(0, 3, 1, 2)
        self.OUT = out                                           # marks the state as primed (EPViT._frame checks it)
        return out

    def _op(self, l, name, fn, h, idx):
        """fn on the rows of h (N, C_in): dense, or recompute rows idx into the op's N-row output buffer."""
        if not self.gated(l, name):
            return fn(h)
        if idx is None:
            y = fn(h)
            b = self.buf.get((l, name))
            if b is None or b.dtype != y.dtype:
                self.buf[(l, name)] = b = torch.empty_like(y)
            b.copy_(y)
            return y
        b = self.buf[(l, name)]
        if idx.numel():
            b.index_copy_(0, idx, fn(h.index_select(0, idx)))
        return b

    def _sel_block(self, l, blk, x, idx):
        a = blk.attn
        _, H, W, C = x.shape
        N, nh = H * W, a.num_heads
        shortcut = x
        qkv = self._op(l, "qkv", a.qkv, blk.norm1(x).reshape(N, C), idx)

        # dense attention in SAM's layout (window-major in windowed blocks), RoPE on all rows
        qkv = qkv.view(1, H, W, -1)
        ws = blk.window_size
        if ws > 0:
            qkv, pad_hw = window_partition(qkv, ws)
        B, h, w = qkv.shape[:3]
        q, k, v = qkv.reshape(B, h * w, 3, nh, -1).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = a._apply_rope(q, k)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.view(B, nh, h, w, -1).permute(0, 2, 3, 1, 4).reshape(B, h, w, -1)
        if ws > 0:
            o = window_unpartition(o, ws, pad_hw, (H, W))

        p = self._op(l, "proj", a.proj, o.reshape(N, C), idx)
        x = shortcut + blk.dropout(blk.drop_path(blk.ls1(p.view(1, H, W, C))))
        y = self._op(l, "mlp", blk.mlp, blk.norm2(x).reshape(N, C), idx)
        return x + blk.dropout(blk.drop_path(blk.ls2(y.view(1, H, W, C))))

    def trunk_macs(self, log_row, C=1024, mlp_ratio=4.625, patch=14):
        """Executed MACs of one encoded frame, from its log row (rows actually processed)."""
        if log_row["mode"] == "cached":
            return 0
        M = log_row["M"] if log_row["mode"] == "sparse" else N_TOK
        ff = int(C * mlp_ratio)
        per_row = {"qkv": 3 * C * C, "proj": C * C, "mlp": 2 * C * ff}
        total = N_TOK * 3 * patch * patch * C
        for l, blk in enumerate(self.vit.blocks):
            total += sum((M if self.gated(l, op) else N_TOK) * r for op, r in per_row.items())
            total += 2 * N_TOK * N_TOK * C if blk.window_size == 0 else 2 * N_WIN * (WIN * WIN) ** 2 * C
        return total

