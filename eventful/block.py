"""EventfulBlock: a drop-in wrapper around one SAM 3.1 ViT `Block` (sam3/model/vitdet.py:649).

It holds a reference to the original block (no weight copies) and re-states `Block.forward` and
`Attention.forward` op for op, but with the three per-token linear ops exposed as hooks:

    _qkv(rows)   rows = norm1 output in attention token order (window order in windowed blocks)
    _proj(rows)  rows = merged attention output, same order
    _mlp(rows)   rows = norm2 output, image order

Without a config the hooks call the original layers, so the block is numerically identical to `Block`
(tests/eventful/test_block_equivalence.py checks it bit for bit). With an EventfulConfig the selected
hooks go through a GatedOp (eventful/gates.py) that recomputes only the K most-changed tokens:
    qkv gate   input = norm1 output rows in attention order; caches qkv BEFORE RoPE, so RoPE (and attention)
               still run on all rows every frame
    proj gate  input = merged attention output rows, same order
    mlp gate   input = norm2 output rows, image order; wraps the whole MLP (fc1 -> GELU -> fc2), the 4736-wide
               hidden activation is never cached
The residual adds stay dense.
"""
import torch.nn as nn
import torch.nn.functional as F

from sam3.model.vitdet import window_partition, window_unpartition

from eventful import nvtx
from eventful.config import resolve_k_schedule
from eventful.gates import GatedOp


class EventfulBlock(nn.Module):
    def __init__(self, blk, layer_idx, cfg=None, stats=None, k_schedule=None):
        super().__init__()
        a = blk.attn
        # The re-statement below assumes the SAM 3.1 trunk configuration; fail loudly if it ever changes.
        assert not a.use_rel_pos and not a.cls_token, "rel-pos / cls token not supported"
        assert isinstance(blk.ls1, nn.Identity) and isinstance(blk.ls2, nn.Identity), "LayerScale not supported"
        self.blk, self.idx, self.cfg = blk, layer_idx, cfg
        self.window_size = blk.window_size          # ViT.forward and tools read this attribute
        self.qkv_gate = self.proj_gate = self.mlp_gate = None
        if cfg is not None and cfg.gated_block(layer_idx):
            ks = k_schedule or resolve_k_schedule(cfg)
            ops = {"qkv": a.qkv, "proj": a.proj, "mlp": blk.mlp}
            for t in cfg.gate_types():
                setattr(self, f"{t}_gate", GatedOp(ops[t], f"blk{layer_idx}.{t}", ks[t][layer_idx], policy=cfg.policy,
                                                   threshold=cfg.threshold, score=cfg.score, stats=stats,
                                                   lag_tau=cfg.lag_tau, ref_dtype=cfg.ref_dtype,
                                                   select=cfg.select, fused=cfg.fused_gemm if t == "mlp" else None))

    def forward(self, x):                            # x: (B, 72, 72, 1024) fp32 residual stream
        blk, a = self.blk, self.blk.attn
        with nvtx.rng(f"blk{self.idx}"):
            shortcut = x
            x = blk.norm1(x)
            if blk.window_size > 0:
                H, W = x.shape[1], x.shape[2]
                x, pad_hw = window_partition(x, blk.window_size)      # (9B, 24, 24, 1024)
            B, h, w, C = x.shape
            rows = x.reshape(B * h * w, C)                            # attention token order
            del x                                                     # (intermediates are released as early as in
            with nvtx.rng(f"blk{self.idx}.qkv"):                      #  Block.forward; `del` changes no arithmetic)
                qkv = self._qkv(rows)                                 # (N, 3072) bf16
            del rows
            qkv = qkv.reshape(B, h * w, 3, a.num_heads, -1)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)           # (B, 16, L, 64) each
            del qkv                                                   # v still views its storage until attention
            with nvtx.rng(f"blk{self.idx}.attn"):
                q, k = a._apply_rope(q, k)
                if a.use_fa3:                                         # mirrors Attention.forward
                    from sam3.perflib.fa3 import flash_attn_func
                    o = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
                else:
                    o = F.scaled_dot_product_attention(q, k, v)
            del q, k, v
            o = o.view(B, a.num_heads, h, w, -1).permute(0, 2, 3, 1, 4).reshape(B * h * w, C)
            with nvtx.rng(f"blk{self.idx}.proj"):
                o = self._proj(o)                                     # (N, 1024) bf16
            x = blk.ls1(o.view(B, h, w, C))
            if blk.window_size > 0:
                x = window_unpartition(x, blk.window_size, pad_hw, (H, W))
            x = shortcut + blk.dropout(blk.drop_path(x))
            with nvtx.rng(f"blk{self.idx}.mlp"):
                y = self._mlp(blk.norm2(x).reshape(-1, C))           # image-order rows (B*5184, 1024)
            x = x + blk.dropout(blk.drop_path(blk.ls2(y.view_as(x))))
            return x

    # plain ops without a gate, the GatedOp otherwise
    def _qkv(self, rows):
        return self.qkv_gate(rows) if self.qkv_gate is not None else self.blk.attn.qkv(rows)

    def _proj(self, rows):
        return self.proj_gate(rows) if self.proj_gate is not None else self.blk.attn.proj(rows)

    def _mlp(self, rows):
        return self.mlp_gate(rows) if self.mlp_gate is not None else self.blk.mlp(rows)
