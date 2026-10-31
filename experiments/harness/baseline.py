"""Named baseline presets (each fix is a separate switch):

  rope         "stock"   SAM's real-valued RoPE in eager mode (22 elementwise kernels per block)
               "compile" the same Python function compiled with torch.compile (fused into ~1-2 kernels)
  bf16_weights cast the 4 linear layers of every block to bf16 once, instead of autocast/addmm_act casting
  compile      torch.compile the whole trunk (mode string), or None
"""
from dataclasses import asdict, dataclass

import torch

import sam3.model.vitdet as vitdet

_ORIG_ROPE = vitdet.apply_rotary_enc_real
_COMPILED_ROPE = None


@dataclass(frozen=True)
class BaselineConfig:
    rope: str = "stock"            # "stock" | "compile"
    bf16_weights: bool = False
    compile: str | None = None     # e.g. "default" | None

    def as_dict(self):
        return asdict(self)


PRESETS = {
    "B1": BaselineConfig(rope="compile", bf16_weights=True),       # eager base EP-ViT is built on
    "B2d": BaselineConfig(bf16_weights=True, compile="default"),   # compiled dense reference, torch.compile "default"
}


def set_rope_mode(mode):
    """Swap the module-global `apply_rotary_enc_real` that vitdet.Attention._apply_rope looks up at call time.

    Only the ViT trunk is affected: the tracker's attention (sam3/model/decoder.py) imports its own reference.
    """
    global _COMPILED_ROPE
    if mode == "stock":
        vitdet.apply_rotary_enc_real = _ORIG_ROPE
    elif mode == "compile":
        # dynamic=False compiles one variant per input shape: (windowed, global) x every trunk batch size seen.
        # The pipeline at grounding batch 8 meets batch sizes 1..8 (a video's last chunk holds T mod 8 frames),
        # i.e. up to 16 variants, above torch's default limit of 8 (FailOnRecompileLimitHit). Allow 64.
        import torch._dynamo.config as dynamo_config
        dynamo_config.recompile_limit = max(dynamo_config.recompile_limit, 64)
        if _COMPILED_ROPE is None:
            _COMPILED_ROPE = torch.compile(_ORIG_ROPE, dynamic=False, fullgraph=True)
        vitdet.apply_rotary_enc_real = _COMPILED_ROPE
    else:
        raise ValueError(mode)


def cast_linear_weights_bf16(vit):
    """qkv, proj, fc1, fc2 weights and biases to bf16, in place (irreversible within the process).

    Autocast would cast these weights to bf16 anyway (same rounding); addmm_act casts the fc1 weight and bias
    on every call. LayerNorms, pos_embed and patch_embed are left untouched.
    """
    for blk in vit.blocks:
        blk = getattr(blk, "blk", blk)  # a wrapped block keeps the original in .blk
        for lin in (blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2):
            lin.to(torch.bfloat16)


def apply_baseline(vit, cfg: BaselineConfig):
    """Apply a baseline config to a freshly loaded trunk. Returns the callable trunk.

    Call once per process: RoPE patching is process-global and the bf16 cast is in place.
    """
    set_rope_mode(cfg.rope)
    if cfg.bf16_weights:
        cast_linear_weights_bf16(vit)
    return torch.compile(vit, mode=cfg.compile, dynamic=False) if cfg.compile else vit
