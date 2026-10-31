"""Baseline fixes that dense and Eventful share, and named baseline presets.

Each fix is a separate switch, so its effect on latency, memory and accuracy can be measured on its own
(the "baseline ladder"):

  rope         "stock"   SAM's real-valued RoPE in eager mode (22 elementwise kernels per block)
               "compile" the same Python function compiled with torch.compile (fused into ~1-2 kernels)
  bf16_weights cast the 4 linear layers of every block to bf16 once, instead of autocast/addmm_act casting
  wrap_blocks  replace every Block by EventfulBlock (no gates) - must be free
  compile      torch.compile the whole trunk (mode string), or None
"""
from dataclasses import asdict, dataclass, replace

import torch

import sam3.model.vitdet as vitdet

_ORIG_ROPE = vitdet.apply_rotary_enc_real
_COMPILED_ROPE = None


@dataclass(frozen=True)
class BaselineConfig:
    rope: str = "stock"            # "stock" | "compile"
    bf16_weights: bool = False
    wrap_blocks: bool = False
    compile: str | None = None     # e.g. "max-autotune-no-cudagraphs" | "default" | None

    def as_dict(self):
        return asdict(self)


_B0 = BaselineConfig()
PRESETS = {
    "B0": _B0,                                                     # stock eager
    "B0+bf16w": replace(_B0, bf16_weights=True),                   # ladder step: weight casts removed
    "B0+rope": replace(_B0, rope="compile"),                       # ladder step: RoPE fused
    "B1": replace(_B0, rope="compile", bf16_weights=True),         # primary eager baseline
    "B3": replace(_B0, rope="compile", bf16_weights=True, wrap_blocks=True),  # B1 with EventfulBlock wrappers
    "B2": replace(_B0, bf16_weights=True, compile="max-autotune-no-cudagraphs"),  # primary compiled baseline
    # B2 with torch.compile mode "default": the same mode the compiled Eventful variants (':c') use, so that
    # "compiled gated vs compiled dense" isolates the gate (overhead measurement)
    "B2d": replace(_B0, bf16_weights=True, compile="default"),
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


def rope_mode():
    return "stock" if vitdet.apply_rotary_enc_real is _ORIG_ROPE else "compile"


def cast_linear_weights_bf16(vit):
    """qkv, proj, fc1, fc2 weights and biases to bf16, in place (irreversible within the process).

    Autocast would cast these weights to bf16 anyway (same rounding); addmm_act casts the fc1 weight and bias
    on every call. LayerNorms, pos_embed and patch_embed are left untouched.
    """
    for blk in vit.blocks:
        blk = getattr(blk, "blk", blk)  # works on Block or EventfulBlock
        for lin in (blk.attn.qkv, blk.attn.proj, blk.mlp.fc1, blk.mlp.fc2):
            lin.to(torch.bfloat16)


class Variant:
    """One configuration that can be switched on per call, so several configurations can share one process and be
    timed interleaved (see build_variants). `activate()` sets the process-global RoPE mode and installs the blocks
    this configuration needs on its (possibly shared) trunk copy; then `__call__` runs one frame.

    Eventful variants ("B1:mlp:k2560", eventful/config.py::parse_spec) own an EventfulController with their own gate
    caches; `__call__` goes through `controller.run` (eager flush frames, compiled steady frames if compiled).
    `new_clip()` must be called before the first frame of every clip (resets the gate caches)."""

    def __init__(self, name, cfg, vit, trunk, controller, batch, peers=(), eventful=None, mode="eager"):
        self.name, self.cfg, self.vit, self.trunk, self.controller, self.batch = name, cfg, vit, trunk, controller, batch
        self.peers = peers            # every controller installed on the same trunk copy (shared list)
        self.eventful, self.mode = eventful, mode

    def activate(self):
        set_rope_mode(self.cfg.rope)
        for c in self.peers:
            if c is not self.controller:
                c.set_enabled(False)
        if self.controller is not None:
            self.controller.set_enabled(self.cfg.wrap_blocks or self.eventful is not None)

    def new_clip(self):
        if self.eventful is not None:
            self.controller.reset()

    def __call__(self, x):
        if self.eventful is not None:
            return self.controller.run(x)
        return self.trunk(x)


def parse_variant_name(name):
    """'B1' -> ('B1', 1); 'B1@b8' -> ('B1', 8); 'B1:mlp:k2560' -> ('B1:mlp:k2560', 1)."""
    preset, _, b = name.partition("@b")
    return preset, int(b) if b else 1


def build_variants(names, load_trunk, stats=None):
    """Variants for several configurations in one process, sharing trunk copies where the weights are identical:
    one fp32 copy (B0, B0+rope), one bf16-weight copy (B0+bf16w, B1, B3 and every eager Eventful variant, each with
    its own controller, B1@b8), and one separate copy per compiled configuration (a compiled module must keep the
    block structure it was traced with). Eventful variants must have batch 1."""
    from eventful.config import compile_mode, parse_spec
    from eventful.controller import EventfulController

    copies, variants = {}, []
    for name in names:
        spec, batch = parse_variant_name(name)
        eventful, mode = None, "eager"
        if ":" in spec:
            base, eventful, mode = parse_spec(spec)
            assert batch == 1, "Eventful variants run one frame per call"
            cfg = PRESETS[base]
            assert not cfg.compile, "Eventful variants build on an eager preset; use the ':c' option to compile"
        else:
            cfg = PRESETS[spec]
        key = (cfg.bf16_weights, cfg.compile) if mode == "eager" else (cfg.bf16_weights, name)
        if key not in copies:
            vit = load_trunk()
            if cfg.bf16_weights:
                cast_linear_weights_bf16(vit)
            trunk = torch.compile(vit, mode=cfg.compile, dynamic=False) if cfg.compile else vit
            copies[key] = dict(vit=vit, trunk=trunk, controllers=[], wrap=None)
        c = copies[key]
        controller = None
        if eventful is not None:
            controller = EventfulController(c["vit"], eventful, stats=stats)
            controller.set_enabled(False)
            if mode in ("compile", "cudagraph"):
                controller.set_enabled(True)                    # trace the compiled graph with gated blocks
                controller.attach_compiled(torch.compile(c["vit"], mode=compile_mode(mode), dynamic=False))
                controller.set_enabled(False)
            c["controllers"].append(controller)
        elif cfg.wrap_blocks and not cfg.compile:
            if c["wrap"] is None:
                c["wrap"] = EventfulController(c["vit"], cfg=None)
                c["wrap"].set_enabled(False)
                c["controllers"].append(c["wrap"])
            controller = c["wrap"]
        variants.append(Variant(name, cfg, c["vit"], c["trunk"], controller, batch, peers=c["controllers"],
                                eventful=eventful, mode=mode))
    return variants


def apply_baseline(vit, cfg: BaselineConfig, controller_cls=None):
    """Apply a baseline config to a freshly loaded trunk. Returns (callable trunk, controller or None).

    Call once per process: RoPE patching is process-global and the bf16 cast is in place.
    """
    set_rope_mode(cfg.rope)
    if cfg.bf16_weights:
        cast_linear_weights_bf16(vit)
    controller = None
    if cfg.wrap_blocks:
        if controller_cls is None:
            from eventful.controller import EventfulController as controller_cls
        controller = controller_cls(vit, cfg=None)
    trunk = vit
    if cfg.compile:
        trunk = torch.compile(vit, mode=cfg.compile, dynamic=False)
    return trunk, controller
