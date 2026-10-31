"""Eventful gating configuration and K schedules.

A gate sits around one per-token op of a block (qkv, proj or the whole MLP). Each frame it scores how much every
token's input changed since the token was last recomputed, recomputes the K highest-scoring tokens, and reuses the
cached output for the rest. K can differ per block and per gate type ("K schedule").
"""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

N_TOKENS = 5184          # 72 x 72 patches at 1008 x 1008, batch 1
N_BLOCKS = 32
GATE_TYPES = ("qkv", "proj", "mlp")


@dataclass(frozen=True)
class EventfulConfig:
    gate_mlp: bool = True
    gate_qkv: bool = False
    gate_proj: bool = False
    k_schedule: str = "uniform:1280"   # "uniform:K" | "linear:K_first:K_last" | "file:<path.json>"
    k_align: int = 128                 # every K rounded to the nearest multiple, clamped to [k_align, N]
    policy: str = "topk"               # "topk" | "threshold" (ablation only: needs a CPU-GPU sync per gate)
    threshold: float = 1.0
    score: str = "triton"              # "triton" | "torch" (reference)
    blocks: tuple | None = None        # None = all 32 blocks gated
    refresh_every: int | None = None   # dense flush every R frames (None = never)
    collect_stats: bool = False
    # lagged K: when lag_tau is set, the gate ranks tokens by
    # relative change ||c - ref|| / ||ref|| and counts how many exceed lag_tau; the controller sets one K per frame
    # for all gates from the previous frame's mean count, rounded up to a multiple of k_bucket. The K schedule's
    # value is K0, used until a count exists (first steady frames after a flush).
    lag_tau: float | None = None
    k_bucket: int = 512
    # dtype of the gate's reference cache and of the gate input it is compared with
    ref_dtype: str = "fp32"            # "fp32" | "bf16"
    # dense fallback: with lagged K, a frame whose K reaches fallback_k runs as a dense flush (compiled), which
    # also refreshes every cache; the next frame starts again at K0. Flush frames then run compiled too.
    fallback_k: int | None = None
    # top-K selection: "torch" (scores.topk) | "triton" (eventful/kernels/select_topk.py: one
    # program, exact, ascending indices, lagged K's count in the same kernel)
    select: str = "torch"
    # CUTLASS kernels for the MLP gate's GEMMs (kernels/fused_gemm/): None | "tuned" (no indexing) | "scatter" (fc2
    # writes rows by index) | "full" (fc1 also reads rows by index). Steady frames only; dense frames are unchanged.
    fused_gemm: str | None = None

    def as_dict(self):
        return asdict(self)

    def gate_types(self):
        return tuple(t for t in GATE_TYPES if getattr(self, f"gate_{t}"))

    def gated_block(self, i):
        return self.blocks is None or i in self.blocks


def _align(k, align, n):
    """K rounded half-up to a multiple of `align`, at least `align`; any request >= n means exactly n (all tokens).
    (Python's round() rounds half to even: round(5184 / 128) = 40 would silently turn K = N into 5120.)"""
    if k >= n:
        return n
    return int(min(n, max(align, align * int(k / align + 0.5))))


def resolve_k_schedule(cfg: EventfulConfig, n=N_TOKENS, n_blocks=N_BLOCKS):
    """{"qkv": [32 ints], "proj": [...], "mlp": [...]} for the schedule string in cfg."""
    kind, _, arg = cfg.k_schedule.partition(":")
    if kind == "uniform":
        ks = {t: [float(arg)] * n_blocks for t in GATE_TYPES}
    elif kind == "linear":
        a, b = (float(v) for v in arg.split(":"))
        ks = {t: [a + (b - a) * i / (n_blocks - 1) for i in range(n_blocks)] for t in GATE_TYPES}
    elif kind == "file":
        raw = json.loads(open(arg).read())
        ks = {t: [float(v) for v in raw[t]] if t in raw else [float(n)] * n_blocks for t in GATE_TYPES}
    else:
        raise ValueError(cfg.k_schedule)
    return {t: [_align(k, cfg.k_align, n) for k in v] for t, v in ks.items()}


SCHEDULE_DIR = str(Path(__file__).resolve().parents[1] / "results/eventful/schedules")


def compile_mode(ev_mode):
    """torch.compile mode for an Eventful execution mode ("compile" -> default, "cudagraph" -> reduce-overhead)."""
    return "reduce-overhead" if ev_mode == "cudagraph" else "default"


def parse_spec(spec):
    """Variant names for the harnesses: "<base preset>:<gates>:<K>[:options]", e.g.
        B1:mlp:k2560              MLP gates, uniform K = 2560, eager, on top of preset B1
        B1:mlp+qkv:lin1280-3840   MLP and qkv gates, K growing linearly from block 0 to block 31
        B1:mlp:sched=cal90        K schedule from results/eventful/schedules/cal90.json
        B1:mlp:k2560:r8           dense refresh every 8 frames
        B1:mlp:k2560:c            steady frames through torch.compile (mode "default"); flush frames eager
        B1:mlp:k2560:cg           as :c, with CUDA graphs (torch.compile mode "reduce-overhead")
        B1:mlp:k2560:torchscore   PyTorch reference score instead of the Triton kernel
        B1:mlp:k2560:b24-31       gate only blocks 24..31 (the others stay dense)
        B1:mlp:k3840:lag0.6:c     lagged per-frame K from a relative-change threshold of 0.6 (K0 = 3840)
        B1:mlp:k3840:bf16ref      bf16 reference cache and gate input
        B1:mlp:k3840:lag0.6:fb4608:c   dense (flush) frame instead of gating when the lagged K reaches 4608
        B1:mlp:k3840:lag0.6:tk:c  Triton top-K selection (and count) instead of torch.topk
        B1:mlp:k3840:fgfull:c     CUTLASS GEMMs in the MLP gate: fgtuned (no indexing), fgscatter, fgfull
    Returns (base preset name, EventfulConfig, execution mode "eager" | "compile")."""
    base, gates, kspec, *opts = spec.split(":")
    kw = {f"gate_{g}": True for g in gates.split("+")}
    kw.setdefault("gate_mlp", False)
    if kspec.startswith("k"):
        kw["k_schedule"] = f"uniform:{int(kspec[1:])}"
    elif kspec.startswith("lin"):
        a, b = kspec[3:].split("-")
        kw["k_schedule"] = f"linear:{int(a)}:{int(b)}"
    elif kspec.startswith("sched="):
        kw["k_schedule"] = f"file:{SCHEDULE_DIR}/{kspec[6:]}.json"
    else:
        raise ValueError(spec)
    mode = "eager"
    for o in opts:
        if o.startswith("r"):
            kw["refresh_every"] = int(o[1:])
        elif o == "c":
            mode = "compile"
        elif o == "cg":                                 # compiled steady frames replayed as CUDA graphs
            mode = "cudagraph"
        elif o == "torchscore":
            kw["score"] = "torch"
        elif o.startswith("lag"):
            kw["lag_tau"] = float(o[3:])
        elif o.startswith("fb"):
            kw["fallback_k"] = int(o[2:])
        elif o == "bf16ref":
            kw["ref_dtype"] = "bf16"
        elif o == "tk":
            kw["select"] = "triton"
        elif o in ("fgtuned", "fgscatter", "fgfull"):
            kw["fused_gemm"] = o[2:]
        elif o.startswith("b") and "-" in o:
            a, b = (int(v) for v in o[1:].split("-"))
            kw["blocks"] = tuple(range(a, b + 1))
        else:
            raise ValueError(f"unknown option {o} in {spec}")
    return base, EventfulConfig(**kw), mode
