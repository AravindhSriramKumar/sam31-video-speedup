"""Test tolerances derived from the measured bf16 noise floor.

Nothing here is a hand-picked number: the floor is read from results/eventful/reference/noise_floor.json, a saved measurement. Three rules:

  same kernels          bitwise equal, or rel_err <= SAME_KERNELS_REL (1e-6)
  different kernels     rel_err <= FLOOR_FACTOR x (B=2 vs B=1 rel_err at that block, max over measured frames)
                        and median token cosine >= the floor's worst median cosine
  bf16 elementwise      every element within 1 bf16 ulp of an fp32 reference
  compiled vs eager     rel_err <= FLOOR_FACTOR x (B2 vs B1 rel_err, max over the 40 reference frames) and median
                        token cosine >= the B2 vs B1 worst median cosine (results/eventful/reference/
                        feature_accuracy.json): the envelope torch.compile's fused kernels alone produce
"""
import json

import torch

from eventful.env import paths

NOISE_FLOOR_FILE = paths.EVENTFUL_RESULTS / "reference" / "noise_floor.json"
SAME_KERNELS_REL = 1e-6
FLOOR_FACTOR = 1.5


def noise_floor():
    if not NOISE_FLOOR_FILE.exists():
        raise FileNotFoundError(f"{NOISE_FLOOR_FILE} missing")
    with open(NOISE_FLOOR_FILE) as fh:
        return json.load(fh)


def different_kernels_limit(block=-1, comparison="batch2_vs_batch1"):
    """(max rel_err allowed, min median cosine allowed) at a block (-1 = trunk output = block 31)."""
    nf = noise_floor()[comparison]["per_block"]
    b = nf[block]
    return FLOOR_FACTOR * b["rel_err"]["max"], b["cos_median"]["min"]


def check_different_kernels(stats, block=-1, comparison="batch2_vs_batch1"):
    rel_lim, cos_lim = different_kernels_limit(block, comparison)
    ok = stats["rel_err"] <= rel_lim and stats["cos_median"] >= cos_lim
    return ok, dict(rel_err=stats["rel_err"], rel_limit=rel_lim, cos_median=stats["cos_median"], cos_limit=cos_lim)


def bf16_ulp(ref):
    """Size of one bf16 ulp at |ref| (8 significant bits): 2^(floor(log2|ref|) - 7)."""
    a = ref.float().abs().clamp_min(torch.finfo(torch.float32).tiny)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def within_ulps(x, ref, ulps=1.0, scale=None):
    """Fraction of elements and max error, in bf16 ulps, of x vs an fp32 reference.

    The ulp is taken at |ref| by default, or at `scale` (same shape as ref) if given. For a rotation such as RoPE
    the right scale is the norm of the rotated (a, b) pair: an output component that nearly cancels
    (a*cos - b*sin ~ 0) has a tiny |ref| and a tiny ulp, but its rounding error is set by the pair's magnitude,
    in the fp32 reference as much as in any other implementation.
    """
    err = (x.float() - ref.float()).abs()
    u = bf16_ulp(ref if scale is None else scale)
    in_ulps = err / u
    return dict(frac_within=(in_ulps <= ulps + 1e-6).float().mean().item(), max_ulps=in_ulps.max().item(),
                max_abs=err.max().item())


def pair_norm(t):
    """For a tensor whose last dim holds adjacent (real, imag) pairs: each element replaced by its pair's norm."""
    p = t.float().reshape(*t.shape[:-1], -1, 2)
    return p.norm(dim=-1, keepdim=True).expand_as(p).reshape(t.shape)


FEATURE_ACCURACY_FILE = paths.EVENTFUL_RESULTS / "reference" / "feature_accuracy.json"


def compiled_vs_eager_limit():
    """(max rel_err allowed, min median cosine allowed) for a compiled trunk against an eager reference."""
    c = json.loads(FEATURE_ACCURACY_FILE.read_text())["comparisons"]["B2_vs_B1"]["overall"]
    return FLOOR_FACTOR * c["rel_err"]["max"], c["cos_median"]["min"]


def check_compiled_vs_eager(stats):
    rel_lim, cos_lim = compiled_vs_eager_limit()
    ok = stats["rel_err"] <= rel_lim and stats["cos_median"] >= cos_lim
    return ok, dict(rel_err=stats["rel_err"], rel_limit=rel_lim, cos_median=stats["cos_median"], cos_limit=cos_lim)
