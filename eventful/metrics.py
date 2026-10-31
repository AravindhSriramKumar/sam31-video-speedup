"""Feature-error metrics between a test trunk output and a reference trunk output.

All metrics compare tensors of shape [B, C, H, W] (trunk output: [B, 1024, 72, 72]) or block outputs
[B, H, W, C] (pass channel_dim=-1). A "token" is one spatial position, i.e. one 1024-dim vector.
"""
import torch
import torch.nn.functional as F


def feature_stats(test, ref, channel_dim=1):
    """rel_err = ||test - ref||_2 / ||ref||_2 over the whole tensor; token cosine statistics over all tokens."""
    t, r = test.float(), ref.float()
    rel = ((t - r).norm() / r.norm().clamp_min(1e-12)).item()
    cos = F.cosine_similarity(t, r, dim=channel_dim).flatten()
    q = torch.quantile(cos, torch.tensor([0.001, 0.01, 0.5], device=cos.device)).tolist()
    return dict(
        rel_err=rel,
        max_abs=(t - r).abs().max().item(),
        cos_mean=cos.mean().item(),
        cos_median=q[2],
        cos_p1=q[1],
        cos_p01=q[0],
        cos_min=cos.min().item(),
        frac_cos_below_0p99=(cos < 0.99).float().mean().item(),
        bitwise_equal=bool(torch.equal(test, ref)),
    )


def aggregate(rows, keys=("rel_err", "cos_median", "cos_p1", "cos_min", "frac_cos_below_0p99")):
    """mean / max / min over a list of feature_stats dicts."""
    out = {}
    for k in keys:
        v = [r[k] for r in rows]
        out[k] = dict(mean=sum(v) / len(v), max=max(v), min=min(v))
    out["n"] = len(rows)
    out["all_bitwise_equal"] = all(r.get("bitwise_equal", False) for r in rows)
    return out
