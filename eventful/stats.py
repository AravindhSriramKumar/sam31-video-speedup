"""Statistics mode for gates. Never active while timing: it synchronises (.item()/.cpu()).

Per gate and steady frame, with c the op input, ref the inputs at last recompute and s = ||c - ref|| per token:
  k                 tokens recomputed
  thr               score of the weakest recomputed token (the effective threshold of top-K this frame)
  s_median, s_p90, s_max      distribution of the absolute change over all tokens
  rel_median, rel_p90         relative change s / ||ref|| over all tokens
  frac_rel_lt_1e-2            tokens whose relative change is below 1% ("unchanged")
  stale_rel_mean, stale_rel_max   relative change of the tokens NOT recomputed: the error the gate accepts
  stale_energy      sum of s^2 over skipped tokens / sum of s^2 over all tokens: share of the input change that
                    the gate leaves unrecomputed (0 = everything that changed was recomputed)
"""
from collections import defaultdict

import torch


class GateStats:
    def __init__(self):
        self.rows = defaultdict(list)          # gate name -> list of per-frame dicts
        self.flushes = defaultdict(int)

    def record_flush(self, name):
        self.flushes[name] += 1

    @torch.no_grad()
    def record(self, name, scores, idx, c, ref):
        s = scores.float()
        rn = ref.float().norm(dim=-1).clamp_min(1e-6)
        rel = s / rn
        mask = torch.ones_like(s, dtype=torch.bool)
        mask[idx] = False                       # True = skipped (reused)
        q = torch.quantile(s, torch.tensor([0.5, 0.9], device=s.device))
        qr = torch.quantile(rel, torch.tensor([0.5, 0.9], device=s.device))
        e_all = (s * s).sum()
        e_skip = (s[mask] * s[mask]).sum()
        self.rows[name].append(dict(
            k=int(idx.numel()), thr=s[idx].min().item(), s_median=q[0].item(), s_p90=q[1].item(),
            s_max=s.max().item(), rel_median=qr[0].item(), rel_p90=qr[1].item(),
            frac_rel_lt_1e_2=(rel < 1e-2).float().mean().item(),
            stale_rel_mean=rel[mask].mean().item() if mask.any() else 0.0,
            stale_rel_max=rel[mask].max().item() if mask.any() else 0.0,
            stale_energy=(e_skip / e_all.clamp_min(1e-30)).item()))

    def summary(self):
        out = {}
        for name, rows in self.rows.items():
            out[name] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
            out[name]["frames"] = len(rows)
            out[name]["flushes"] = self.flushes[name]
        return out
