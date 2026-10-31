"""Per-frame token plan for EP-ViT: which tokens to recompute and which to reuse (paper Sec. 4.1 and 4.3).

The scheduler runs on the host, one step per frame the trunk actually encodes, so its reuse-chain state always
matches the encoder's buffers (a reset or a skipped frame flushes both). No GPU work and no GPU sync.

Classification (Eq. 6), per token with MV v (token units), residual r, intra share q:
    static       |v| = 0 and r < tau1
    matchable    |v| > 0 and r < tau2 and the source token (i + dy, j + dx) lies inside the grid
    unmatchable  otherwise, or q > intra_max
Without warping, matchable tokens are recomputed (ablation: the tiled absolute position embedding of SAM 3.1 makes a
moved token's cached features carry its source position).

Degradation-Aware Update Link (Eq. 11), along each token's reuse chain:
    iou[j]   <- iou[src] * IoU(unit cell shifted by the fractional part of v, its grid cell)
    score[j]  = lam * |v - v_prev[src]|^2 + (1 - iou[j])
    score > theta  ->  forced recompute; recomputing resets iou = 1.

Recompute count:
    budget k     exactly k tokens: all unmatchable first, ranked by priority, filled with the most degraded reused ones
    threshold    the unmatchable tokens, rounded up to a multiple of `align` (at least k_min) with the same fill
A reused token that is not recomputed keeps its own cached row (static) or takes its source's row (warp).
"""
from dataclasses import asdict, dataclass

import numpy as np

from epvit import GRID

WIN = 24                     # SAM 3.1 window size in tokens
N_TOK = GRID * GRID


@dataclass(frozen=True)
class ScheduleConfig:
    mode: str = "budget"           # "budget" | "threshold" | "dense"
    k: int = N_TOK                 # budget mode: tokens recomputed per frame
    tau1: float = 4.0              # static residual threshold (mean |r|, luma 0-255)
    tau2: float = 6.0              # matchable residual threshold
    lam: float = 1.0               # Eq. 11 weight of the MV change (token units^2)
    theta: float = 0.5             # Eq. 11 threshold
    intra_max: float = 0.5         # tokens with more intra-coded pixels than this are unmatchable
    warp: bool = True              # reuse matchable tokens from their MV source; False -> recompute them
    align: int = 128               # threshold mode: round the recompute count up to a multiple of this
    k_min: int = 128               # threshold mode: lower bound on the recompute count

    def as_dict(self):
        return asdict(self)


@dataclass
class FramePlan:
    flush: bool                    # dense forward; buffers rebuilt from scratch
    idx: np.ndarray                # (M,) int64 sorted token ids to recompute (raster order)
    warp_src: np.ndarray           # (W,) int64 rows copied from ...
    warp_dst: np.ndarray           # (W,) int64 ... to these rows, before the recompute
    max_m_win: int                 # largest number of recomputed tokens in one 24 x 24 window
    stats: dict


_WIN_ID = ((np.arange(GRID)[:, None] // WIN) * (GRID // WIN) + np.arange(GRID)[None, :] // WIN).reshape(-1)


def _cell_iou(fx, fy):
    inter = (1 - np.abs(fx)) * (1 - np.abs(fy))
    return inter / (2 - inter)


class Scheduler:
    def __init__(self, priors, cfg: ScheduleConfig):
        self.cfg = cfg
        self.mv = priors["mv"].reshape(-1, N_TOK, 2).astype(np.float32)
        self.res = priors["res"].reshape(-1, N_TOK).astype(np.float32)
        self.intra = priors["intra"].reshape(-1, N_TOK).astype(np.float32)
        self.ftype = priors["ftype"]
        self.T = len(self.ftype)
        self.reset()

    def reset(self):
        self.last = None
        self.iou = np.ones(N_TOK, np.float32)
        self.vprev = np.zeros((N_TOK, 2), np.float32)

    def _flush(self, t):
        self.last = t
        self.iou[:] = 1
        self.vprev[:] = 0
        idx = np.arange(N_TOK)
        e = np.empty(0, np.int64)
        return FramePlan(True, idx, e, e, int(np.bincount(_WIN_ID).max()),
                         {"t": t, "M": N_TOK, "static": 0, "matchable": 0, "unmatchable": N_TOK, "forced": 0,
                          "warped": 0})

    def step(self, t):
        cfg = self.cfg
        if (cfg.mode == "dense" or self.last is None or t != self.last + 1 or t >= self.T
                or self.ftype[t] == "I"):
            return self._flush(t)
        self.last = t
        v, r, q = self.mv[t], self.res[t], self.intra[t]
        moving = np.any(v != 0, axis=1)
        ok = q <= cfg.intra_max

        # reuse source of every token (itself when static)
        rows, cols = np.divmod(np.arange(N_TOK), GRID)
        si = np.rint(rows + v[:, 1]).astype(np.int64)
        sj = np.rint(cols + v[:, 0]).astype(np.int64)
        inside = (si >= 0) & (si < GRID) & (sj >= 0) & (sj < GRID)
        src = np.where(moving & inside, si * GRID + sj, np.arange(N_TOK))

        static = ~moving & (r < cfg.tau1) & ok
        matchable = moving & (r < cfg.tau2) & ok & inside
        if not cfg.warp:
            matchable[:] = False

        # Eq. 11 along the reuse chain
        frac = v - np.rint(v)
        iou_new = self.iou[src] * np.where(moving, _cell_iou(frac[:, 0], frac[:, 1]), 1.0)
        score = cfg.lam * np.sum((v - self.vprev[src]) ** 2, axis=1) + (1 - iou_new)
        reusable = (static | matchable)
        forced = reusable & (score > cfg.theta)
        unmatch = ~(reusable & ~forced)

        # priority: unmatchable first (largest residual first), then the most degraded reused tokens
        prio = np.where(unmatch, 1e6 + r, score)
        n_un = int(unmatch.sum())
        if cfg.mode == "budget":
            m = cfg.k
        else:
            m = min(N_TOK, max(cfg.k_min, -(-n_un // cfg.align) * cfg.align))
        if m >= N_TOK:
            idx = np.arange(N_TOK)
        elif m <= 0:
            idx = np.empty(0, np.int64)
        else:
            idx = np.sort(np.argpartition(-prio, m - 1)[:m])
        recompute = np.zeros(N_TOK, bool)
        recompute[idx] = True

        # tokens kept from the cache: warp the moving ones (or keep in place if warping is off / impossible)
        if cfg.warp:
            wmask = ~recompute & moving & inside
        else:
            wmask = np.zeros(N_TOK, bool)
        warp_dst = np.nonzero(wmask)[0]
        warp_src = src[warp_dst]

        # state update: reused tokens inherit their chain, recomputed tokens restart it
        chain_src = np.where(wmask, src, np.arange(N_TOK))
        new_iou = np.where(recompute, 1.0, np.where(wmask, iou_new, self.iou[chain_src])).astype(np.float32)
        self.iou = new_iou
        self.vprev = v.copy()

        return FramePlan(False, idx.astype(np.int64), warp_src.astype(np.int64), warp_dst.astype(np.int64),
                         int(np.bincount(_WIN_ID[idx], minlength=9).max()),
                         {"t": t, "M": int(len(idx)), "static": int(static.sum()),
                          "matchable": int((moving & (r < cfg.tau2) & ok & inside).sum()),
                          "unmatchable": n_un, "forced": int(forced.sum()), "warped": int(len(warp_dst))})


def load_priors(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
