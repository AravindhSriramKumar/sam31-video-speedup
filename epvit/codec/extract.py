"""H.264 bitstream -> EP-ViT token priors on SAM 3.1's 72x72 grid (paper Sec. 4.4, MB-Token matching).

Per decoded frame t:
  * motion vectors from the decoder (PyAV, flags2=+export_mvs). Each exported block (w x h, centred at dst) points
    to the previous frame (the encode uses -refs 1, -bf 0). Pixels no block covers are intra-coded.
  * residual: H.264's own residual is not exposed by the decoder, so it is reconstructed as
    |Y_t - MC(Y_{t-1}, MV)| on decoded luma, i.e. what is left after motion compensation (paper Eq. 3).
  * mapping to tokens: SAM 3.1 stretches every W x H frame to 1008 x 1008, so token (i, j) covers source rows
    [i H/72, (i+1) H/72) and columns [j W/72, (j+1) W/72). Its MV is the majority vote of the MVs inside (Eq. 12,
    sampled on the 4 x 4 grid of the smallest H.264 partition), converted to token units (x * 72/W, y * 72/H).

Sign convention: mv[t, i, j] = (dx, dy) in tokens points from the token's position in frame t to where its content
was in frame t-1, i.e. the reuse source of token (i, j) is (i + dy, j + dx).

Output arrays (T = frames):
  mv     (T, 72, 72, 2) float32  majority MV, token units (0 on I-frames)
  agree  (T, 72, 72)    float16  share of the token's samples that voted for the majority MV
  res    (T, 72, 72)    float16  mean |residual| (luma, 0-255)
  intra  (T, 72, 72)    float16  share of the token's pixels that are intra-coded (1 on I-frames)
  ftype  (T,)           <U1      'I' / 'P'
"""
import time

import av
import numpy as np

from epvit import GRID


def token_ids(h, w, grid=GRID):
    """Per-pixel token index (row-major) for a W x H frame stretched to the grid."""
    ti = (np.arange(h) * grid // h).astype(np.int64)
    tj = (np.arange(w) * grid // w).astype(np.int64)
    return ti[:, None] * grid + tj[None, :]


def mv_field(mvs, h, w):
    """Rasterise exported MV blocks into a per-pixel displacement field (current -> reference, pixels)."""
    field = np.zeros((h, w, 2), np.float32)
    covered = np.zeros((h, w), bool)
    if mvs is None:
        return field, covered
    a = mvs.to_ndarray()
    a = a[a["source"] < 0]                       # past reference only (there are no B-frames anyway)
    scale = np.maximum(a["motion_scale"], 1).astype(np.float32)
    dx = a["motion_x"] / scale
    dy = a["motion_y"] / scale
    x0 = a["dst_x"] - a["w"] // 2
    y0 = a["dst_y"] - a["h"] // 2
    for k in range(len(a)):
        ys, xs = max(y0[k], 0), max(x0[k], 0)
        ye, xe = min(y0[k] + a["h"][k], h), min(x0[k] + a["w"][k], w)
        if ye > ys and xe > xs:
            field[ys:ye, xs:xe, 0] = dx[k]
            field[ys:ye, xs:xe, 1] = dy[k]
            covered[ys:ye, xs:xe] = True
    return field, covered


def motion_compensate(prev, field):
    """Nearest-pel motion compensation of the previous luma frame."""
    h, w = prev.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sx = np.clip(np.rint(xx + field[..., 0]).astype(np.int64), 0, w - 1)
    sy = np.clip(np.rint(yy + field[..., 1]).astype(np.int64), 0, h - 1)
    return prev[sy, sx]


def token_majority_mv(field, tid, h, w, step=4, grid=GRID):
    """Eq. 12: per-token majority vote of the MVs, sampled every `step` pixels. Returns (mv_px (N,2), agree (N,))."""
    s = field[step // 2::step, step // 2::step].reshape(-1, 2)
    t = tid[step // 2::step, step // 2::step].reshape(-1)
    q = np.rint(s * 4).astype(np.int64)                       # quarter-pel keys
    key = (t << 32) | (((q[:, 0] + 32768) & 0xFFFF) << 16) | ((q[:, 1] + 32768) & 0xFFFF)
    uk, cnt = np.unique(key, return_counts=True)
    ut = uk >> 32
    order = np.lexsort((-cnt, ut))                            # per token, most votes first
    first = order[np.r_[True, ut[order][1:] != ut[order][:-1]]]
    n = grid * grid
    mv = np.zeros((n, 2), np.float32)
    agree = np.zeros(n, np.float32)
    tok = ut[first]
    mv[tok, 0] = (((uk[first] >> 16) & 0xFFFF) - 32768) / 4
    mv[tok, 1] = ((uk[first] & 0xFFFF) - 32768) / 4
    agree[tok] = cnt[first] / np.bincount(t, minlength=n)[tok]
    return mv, agree


def extract(path, size, grid=GRID):
    """path: encoded .mp4; size: (W, H) of the source frames. Returns (arrays dict, timing dict)."""
    w, h = size
    tid = token_ids(h, w, grid)
    npix = np.bincount(tid.reshape(-1), minlength=grid * grid).astype(np.float32)
    out = {k: [] for k in ("mv", "agree", "res", "intra", "ftype")}
    t_dec = t_map = 0.0
    prev = None
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        st.codec_context.options = {"flags2": "+export_mvs"}
        it = c.decode(st)
        while True:
            t0 = time.perf_counter()
            try:
                frame = next(it)
            except StopIteration:
                break
            y = frame.to_ndarray(format="gray")
            if y.shape[0] - h not in (0, 1) or y.shape[1] - w not in (0, 1):   # encode pads odd sizes by 1
                raise ValueError(f"{path}: decoded {y.shape[1]}x{y.shape[0]}, expected {w}x{h}")
            y = y[:h, :w].astype(np.int16)
            mvs = frame.side_data.get(av.sidedata.sidedata.Type.MOTION_VECTORS)
            ft = frame.pict_type.name if hasattr(frame.pict_type, "name") else str(frame.pict_type)
            t1 = time.perf_counter()
            t_dec += t1 - t0
            if ft == "I" or prev is None:
                out["mv"].append(np.zeros((grid * grid, 2), np.float32))
                out["agree"].append(np.ones(grid * grid, np.float32))
                out["res"].append(np.zeros(grid * grid, np.float32))
                out["intra"].append(np.ones(grid * grid, np.float32))
                out["ftype"].append("I")
            else:
                field, covered = mv_field(mvs, h, w)
                pred = motion_compensate(prev, field)
                r = np.abs(y - pred).astype(np.float32)
                mv_px, agree = token_majority_mv(field, tid, h, w, grid=grid)
                mv_tok = mv_px * np.array([grid / w, grid / h], np.float32)
                out["mv"].append(mv_tok)
                out["agree"].append(agree)
                out["res"].append(np.bincount(tid.reshape(-1), weights=r.reshape(-1), minlength=grid * grid) / npix)
                out["intra"].append(np.bincount(tid.reshape(-1), weights=(~covered).reshape(-1).astype(np.float32),
                                                minlength=grid * grid) / npix)
                out["ftype"].append("P")
            prev = y
            t_map += time.perf_counter() - t1
    T = len(out["ftype"])
    arrays = {"mv": np.stack(out["mv"]).reshape(T, grid, grid, 2).astype(np.float32),
              "agree": np.stack(out["agree"]).reshape(T, grid, grid).astype(np.float16),
              "res": np.stack(out["res"]).reshape(T, grid, grid).astype(np.float16),
              "intra": np.stack(out["intra"]).reshape(T, grid, grid).astype(np.float16),
              "ftype": np.array(out["ftype"])}
    timing = {"frames": T, "decode_mv_ms_per_frame": 1e3 * t_dec / T, "map_ms_per_frame": 1e3 * t_map / T}
    return arrays, timing
