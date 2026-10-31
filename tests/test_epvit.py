"""EP-ViT correctness tests. Run directly: `python epvit/tests/test_epvit.py` (CPU tests only), or with
`EPVIT_GPU=<index>` to add the GPU tests (they load only the ViT weights from the SAM 3.1 checkpoint).
Writes results/epvit_evaluation/tests/test_epvit.json.

  1 codec      synthetic clip (static texture + a square moving one token per frame): encode -> extract recovers
               the motion, the background is static with ~0 residual
  2 schedule   I-frame flush, Eq. 11 forces a recompute once the IoU has decayed, budget mode gives exactly k
  3 equality   EPViT == SAM 3.1's ViT (bitwise) on a flush and on a steady frame with every token recomputed
  4 reuse      all-static frame returns the previous features exactly; a one-token warp moves the buffers
  5 no-sync    steady frames run without a device-to-host synchronisation
"""
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GPU = os.environ.get("EPVIT_GPU")
if GPU is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import numpy as np

import paths
from epvit import GRID
from epvit.codec.encode import encode
from epvit.codec.extract import extract
from epvit.codec.schedule import N_TOK, ScheduleConfig, Scheduler

OUT = paths.RUNS / "tests/test_epvit.json"


# ---------------------------------------------------------------- CPU
def synthetic_clip(d, T=8, size=504, sq=70, step=7, seed=0):
    from PIL import Image
    rng = np.random.default_rng(seed)
    bg = (rng.random((size // 7, size // 7, 3)) * 255).astype(np.uint8).repeat(7, 0).repeat(7, 1)
    tex = (rng.random((sq // 7, sq // 7, 3)) * 255).astype(np.uint8).repeat(7, 0).repeat(7, 1)
    files = []
    for t in range(T):
        im = bg.copy()
        x0 = 140 + t * step
        im[200:200 + sq, x0:x0 + sq] = tex
        f = Path(d) / f"{t:05d}.jpg"
        Image.fromarray(im).save(f, quality=98)
        files.append(f)
    return files, size, sq, step


def test_1_codec():
    with tempfile.TemporaryDirectory() as d:
        files, size, sq, step = synthetic_clip(d)
        log = encode(files, Path(d) / "clip.mp4", crf=18, gop=48)
        pr, timing = extract(Path(d) / "clip.mp4", (size, size))
    assert log["frame_types"] == "I" + "P" * (len(files) - 1), log["frame_types"]
    tok = size // GRID                                         # 7 px per token
    t = 4
    x0 = 140 + t * step
    rows = slice(200 // tok + 1, (200 + sq) // tok - 1)        # interior of the square
    cols = slice(x0 // tok + 1, (x0 + sq) // tok - 1)
    mv_sq = pr["mv"][t, rows, cols].reshape(-1, 2)
    expect = np.array([-step / tok, 0.0])                      # content came from one token to the left
    frac_sq = float(np.mean(np.all(np.abs(mv_sq - expect) < 0.25, axis=1)))
    bg = np.ones((GRID, GRID), bool)
    bg[200 // tok - 3:(200 + sq) // tok + 3, 140 // tok - 3:(140 + 8 * step + sq) // tok + 3] = False
    mv_bg = np.abs(pr["mv"][t][bg]).max(axis=-1)
    frac_bg_static = float(np.mean(mv_bg == 0))
    res_bg = float(np.median(pr["res"][t][bg].astype(float)))
    assert frac_sq > 0.8, frac_sq
    assert frac_bg_static > 0.9, frac_bg_static
    assert res_bg < 2.0, res_bg
    return {"square_tokens_with_expected_mv": frac_sq, "background_static": frac_bg_static,
            "background_median_residual": res_bg, "decode_mv_ms_per_frame": timing["decode_mv_ms_per_frame"]}


def _priors(T, mv=(0.0, 0.0), res=0.0, intra=0.0, ftype=None):
    return {"mv": np.broadcast_to(np.array(mv, np.float32), (T, GRID, GRID, 2)).copy(),
            "res": np.full((T, GRID, GRID), res, np.float16), "intra": np.full((T, GRID, GRID), intra, np.float16),
            "ftype": np.array(list(ftype or "I" + "P" * (T - 1)))}


def test_2_schedule():
    # I-frames and non-consecutive frames flush
    s = Scheduler(_priors(6, ftype="IPPIPP"), ScheduleConfig(mode="budget", k=0))
    flush = [s.step(t).flush for t in range(6)]
    assert flush == [True, False, False, True, False, False], flush
    assert s.step(4).flush                                     # repeated frame -> flush
    # static everywhere: nothing recomputed with k=0
    s = Scheduler(_priors(3), ScheduleConfig(mode="budget", k=0))
    s.step(0)
    p = s.step(1)
    assert len(p.idx) == 0 and p.stats["static"] == N_TOK
    # Eq. 11: a quarter-token motion keeps IoU 0.6 (score 0.4 < 0.5) on the first reuse, 0.36 (score 0.64) next
    s = Scheduler(_priors(4, mv=(0.25, 0.0)), ScheduleConfig(mode="threshold", theta=0.5, k_min=0, align=1))
    s.step(0)
    p1, p2 = s.step(1), s.step(2)
    assert p1.stats["forced"] == 0 and len(p1.warp_dst) > 0, p1.stats
    assert p2.stats["forced"] > 0.9 * N_TOK, p2.stats
    # budget mode: exactly k, unmatchable (large residual) first
    pr = _priors(2, res=0.0)
    pr["res"][1, :10] = 50                                     # 720 tokens with a large residual
    s = Scheduler(pr, ScheduleConfig(mode="budget", k=1024))
    s.step(0)
    p = s.step(1)
    hot = set(np.arange(10 * GRID))
    assert len(p.idx) == 1024 and hot <= set(p.idx.tolist())
    return {"flush_pattern": flush, "eq11_forced_frame2": p2.stats["forced"], "budget_M": len(p.idx)}


# ---------------------------------------------------------------- GPU
_VIT = {}


def _load_vit():
    if "vit" in _VIT:
        return _VIT["vit"]
    import torch
    from sam3.model_builder import _create_vit_backbone
    vit = _create_vit_backbone(use_fa3=False, use_rope_real=True)
    sd = torch.load(paths.CKPT, map_location="cpu", mmap=True, weights_only=True)
    sd = sd.get("model", sd)
    pre = "detector.backbone.vision_backbone.trunk."
    missing, unexpected = vit.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)},
                                              strict=False)
    assert all("freqs_cis" in k for k in missing + unexpected), (missing, unexpected)
    vit = vit.cuda().eval()
    _VIT["vit"] = vit
    return vit


def _frames(n=3, video="sav_017083"):
    import torch
    from PIL import Image
    from data.saco_data import MEDIA, load_gt
    v = next(v for v in load_gt()["videos"] if v["video_name"] == video)
    xs = []
    for fn in v["file_names"][:n]:
        im = Image.open(MEDIA / fn).convert("RGB").resize((1008, 1008), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
        xs.append((x - 0.5) / 0.5)
    return torch.stack(xs).cuda(), v


def _ctx():
    import torch
    return torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


def _need_gpu():
    if GPU is None:
        raise SkipTest("set EPVIT_GPU=<index>")


class SkipTest(Exception):
    pass


def test_3_equality():
    _need_gpu()
    import torch
    from epvit.model.epvit import EPViT
    vit = _load_vit()
    x, v = _frames(3)
    ep = EPViT.from_vit(vit, ScheduleConfig(mode="budget", k=N_TOK))
    ep.set_video(v["video_name"], _priors(3, mv=(0.25, 0.0), res=1.0))
    out = {}
    a, b = _ctx()
    with a, b:
        ref = [vit(x[t:t + 1])[-1] for t in range(3)]
        for t in range(3):
            ep.set_frame_ids([t])
            got = ep(x[t:t + 1])[-1]
            out[f"frame{t}_{ep.log[-1]['mode']}_bitwise"] = bool(torch.equal(got, ref[t]))
            out[f"frame{t}_max_abs"] = float((got.float() - ref[t].float()).abs().max())
            assert got.stride() == ref[t].stride() and got.dtype == ref[t].dtype
        # no priors: plain dense trunk
        ep.set_video(v["video_name"], None)
        out["no_prior_bitwise"] = bool(torch.equal(ep(x[:1])[-1], ref[0]))
    assert all(val for k, val in out.items() if k.endswith("bitwise")), out
    return out


def test_4_reuse_and_warp():
    _need_gpu()
    import torch
    from epvit.model.epvit import EPViT
    vit = _load_vit()
    x, v = _frames(3)
    ep = EPViT.from_vit(vit, ScheduleConfig(mode="budget", k=0))
    a, b = _ctx()
    with a, b:
        ep.set_video(v["video_name"], _priors(3))                       # all static
        ep.set_frame_ids([0]); f0 = ep(x[:1])[-1]
        ep.set_frame_ids([1]); f1 = ep(x[1:2])[-1]
        static_ok = bool(torch.equal(f0, f1))
        ep.set_video(v["video_name"], _priors(3, mv=(-1.0, 0.0)))      # all content moved one token right
        ep.set_frame_ids([0]); g0 = ep(x[:1])[-1]
        k0 = ep.K.clone()
        ep.set_frame_ids([1]); g1 = ep(x[1:2])[-1]
        warp_out = bool(torch.equal(g1[..., 1:], g0[..., :-1]) and torch.equal(g1[..., 0], g0[..., 0]))
        kk = ep.K.view(ep.K.shape[0], GRID, GRID, -1)
        k0 = k0.view_as(kk)
        warp_kv = bool(torch.equal(kk[:, :, 1:], k0[:, :, :-1]))
    assert static_ok and warp_out and warp_kv, (static_ok, warp_out, warp_kv)
    return {"static_returns_previous": static_ok, "warp_moves_output": warp_out, "warp_moves_kv": warp_kv}


def test_5_no_sync():
    _need_gpu()
    import torch
    from epvit.model.epvit import EPViT
    vit = _load_vit()
    x, v = _frames(3)
    pr = _priors(3, res=0.0)
    pr["res"][1:, :20] = 50
    ep = EPViT.from_vit(vit, ScheduleConfig(mode="budget", k=2560))
    ep.set_video(v["video_name"], pr)
    a, b = _ctx()
    with a, b:
        ep.set_frame_ids([0]); ep(x[:1])
        torch.cuda.synchronize()
        ep.set_frame_ids([1])
        torch.cuda.set_sync_debug_mode("error")
        try:
            ep(x[1:2])
        finally:
            torch.cuda.set_sync_debug_mode(0)
    return {"steady_frame_M": ep.log[-1]["M"]}


# ---------------------------------------------------------------- runner
def main():
    tests = [(k, f) for k, f in sorted(globals().items()) if k.startswith("test_") and callable(f)]
    results, failed = [], 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            det, status = fn(), "passed"
        except SkipTest as e:
            det, status = {"reason": str(e)}, "skipped"
        except Exception as e:                                   # noqa: BLE001
            det, status = {"error": repr(e), "traceback": traceback.format_exc()}, "failed"
            failed += 1
        results.append({"test": name, "status": status, "seconds": round(time.perf_counter() - t0, 2),
                        "details": det})
        print(f"{status:8s} {name} {json.dumps(det, default=str)[:300]}", flush=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True).stdout.strip()
    prov = {"commit": commit, "host": platform.node(), "gpu_env": GPU, "argv": sys.argv}
    if GPU is not None:
        import torch
        prov["gpu"] = torch.cuda.get_device_name(0)
        prov["torch"] = torch.__version__
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(OUT.read_text()) if OUT.exists() else {"results": []}
    merged = {r["test"]: r for r in old["results"]}
    merged.update({r["test"]: r for r in results if r["status"] != "skipped" or r["test"] not in merged})
    OUT.write_text(json.dumps({"module": "test_epvit", "results": list(merged.values()), "provenance": prov},
                              indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
