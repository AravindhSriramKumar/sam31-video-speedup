"""The included sample video (data/sample_video.mp4): decoded frames and a short session report.

A smoke test of a runner on one clip, with no ground truth and no accuracy score. Full evaluation is --subset.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

import paths  # noqa: E402
from experiments.harness import env  # noqa: E402
from experiments.harness.run_pipeline_eval import predictions_from, run_session  # noqa: E402

DEFAULT_PROMPT = "a paw"


def probe():
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,nb_read_frames", "-of", "json", str(paths.SAMPLE_VIDEO)],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"], int(s["nb_read_frames"])


def frame_dir():
    """Frames of the sample as 0.jpg .. N-1.jpg, the layout the predictor loads."""
    w, h, n = probe()
    out = paths.SAMPLE_FRAMES
    if not (out.is_dir() and len(list(out.glob("*.jpg"))) == n):
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(paths.SAMPLE_VIDEO),
                        "-vsync", "0", "-q:v", "2", "-start_number", "0", str(out / "%d.jpg")], check=True)
    return out, dict(id=-1, video_name="sample_video", width=w, height=h, length=n)


def warmup_dir(n=16):
    out = paths.SAMPLE_FRAMES.parent / f"sample_frames__first{n}"
    out.mkdir(exist_ok=True)
    src, _ = frame_dir()
    for i in range(n):
        dst = out / f"{i}.jpg"
        if not dst.exists():
            dst.symlink_to((src / f"{i}.jpg").resolve())
    return out


def run(pred, timer, method, prompt, out=None):
    """One session on the sample. Writes results/runs/sample/<method>.json with masks and timings."""
    folder, video = frame_dir()
    outs, prompt_ms, prop_ms, n_yield, status = run_session(pred, timer, str(folder), prompt)
    t = timer.take()
    steps = [s["ms"] for s in t["steps"] if s["phase"] == "propagate"]
    trunk = [x for x in t["trunk"] if x["phase"] == "propagate"]
    rec = dict(method=method, prompt=prompt, status=status, frames=video["length"], frames_yielded=n_yield,
               step_ms_median=float(np.median(steps)) if steps else None,
               trunk_ms_per_frame=sum(x["ms"] for x in trunk) / max(sum(x["batch"] for x in trunk), 1),
               objects_per_frame=[len(outs[k]["out_obj_ids"]) for k in sorted(outs)],
               predictions=predictions_from(outs, video, dict(category_id=-1)))
    out = Path(out) if out else paths.RUNS / "sample" / f"{method}.json"
    env.write_json(out, rec)
    print(f"sample {method} '{prompt}': {status}, {n_yield} frames, step median {rec['step_ms_median']:.1f} ms, "
          f"trunk {rec['trunk_ms_per_frame']:.1f} ms/frame, {len(rec['predictions'])} objects -> {out}", flush=True)
    return rec
