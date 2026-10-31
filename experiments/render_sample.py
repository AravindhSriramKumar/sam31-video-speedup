"""Draw the masks of the four sample runs side by side. Run the four --sample commands first.

    python experiments/render_sample.py

Reads results/runs/sample/*.json and writes results/sample_demo/: comparison.mp4 (2 x 2 grid), contact_sheet.png (one
row per motion level) and sample_results.csv. The sample has no ground truth: this shows the methods run and how their
masks compare, not which is more accurate.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402

import paths  # noqa: E402

METHODS = {"dense_B2d": "Dense", "epvit_D_ep75_c": "EP-ViT", "eventful": "Eventful", "eventful_fused": "Eventful + fused"}
MOMENTS = {96: "4 s, high motion", 192: "8 s, moderate motion", 348: "14.5 s, low motion"}
PANEL = (640, 362)
OUT = paths.RESULTS / "sample_demo"


def colour(obj_id):
    """The same object id gets the same colour in every panel and frame."""
    hue = int(obj_id * 47 % 180)
    return cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


def panel(image, preds, frame, title):
    out = image.astype(np.float32)
    for p in preds:
        rle = p["segmentations"][frame]
        if rle is None:
            continue
        m = mask_utils.decode(rle).astype(bool)
        out[m] = 0.45 * out[m] + 0.55 * colour(p["obj_id"])
    out = cv2.resize(out.astype(np.uint8), PANEL, interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (PANEL[0], 28), (0, 0, 0), -1)
    cv2.putText(out, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    runs = {}
    for name in METHODS:
        f = paths.RUNS / "sample" / f"{name}.json"
        if not f.exists():
            raise SystemExit(f"missing {f}: run the --sample commands first")
        runs[name] = json.loads(f.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    dense_trunk = runs["dense_B2d"]["trunk_ms_per_frame"]
    with open(OUT / "sample_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "prompt", "frames", "objects", "trunk_ms_per_frame", "trunk_speedup", "step_ms_median"])
        for name, label in METHODS.items():
            r = runs[name]
            w.writerow([label, r["prompt"], r["frames_yielded"], len(r["predictions"]), round(r["trunk_ms_per_frame"], 1),
                        round(dense_trunk / r["trunk_ms_per_frame"], 2), round(r["step_ms_median"], 1)])

    folder = paths.SAMPLE_FRAMES
    n = runs["dense_B2d"]["frames_yielded"]
    sheet = []
    for frame, moment in MOMENTS.items():
        image = cv2.imread(str(folder / f"{frame}.jpg"))
        sheet.append(np.hstack([panel(image, runs[m]["predictions"], frame, f"{label}: {moment}") for m, label in METHODS.items()]))
    cv2.imwrite(str(OUT / "contact_sheet.png"), np.vstack(sheet))

    h, w = PANEL[1] * 2, PANEL[0] * 2
    ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                           "-r", "24", "-i", "-", "-c:v", "libx264", "-crf", "30", "-preset", "slow", "-pix_fmt", "yuv420p",
                           str(OUT / "comparison.mp4")], stdin=subprocess.PIPE)
    for frame in range(n):
        image = cv2.imread(str(folder / f"{frame}.jpg"))
        p = [panel(image, runs[m]["predictions"], frame, label) for m, label in METHODS.items()]
        ff.stdin.write(np.vstack([np.hstack(p[:2]), np.hstack(p[2:])]).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
