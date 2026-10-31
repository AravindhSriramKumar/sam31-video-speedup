"""Horizontal bar charts of the four methods. Run render_sample.py first.

    python experiments/plot_results.py

Writes results/figures/sample_video.png and results/figures/all_videos.png. The sample run has no ground truth, so
sample_video.png takes its accuracy from the scored pairs of the same video (sav_017083) in per_video_results.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402

SAMPLE = "sav_017083"
METHODS = {"B2d": ("Dense", "#2a78d6"), "D_ep75": ("EP-ViT", "#eb6834"), "eventful": ("Eventful", "#1baf7a"),
           "eventful_fused": ("Eventful + fused", "#eda100")}
LABELS = {v[0]: k for k, v in METHODS.items()}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
OUT = paths.RESULTS / "figures"


def panel(ax, values, title, subtitle, fmt):
    names = [v[0] for v in METHODS.values()][::-1]
    vals = [values[LABELS[n]] for n in names]
    bars = ax.barh(names, vals, height=0.46, color=[METHODS[LABELS[n]][1] for n in names])
    top = max(vals)
    for b, v in zip(bars, vals):
        ax.text(v + top * 0.015, b.get_y() + b.get_height() / 2, fmt.format(v), va="center", fontsize=10, color=INK)
    ax.set_xlim(0, top * 1.18)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=20)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)
    ax.set_facecolor(SURFACE)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="y", colors=INK, labelsize=10, length=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)


def figure(panels, title, name):
    fig, axes = plt.subplots(len(panels) // 2, 2, figsize=(11, 2.75 * (len(panels) // 2)), facecolor=SURFACE)
    for ax, p in zip(axes.flat, panels):
        panel(ax, *p)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5, w_pad=3)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, facecolor=SURFACE)
    print(f"-> {OUT / name}")


def main():
    table = pd.read_csv(paths.RESULTS / "sample_demo/sample_results.csv").set_index("method")
    run = {k: table.loc[v[0]] for k, v in METHODS.items()}
    video = pd.read_csv(paths.RESULTS / "per_video_results.csv")
    video = video[video.video == SAMPLE].set_index("run")
    pairs = int(video.pairs.iloc[0])
    figure([(dict(zip(run, (r.trunk_speedup for r in run.values()))), "Encoder speedup", "Sample run, against Dense", "{:.2f}×"),
            (dict(zip(run, (r.step_speedup for r in run.values()))), "End-to-end speedup", "Sample run, median model step", "{:.2f}×"),
            (video.pHOTA.to_dict(), "pHOTA", f"This video, {pairs} scored pairs", "{:.2f}"),
            (video.cgF1.to_dict(), "cgF1", f"This video, {pairs} scored pairs (noisy on one video)", "{:.2f}"),
            (dict(zip(run, (r.iou_vs_dense for r in run.values()))), "Mask agreement with Dense",
             f"Sample run, mean IoU, prompt '{run['B2d'].prompt}'. Not accuracy", "{:.3f}"),
            (dict(zip(run, (r.trunk_ms_per_frame for r in run.values()))), "Encoder time", "Sample run, ms per frame, lower is faster", "{:.1f}")],
           f"Sample video {SAMPLE}: 577 frames, 24 fps, one GPU, run back to back", "sample_video.png")
    all20 = pd.read_csv(paths.RESULTS / "compiled_results.csv")
    all20 = all20[all20.scope == "all20"].set_index("run")
    figure([(all20.pHOTA.to_dict(), "pHOTA", "189 pairs", "{:.2f}"), (all20.cgF1.to_dict(), "cgF1", "189 pairs", "{:.2f}"),
            (all20.enc_speedup.to_dict(), "Encoder speedup", "Against Dense", "{:.2f}×"),
            (all20.enc_ms_per_frame.to_dict(), "Encoder time", "ms per frame, lower is faster", "{:.1f}")],
           "All evaluation videos: 20 videos at 24 fps, official evaluator", "all_videos.png")


if __name__ == "__main__":
    main()
