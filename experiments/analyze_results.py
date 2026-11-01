"""Score Dense B2d, EP-ViT D_ep75, Eventful, Eventful + fused and Eventful three gates on the 189 frozen pairs and compare,
or check the committed tables.

    python experiments/analyze_results.py --committed      # offline: internal checks of results/ (no data needed)
    python experiments/analyze_results.py                  # re-score full runs in results/runs/ (needs annotations)

Rules: accuracy on the pairs of data/eval_pairs.json, official evaluator, scopes all20, normal, high_motion and each
video; encoder ms/frame = trunk GPU ms / frames passed to the trunk; E2E = propagate-step median; executed trunk MACs per
frame passed to the trunk (EP-ViT log, dense 2.699 TMAC; not logged for Eventful, so blank). Speedups are against dense
B2d. Writes results/runs/analysis/.
"""
import argparse
import contextlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import paths  # noqa: E402
from data import saco_data  # noqa: E402

ENC = "crf23_g192_iffix"
DENSE_TMAC = 2.699287068672
SUBSETS = ("normal", "high_motion")
LABELS = {"B2d": "Dense SAM 3.1 (B2d, compiled)", "D_ep75": "EP-ViT 75%", "eventful": "Eventful", "eventful_fused": "Eventful + fused kernel",
          "eventful_three_gates": "Eventful three gates"}
EVENTFUL_TAG = "B1_mlp_k3840_lag0.6_bf16ref_torchscore_fb4608"
THREE_GATES_TAG = "B1_mlp+qkv+proj_k3840_lag0.6_bf16ref_torchscore_fb4608"
COMMITTED = paths.RESULTS / "compiled_results.csv"
COMMITTED_VIDEOS = paths.RESULTS / "per_video_results.csv"
COMPARED = ("pHOTA", "cgF1", "enc_ms_per_frame", "e2e_ms_median", "tmac_per_frame")


def load_pairs(d, ids):
    recs = {}
    for i in ids:
        f = d / "pairs" / f"{i}.json"
        if not f.exists():
            raise SystemExit(f"run not finished: {f} is missing")
        recs[i] = json.loads(f.read_text())
        assert recs[i]["status"] == "ok", f"pair {i} did not complete in {d}"
    return recs


def _score(job):
    from experiments.harness.run_pipeline_eval import finalize, key_metrics

    out_dir, recs = job
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        m = finalize(Path(out_dir), saco_data.load_gt(), recs, tag="subset")["dataset_results"]
    (Path(out_dir) / "predictions_subset.json").unlink(missing_ok=True)          # large; regenerable
    return key_metrics(m)


def _score_video(job):
    """One video alone. The evaluator raises when a method predicted nothing on it; that is recorded, not fatal."""
    try:
        return _score(job), ""
    except Exception as e:
        return dict(cgF1=float("nan"), pHOTA=float("nan")), f"{type(e).__name__}: {e}"


def cost(run, X):
    trunk = [t for x in X for t in x["timing"]["trunk"]]
    steps = [s["ms"] for x in X for s in x["timing"]["steps"] if s["phase"] == "propagate"]
    fin = sum(t["batch"] for t in trunk)
    c = dict(pairs=len(X), trunk_frames_in=fin, enc_ms_per_frame=sum(t["ms"] for t in trunk) / fin,
             e2e_ms_median=float(np.median(steps)), e2e_ms_mean=float(np.mean(steps)))
    if run == "B2d":
        c["tmac_per_frame"] = DENSE_TMAC
    elif run != "D_ep75":
        c["tmac_per_frame"] = float("nan")
    else:
        E = [x["epvit"] for x in X]
        assert sum(e["frames_in"] for e in E) == fin
        c["tmac_per_frame"] = sum(e["trunk_macs"] for e in E) / fin / 1e12
    return c


def derive(df):
    ref = df[df.run == "B2d"].set_index("scope")
    df = df.copy()
    df["dcgF1"] = df.cgF1 - df.scope.map(ref.cgF1)
    df["dpHOTA"] = df.pHOTA - df.scope.map(ref.pHOTA)
    df["mac_reduction_pct"] = 100 * (1 - df.tmac_per_frame / DENSE_TMAC)
    df["enc_speedup"] = df.scope.map(ref.enc_ms_per_frame) / df.enc_ms_per_frame
    df["e2e_speedup"] = df.scope.map(ref.e2e_ms_median) / df.e2e_ms_median
    df["e2e_mean_speedup"] = df.scope.map(ref.e2e_ms_mean) / df.e2e_ms_mean
    df["rest_of_step_ms"] = df.e2e_ms_median - df.enc_ms_per_frame
    return df


def check_committed():
    df = pd.read_csv(COMMITTED)
    frozen = saco_data.load_eval_pairs()
    want = {"normal": len(frozen["pairs"]["normal"]), "high_motion": len(frozen["pairs"]["high_motion"])}
    want["all20"] = sum(want.values())
    bad = []

    def check(cond, msg):
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            bad.append(msg)

    check(want["all20"] == 189, "189 frozen pairs")
    for sc, n in want.items():
        check(set(df[df.scope == sc].pairs) == {n} and len(df[df.scope == sc]) == len(LABELS),
              f"{sc}: all {len(LABELS)} methods scored on {n} pairs")
    d = derive(df.drop(columns=[c for c in df.columns if c in ("dcgF1", "dpHOTA", "mac_reduction_pct", "enc_speedup",
                                                               "e2e_speedup", "e2e_mean_speedup", "rest_of_step_ms")]))
    for c in ("dpHOTA", "enc_speedup", "e2e_speedup"):
        check(np.allclose(d[c], df[c]), f"{c} follows from the measured columns")
    for folder in ("dense", "epvit", "eventful", "eventful_fused", "eventful_three_gates"):
        for s in SUBSETS:
            S = json.loads((paths.RESULTS / folder / s / "summary.json").read_text())
            M = json.loads((paths.RESULTS / folder / s / "metrics.json").read_text())
            check(S["pairs_ok"] >= want[s] and S["fps"] == 24, f"{folder}/{s}: {S['pairs_ok']} pairs ok at 24 fps")
            check(M["dataset_results"]["video_mask_all_phrase_HOTA"] == S["key_metrics"]["pHOTA"],
                  f"{folder}/{s}: summary pHOTA equals the evaluator output")
    print(df[df.scope == "all20"][["method", "pairs", "pHOTA", "enc_ms_per_frame", "e2e_ms_median",
                                   "enc_speedup"]].round(3).to_string(index=False))
    if bad:
        raise SystemExit(f"{len(bad)} checks failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--committed", action="store_true", help="check results/ offline")
    ap.add_argument("--dense-root", default=paths.RUNS / "dense", type=Path)
    ap.add_argument("--epvit-root", default=paths.RUNS / "epvit", type=Path)
    ap.add_argument("--eventful-root", default=paths.RUNS / "eventful", type=Path)
    ap.add_argument("--fused-root", default=paths.RUNS / "eventful_fused", type=Path)
    ap.add_argument("--three-gates-root", default=paths.RUNS / "eventful_three_gates", type=Path)
    ap.add_argument("--out", default=paths.RUNS / "analysis", type=Path)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    if a.committed:
        return check_committed()

    frozen = saco_data.load_eval_pairs()["pairs"]
    recs = {}
    for s in SUBSETS:
        recs["B2d", s] = load_pairs(a.dense_root / f"{s}__B2d__gb1_pp1__adapter__24fps", frozen[s])
        recs["D_ep75", s] = load_pairs(a.epvit_root / f"{s}__D_ep75_c__{ENC}__24fps", frozen[s])
        recs["eventful", s] = load_pairs(a.eventful_root / f"{s}__{EVENTFUL_TAG}_tk_cg__gb1_pp1__adapter__24fps", frozen[s])
        recs["eventful_fused", s] = load_pairs(a.fused_root / f"{s}__{EVENTFUL_TAG}_tk_fgfull_cg__gb1_pp1__adapter__24fps", frozen[s])
        recs["eventful_three_gates", s] = load_pairs(a.three_gates_root / f"{s}__{THREE_GATES_TAG}_tk_cg__gb1_pp1__adapter__24fps", frozen[s])
    scopes = {"all20": SUBSETS, "normal": ("normal",), "high_motion": ("high_motion",)}
    keys, jobs = [], []
    for r in LABELS:
        for sc, subs in scopes.items():
            out = a.out / f"{sc}__{r}"
            out.mkdir(parents=True, exist_ok=True)
            keys.append((r, sc, subs))
            jobs.append((str(out), [x for s in subs for x in recs[r, s].values()]))
    videos = sorted({(s, x["video"]) for s in SUBSETS for x in recs["B2d", s].values()})
    vkeys, vjobs = [], []
    for r in LABELS:
        for s, v in videos:
            out = a.out / f"video_{v}__{r}"
            out.mkdir(parents=True, exist_ok=True)
            vkeys.append((r, s, v))
            vjobs.append((str(out), [x for x in recs[r, s].values() if x["video"] == v]))
    with ProcessPoolExecutor(a.workers) as ex:
        scored = list(ex.map(_score, jobs))
        vscored = list(ex.map(_score_video, vjobs))
    rows = [dict(method=LABELS[r], run=r, scope=sc, cgF1=100 * km["cgF1"], pHOTA=100 * km["pHOTA"],
                 **cost(r, [x for s in subs for x in recs[r, s].values()]))
            for (r, sc, subs), km in zip(keys, scored)]
    df = derive(pd.DataFrame(rows))
    df.to_csv(a.out / "compiled_results.csv", index=False)
    vd = pd.DataFrame([dict(method=LABELS[r], run=r, subset=s, video=v, cgF1=100 * km["cgF1"], pHOTA=100 * km["pHOTA"],
                            eval_error=err, **cost(r, X)) for (r, s, v), (km, err), (_, X) in zip(vkeys, vscored, vjobs)])
    vd = derive(vd.rename(columns=dict(video="scope"))).rename(columns=dict(scope="video"))
    vd.drop(columns=["e2e_mean_speedup", "rest_of_step_ms"]).to_csv(a.out / "per_video_results.csv", index=False)
    new = df.set_index(["run", "scope"])
    print(new[list(COMPARED)].round(3).to_string())
    for name, committed, table, key in (("compiled", COMMITTED, df, ["run", "scope"]), ("per-video", COMMITTED_VIDEOS, vd, ["run", "video"])):
        ref, cur = pd.read_csv(committed).set_index(key), table.set_index(key)
        both = cur.index.intersection(ref.index)
        print(f"\n{name}, new - committed, largest absolute difference over {len(both)} rows:")
        print((cur.loc[both, list(COMPARED)] - ref.loc[both, list(COMPARED)]).abs().max().round(4).to_string())


if __name__ == "__main__":
    main()
