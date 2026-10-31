"""Session, prediction and scoring functions of the SA-Co/VEval pipeline evaluation (24 fps protocol).

Protocol:
  * one session per (video, noun phrase) pair, on the native 24 fps frames from 0 to the last annotated frame;
  * add_prompt(text) on frame 0, then forward propagation; only the annotated frames (every 4th) are scored;
  * every object id becomes one prediction: score = mean output probability over the scored frames it appears in,
    segmentations = one compressed RLE mask per annotated frame (empty where the object is absent);
  * predictions are scored by SAM's own evaluator, sam3.eval.saco_veval_eval.VEvalEvaluator, against the
    ground truth filtered to exactly the evaluated pairs (data/saco_data.py::filtered_gt).

Timing (measurement only, instrument.py::StepTimer, CUDA events resolved once per session):
  step_ms     one per-frame model step = model._det_track_one_frame (detector incl. trunk when needed, tracker,
              planning, memory encoder, outputs). Excludes JPEG decoding (start_session, untimed), output
              post-processing and the hotstart buffer. This is the per-frame compute latency.
  trunk_ms    every trunk call.
  wall        host time of add_prompt and of the whole propagation (sync before and after), for throughput.
Memory: torch.cuda.max_memory_allocated() per pair (peak is reset before each pair).

Run output: pairs/<pair_id>.json (raw per-pair record, a run resumes from these), predictions.json, gt.json,
metrics.json (official evaluator output), summary.json, summary.md.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools import mask as mask_utils  # noqa: E402

from data import saco_data  # noqa: E402
from experiments.harness import env  # noqa: E402
from experiments.harness.baseline import PRESETS  # noqa: E402
from experiments.harness.timing import summarize  # noqa: E402


# Headline metrics: our name -> key in the official evaluator output (VEvalEvaluator.run_eval()["dataset_results"]).
# All are mask-based, in [0, 1] (SAM's papers report them x100). cgF1 = positive micro-F1 x IL_MCC (image-level
# Matthews correlation: does the model correctly say whether the phrase is present at all); pHOTA = phrase-level
# HOTA (detection and association quality of the tracks).
KEY_METRICS = {
    "cgF1": "video_mask_demo_cgf1_micro_50_95",
    "pmF1": "video_mask_demo_positive_micro_f1_50_95",
    "IL_MCC": "video_mask_demo_ilmcc_50_95",
    "pHOTA": "video_mask_all_phrase_HOTA",
    "pDetA": "video_mask_all_phrase_DetA",
    "pAssA": "video_mask_all_phrase_AssA",
    "mask_mAP": "video_mask_mAP_50_95",
    "TETA": "video_mask_teta",
}


def key_metrics(dataset_results):
    return {name: dataset_results.get(k) for name, k in KEY_METRICS.items()}


def rle(mask):
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


UPSTREAM_BUG = "No points are provided; please add points first"


def run_session(pred, timer, folder, text):
    """Returns (outputs per frame, prompt wall ms, propagation wall ms, frames yielded, status).

    status is "ok", "failed_cuda_oom" when the session exceeds GPU memory (possible on long 24 fps sessions), or
    "crashed_upstream_tracker_bug" when SAM 3.1 raises the known tracker error : hotstart removes the objects that own a tracker state's
    only conditioning frame, `clear_all_points_in_frame` then calls `_reset_tracking_results`, which wipes the
    remaining objects' memory, and the next propagation raises. SAM's behaviour is not altered; the pair is recorded
    as crashed and left out of the accuracy evaluation (and out of every compared configuration's common subset).
    """
    sid = pred.handle_request(dict(type="start_session", resource_path=folder, offload_video_to_cpu=True))["session_id"]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    timer.phase = "prompt"
    outs = {0: pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=text))["outputs"]}
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    timer.phase = "propagate"
    n_yield, status = 0, "ok"
    try:
        for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
                                                 propagation_direction="forward")):
            outs[r["frame_index"]] = r["outputs"]
            n_yield += 1
    except torch.OutOfMemoryError:
        # 24 fps sessions are 4x longer; SAM keeps per-frame tracking state on the GPU for the whole session, so peak
        # memory grows with (tracked objects x frames). Long, crowded pairs can exceed the 24 GiB card in any
        # configuration (dense included). The pair is recorded and excluded like an upstream crash.
        status = "failed_cuda_oom"
    except RuntimeError as e:
        if UPSTREAM_BUG not in str(e):
            raise
        status = "crashed_upstream_tracker_bug"
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    try:
        pred.handle_request(dict(type="close_session", session_id=sid))
    except Exception:                                    # a session broken by OOM may not close cleanly
        if status != "failed_cuda_oom":
            raise
    if status == "failed_cuda_oom":
        import gc
        outs.clear()
        gc.collect()
        torch.cuda.empty_cache()
    return outs, (t1 - t0) * 1e3, (t2 - t1) * 1e3, n_yield, status


def predictions_from(outs, video, pair, frame_map=None):
    """One prediction per object id: score = mean output probability over the scored frames it appears in,
    segmentations = one RLE per annotated frame. frame_map (24 fps runs): {model frame index: annotated index};
    outputs of frames not in it are not scored."""
    objects = {}
    for mfi, out in sorted(outs.items()):
        if frame_map is not None:
            if mfi not in frame_map:
                continue
            fi = frame_map[mfi]
        else:
            fi = mfi
        for j, oid in enumerate(out["out_obj_ids"]):
            row = objects.setdefault(int(oid), dict(masks=[None] * video["length"], probs=[]))
            row["masks"][fi] = rle(out["out_binary_masks"][j])
            row["probs"].append(float(out["out_probs"][j]))
    return [dict(video_id=video["id"], category_id=pair["category_id"], obj_id=oid,
                 score=float(np.mean(r["probs"])), segmentations=r["masks"]) for oid, r in objects.items()]


def finalize(out_dir, gt, records, tag=""):
    """Score the given records (callers pass completed pairs only). tag="" -> predictions.json/metrics.json."""
    videos = {v["id"]: v for v in gt["videos"]}
    preds = []
    for rec in records:
        v = videos[rec["video_id"]]
        empty = rle(np.zeros((v["height"], v["width"]), dtype=np.uint8))
        for p in rec["predictions"]:
            p = dict(p)
            p["segmentations"] = [m if m is not None else empty for m in p["segmentations"]]
            p["bboxes"] = [mask_utils.toBbox(m).tolist() for m in p["segmentations"]]
            p["areas"] = [float(mask_utils.area(m)) for m in p["segmentations"]]
            p.pop("obj_id", None)
            preds.append(p)
    sfx = f"_{tag}" if tag else ""
    env.write_json(out_dir / f"predictions{sfx}.json", preds)
    keys = [(r["video_id"], r["category_id"]) for r in records]
    saco_data.filtered_gt(gt, sorted({k[0] for k in keys}), keys, out_dir / f"gt{sfx}.json")
    from sam3.eval.saco_veval_eval import VEvalEvaluator

    return VEvalEvaluator(str(out_dir / f"gt{sfx}.json"), str(out_dir / f"metrics{sfx}.json")).run_eval(
        str(out_dir / f"predictions{sfx}.json"))


def summarize_run(records, metrics, args, prov):
    steps = [s["ms"] for r in records for s in r["timing"]["steps"] if s["phase"] == "propagate"]
    prompt_steps = [s["ms"] for r in records for s in r["timing"]["steps"] if s["phase"] == "prompt"]
    trunk = [t for r in records for t in r["timing"]["trunk"]]
    frames_enc = sum(t["batch"] for t in trunk)
    prop_frames = sum(r["frames_yielded"] for r in records)
    prop_wall = sum(r["propagate_wall_ms"] for r in records)
    ds = metrics["dataset_results"]
    key = key_metrics(ds)
    return dict(
        subset=args.subset, preset=args.preset, preset_config=PRESETS[args.preset].as_dict(),
        eventful=getattr(args, "eventful", None), eventful_config=getattr(args, "eventful_config", None),
        grounding_batch=args.grounding_batch, postprocess_batch=args.postprocess_batch, fps=getattr(args, "fps", 6),
        hotstart_delay=records[0]["hotstart_delay"] if records else None, adapter=args.adapter,
        pairs=len(records), pairs_ok=sum(r["status"] == "ok" for r in records),
        pairs_crashed=[dict(pair_id=r["pair_id"], video=r["video"], noun_phrase=r["noun_phrase"],
                            frames_yielded=r["frames_yielded"], frames=r["frames"])
                       for r in records if r["status"] != "ok"],
        videos=len({r["video_id"] for r in records}), propagate_frames=prop_frames,
        key_metrics=key, all_metrics=ds,
        latency=dict(
            step_ms=summarize(steps), prompt_step_ms=summarize(prompt_steps),
            trunk_call_ms=summarize(t["ms"] for t in trunk),
            trunk_ms_per_frame=sum(t["ms"] for t in trunk) / max(frames_enc, 1),
            trunk_frames_encoded=frames_enc,
            propagate_wall_ms_per_frame=prop_wall / max(prop_frames, 1),
            throughput_fps=1e3 * prop_frames / max(prop_wall, 1e-9)),
        memory=dict(peak_mib_max=max(r["peak_mib"] for r in records),
                    peak_mib_median=float(np.median([r["peak_mib"] for r in records]))),
        eventful_reuse=dict(steady_trunk_calls=len(ks := [k for r in records for k in r.get("eventful_k", [])]),
                            k_mean=sum(ks) / len(ks) if ks else None,
                            recompute_frac=sum(ks) / len(ks) / 5184 if ks else None)
        if any("eventful_k" in r for r in records) else None,
        adapter_sessions=dict(encodes=sum(r["adapter"]["encodes"] for r in records if r.get("adapter")),
                              reprimes=sum(r["adapter"]["reprimes"] for r in records if r.get("adapter")))
        if args.adapter else None,
        provenance=prov)


def write_md(path, s):
    L = s["latency"]
    lines = [f"# Pipeline run: {s['subset']} / {s.get('eventful') or s['preset']} / grounding batch {s['grounding_batch']} / "
             f"postprocess batch {s['postprocess_batch']}{' / adapter' if s['adapter'] else ''}", "",
             f"{s['pairs']} pairs ({s['pairs_ok']} completed, {len(s['pairs_crashed'])} crashed on the upstream tracker "
             f"bug and excluded from the metrics), {s['videos']} videos, {s['propagate_frames']} propagated frames, "
             f"hotstart_delay {s['hotstart_delay']}. Commit {s['provenance']['git_head']}, "
             f"GPU {s['provenance']['gpu']['physical_index']} ({s['provenance']['gpu']['name']}).", "",
             "| metric | value |", "|---|---:|"]
    lines += [f"| {k} ({KEY_METRICS[k]}) | {v:.4f} |" for k, v in s["key_metrics"].items() if v is not None]
    lines += ["", "| latency | median | p90 | max | mean |", "|---|---:|---:|---:|---:|"]
    for k in ("step_ms", "prompt_step_ms", "trunk_call_ms"):
        v = L[k]
        if v:
            lines.append(f"| {k} | {v['median']:.2f} | {v['p90']:.2f} | {v['max']:.2f} | {v['mean']:.2f} |")
    lines += ["", f"trunk ms per encoded frame: {L['trunk_ms_per_frame']:.2f} ({L['trunk_frames_encoded']} frames)",
              f"propagation wall ms per frame: {L['propagate_wall_ms_per_frame']:.2f}; "
              f"throughput {L['throughput_fps']:.2f} frames/s",
              f"peak memory: max {s['memory']['peak_mib_max']:.0f} MiB, median {s['memory']['peak_mib_median']:.0f} MiB"]
    g = s.get("gpu_state", {})
    if g.get("sm_mhz"):
        lines.append(f"GPU state while measuring ({g['samples']} samples): SM clock median {g['sm_mhz']['median']:.0f} MHz "
                     f"(p10 {g['sm_mhz']['p10']:.0f}), temperature median {g['temp_c']['median']:.0f} C (max "
                     f"{g['temp_c']['max']:.0f}), power median {g['power_w']['median']:.0f} W")
    Path(path).write_text("\n".join(lines) + "\n")
