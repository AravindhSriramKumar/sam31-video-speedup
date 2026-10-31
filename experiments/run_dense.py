"""Dense SAM 3.1 (B2d: compiled trunk with bf16 linear weights) on the 24 fps protocol, or on the included sample.

    CUDA_VISIBLE_DEVICES=0 python experiments/run_dense.py --sample
    CUDA_VISIBLE_DEVICES=0 python experiments/run_dense.py --subset normal
    CUDA_VISIBLE_DEVICES=0 python experiments/run_dense.py --subset high_motion

Full runs write results/runs/dense/<subset>__B2d__gb1_pp1__adapter__24fps/ and resume from its pairs/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import paths  # noqa: E402
from data import saco_data  # noqa: E402
from experiments.harness import env, sample  # noqa: E402
from experiments.harness.baseline import PRESETS, apply_baseline  # noqa: E402
from experiments.harness.instrument import StepTimer  # noqa: E402
from experiments.harness.pipeline import PipelineHandle, find_vit  # noqa: E402
from experiments.harness.run_pipeline_eval import finalize, predictions_from, run_session, summarize_run, write_md  # noqa: E402
from experiments.harness.timing import GpuStateSampler, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subset", help="normal | high_motion | smoke (data/subsets.yaml)")
    g.add_argument("--sample", action="store_true", help="smoke test on data/sample_video.mp4; no accuracy")
    ap.add_argument("--prompt", default=sample.DEFAULT_PROMPT, help="--sample: text prompt")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--pair-ids", default=None, help="JSON list of pair ids: run only these pairs")
    ap.add_argument("--warmup-frames", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.preset, args.grounding_batch, args.postprocess_batch, args.adapter, args.fps = "B2d", 1, 1, True, 24

    prov = env.provenance()
    pred = env.build_predictor(args.grounding_batch, args.postprocess_batch, None)
    vit = find_vit(pred.model)
    trunk = apply_baseline(vit, PRESETS[args.preset])
    with torch.inference_mode():                          # compile before anything is timed
        trunk(torch.zeros(1, 3, 1008, 1008, device="cuda"))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    handle = PipelineHandle(pred, None, trunk=trunk)
    timer = StepTimer(pred, vit, trunk_module=handle.adapter)

    if args.sample:
        with torch.inference_mode():
            run_session(pred, timer, str(sample.warmup_dir(args.warmup_frames)), args.prompt)
            timer.take()
            handle.last_session()
            sample.run(pred, timer, "dense_B2d", args.prompt, args.out)
        return

    tag = f"{args.subset}__{args.preset}__gb1_pp1__adapter__24fps"
    out_dir = Path(args.out) if args.out else paths.RUNS / "dense" / tag
    (out_dir / "pairs").mkdir(parents=True, exist_ok=True)

    gt = saco_data.load_gt()
    by_name = saco_data.video_by_name(gt)
    todo = []
    for name in saco_data.load_subsets()[args.subset]:
        v = by_name[name]
        todo += [(v, p) for p in saco_data.pairs_for_video(gt, v["id"])]
    todo = todo[:args.max_pairs] if args.max_pairs else todo
    if args.pair_ids:
        ids = set(json.loads(Path(args.pair_ids).read_text()))
        todo = [(v, p) for v, p in todo if p["id"] in ids]

    with torch.inference_mode():
        v0, p0 = todo[0]                                  # warm-up session, discarded
        run_session(pred, timer, str(saco_data.truncated_frame_dir(v0, args.warmup_frames)), p0["noun_phrase"])
        timer.take()
        handle.last_session()
        # GPU state (SM clock / temperature / power) every 5 s while this process measures pairs; one file per
        # process segment (a resumed run adds a segment), merged in the summary.
        sampler = GpuStateSampler(env.physical_gpu(), period_s=5.0).__enter__()
        seg_start = time.strftime("%Y%m%d-%H%M%S")
        for i, (v, p) in enumerate(todo):
            f = out_dir / "pairs" / f"{p['id']}.json"
            if f.exists():
                continue
            torch.cuda.reset_peak_memory_stats()
            folder, ann_idx = saco_data.frame_dir_24fps(v)
            frame_map = {m: j for j, m in enumerate(ann_idx)}
            outs, prompt_ms, prop_ms, n_yield, status = run_session(pred, timer, str(folder), p["noun_phrase"])
            rec = dict(pair_id=p["id"], status=status, video=v["video_name"], video_id=v["id"], category_id=p["category_id"],
                       noun_phrase=p["noun_phrase"], num_masklets_gt=p["num_masklets"], frames=v["length"],
                       frames_yielded=n_yield, prompt_wall_ms=prompt_ms, propagate_wall_ms=prop_ms,
                       peak_mib=torch.cuda.max_memory_allocated() / 2**20,
                       hotstart_delay=pred.model.hotstart_delay,
                       objects_per_frame=[len(outs[k]["out_obj_ids"]) for k in sorted(outs)],
                       timing=timer.take(), adapter=None, predictions=predictions_from(outs, v, p, frame_map),
                       fps=args.fps, model_frames=len(outs))
            s = handle.last_session()
            rec["adapter"] = dict(encodes=s["encodes"], reprimes=s["reprimes"])
            env.write_json(f, rec)
            st = [s["ms"] for s in rec["timing"]["steps"] if s["phase"] == "propagate"]
            print(f"[{i + 1}/{len(todo)}] {v['video_name']} '{p['noun_phrase']}': {status}, {n_yield} frames, "
                  f"step median {np.median(st) if st else float('nan'):.1f} ms, {len(rec['predictions'])} objects",
                  flush=True)
        sampler.__exit__()
        env.write_json(out_dir / "gpu_state" / f"segment_{seg_start}.json",
                       dict(summary=sampler.summary(), samples=sampler.samples, provenance=prov))

    records = [json.loads((out_dir / "pairs" / f"{p['id']}.json").read_text()) for _, p in todo]
    metrics = finalize(out_dir, gt, [r for r in records if r["status"] == "ok"])
    summary = summarize_run(records, metrics, args, prov)
    segs = [json.loads(f.read_text()) for f in sorted((out_dir / "gpu_state").glob("segment_*.json"))]
    allsamp = [x for sg in segs for x in sg["samples"]]
    summary["gpu_state"] = dict(segments=len(segs), samples=len(allsamp),
                                **{k: summarize(x[k] for x in allsamp) for k in ("sm_mhz", "temp_c", "power_w")}
                                if allsamp else {})
    env.write_json(out_dir / "summary.json", summary)
    write_md(out_dir / "summary.md", summary)
    print(json.dumps(summary["key_metrics"], indent=1))


if __name__ == "__main__":
    main()
