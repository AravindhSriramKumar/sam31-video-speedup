"""EP-ViT on the 24 fps protocol, or on the included sample. Only the module installed as the ViT trunk differs from
dense SAM 3.1: same predictor build (use_fa3=False, use_rope_real=True, grounding and postprocess batch 1, hotstart 15),
same weights, same sessions, same StepTimer, same prediction format, same official evaluator.

  A_ep100        EPViT, budget k = 5184 (every token, every frame)
  B..F_ep90..50  EPViT, budget k = round(f * 5184): all operations of all blocks on the plan's rows
  G_ep75_qkv     EPViTSelective(ops=("qkv",)), k = 3888
  H_ep75_mlp     EPViTSelective(ops=("mlp",)), k = 3888

EP-ViT code (schedule, warps, residual stream, caches, flushes, output cache for repeated frames) is not modified.
Measurement-only additions: host-time wrappers around the scheduler step and the plan upload (gating overhead on the
CPU), and a per-frame log copy. The codec priors are precomputed (data/generate_codec_priors.py); their cost is reported
separately and is not part of the timed step.

The reported configuration is D_ep75 with --compile:

    CUDA_VISIBLE_DEVICES=0 python experiments/run_epvit.py --sample --config D_ep75 --compile
    CUDA_VISIBLE_DEVICES=0 python experiments/run_epvit.py --subset normal --config D_ep75 --compile
    CUDA_VISIBLE_DEVICES=0 python experiments/run_epvit.py --subset high_motion --config D_ep75 --compile

Full runs write results/runs/epvit/<subset>__<config>_c__<enc>__24fps/ and resume from its pairs/.
"""
import argparse
import functools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import paths  # noqa: E402
from data import saco_data  # noqa: E402
from experiments.harness import env, sample  # noqa: E402
from experiments.harness.baseline import PRESETS, apply_baseline  # noqa: E402
from experiments.harness.instrument import StepTimer  # noqa: E402
from experiments.harness.pipeline import find_vit  # noqa: E402
from experiments.harness.run_pipeline_eval import finalize, predictions_from, run_session, summarize_run, write_md  # noqa: E402
from experiments.harness.timing import GpuStateSampler, summarize  # noqa: E402

from epvit.codec.schedule import N_TOK, ScheduleConfig, load_priors  # noqa: E402
from epvit.model.epvit import EPViT, trunk_macs  # noqa: E402
from epvit.model.install import installed  # noqa: E402
from epvit.model.selective import EPViTSelective  # noqa: E402

ENC = "crf23_g192_iffix"

CONFIGS = {
    "A_ep100": dict(frac=1.0, ops="all"),
    "B_ep90": dict(frac=0.9, ops="all"),
    "C_ep80": dict(frac=0.8, ops="all"),
    "D_ep75": dict(frac=0.75, ops="all"),
    "E_ep60": dict(frac=0.6, ops="all"),
    "F_ep50": dict(frac=0.5, ops="all"),
    "G_ep75_qkv": dict(frac=0.75, ops=("qkv",)),
    "H_ep75_mlp": dict(frac=0.75, ops=("mlp",)),
}


def make_trunk(vit, name, compiled=False):
    c = CONFIGS[name]
    cfg = ScheduleConfig(mode="budget", k=int(round(c["frac"] * N_TOK)))      # notebook 07's rounding
    if compiled:                                                              # epvit_compiled.py: same results, one
        from epvit_compiled import CompiledEPViT, CompiledEPViTSelective      # torch.compile graph per frame
        if c["ops"] == "all":
            return CompiledEPViT(vit, cfg)
        return CompiledEPViTSelective(vit, cfg, ops=c["ops"])
    if c["ops"] == "all":
        return EPViT.from_vit(vit, cfg)                                       # the original EP-ViT (notebook 06)
    return EPViTSelective(vit, cfg, ops=c["ops"])                             # notebook 07's op-selective variant


def frame_macs(trunk, row):
    return trunk.trunk_macs(row) if isinstance(trunk, EPViTSelective) else trunk_macs(row)


class HostTimers:
    """Host (CPU) time of EP-ViT's per-frame gating work: Scheduler.step (classification, Eq. 11, top-k on the host)
    and EPViT._device_plan (index arrays, pinned H2D copy). Instance-level wrappers; results are unchanged."""

    def __init__(self, trunk):
        self.trunk, self.sched_ms, self.plan_ms = trunk, [], []
        orig_plan = trunk._device_plan

        def plan(*a, **kw):
            t0 = time.perf_counter()
            out = orig_plan(*a, **kw)
            self.plan_ms.append(1e3 * (time.perf_counter() - t0))
            return out

        trunk._device_plan = plan

    def attach(self):                                     # the scheduler is rebuilt by every set_video
        s = self.trunk.sched
        orig_step = s.step

        def step(*a, **kw):
            t0 = time.perf_counter()
            out = orig_step(*a, **kw)
            self.sched_ms.append(1e3 * (time.perf_counter() - t0))
            return out

        s.step = step

    def take(self):
        out = dict(sched_ms=self.sched_ms, plan_ms=self.plan_ms)
        self.sched_ms, self.plan_ms = [], []
        return out


def warmup_dir(video, n=16):
    """First n native 24 fps frames of a video (symlinks), so the warm-up session exercises flush and sparse frames."""
    src, _ = saco_data.frame_dir_24fps(video)
    out = saco_data.SACO / "warmup_frames" / f"{video['video_name']}__first{n}"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        dst = out / f"{i}.jpg"
        if not dst.exists():
            dst.symlink_to((src / f"{i}.jpg").resolve())
    return out


def run_sample(args, pred, vit, trunk, host):
    """Smoke test on data/sample_video.mp4: warm-up, then one session with the sample's own priors."""
    pr = load_priors(paths.SAMPLE_PRIORS / args.enc / "sample_video.npz")
    n = sample.probe()[2]
    assert len(pr["ftype"]) == n, (len(pr["ftype"]), n)
    with installed(pred.model, trunk), torch.inference_mode():
        functools.update_wrapper(pred.model.init_state, type(pred.model).init_state.__get__(pred.model))
        timer = StepTimer(pred, vit, trunk_module=trunk)
        trunk.set_video("sample_video", pr)
        host.attach()
        run_session(pred, timer, str(sample.warmup_dir(args.warmup_frames)), args.prompt)
        timer.take(); host.take(); trunk.log.clear()
        trunk.set_video("sample_video", pr)
        host.attach()
        method = f"epvit_{args.config}{'_c' if args.compile else ''}"
        sample.run(pred, timer, method, args.prompt, args.out)
        modes = {}
        for r in trunk.log:
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        print(f"frame modes {modes}, {sum(r['M'] for r in trunk.log) / max(len(trunk.log), 1):.0f} tokens per frame",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--subset", help="normal | high_motion | smoke (data/subsets.yaml)")
    g.add_argument("--sample", action="store_true", help="smoke test on data/sample_video.mp4; no accuracy")
    ap.add_argument("--prompt", default=sample.DEFAULT_PROMPT, help="--sample: text prompt")
    ap.add_argument("--config", default="D_ep75", choices=sorted(CONFIGS))
    ap.add_argument("--enc", default=ENC, help="priors enc_tag")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--pair-ids", default=None, help="JSON list of pair ids: run only these pairs")
    ap.add_argument("--warmup-frames", type=int, default=16)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compile", action="store_true",
                    help="epvit_compiled.py: torch.compile(mode='default', dynamic=False), as dense B2d / Eventful ':c'")
    ap.add_argument("--no-output-cache", action="store_true",
                    help="diagnostics only: EPViT(cache_frames=0), repeated frames are re-encoded")
    args = ap.parse_args()
    args.preset, args.grounding_batch, args.postprocess_batch, args.adapter, args.fps = "B1", 1, 1, False, 24

    tag = f"{args.subset}__{args.config}{'_c' if args.compile else ''}__{args.enc}__24fps"
    out_dir = Path(args.out) if args.out else paths.RUNS / "epvit" / tag
    if not args.sample:
        (out_dir / "pairs").mkdir(parents=True, exist_ok=True)
    prov = env.provenance()
    prov["epvit"] = dict(config=args.config, spec=CONFIGS[args.config], enc=args.enc,
                         compile=dict(mode="default", dynamic=False) if args.compile else None)

    pred = env.build_predictor(args.grounding_batch, args.postprocess_batch, None)
    vit = find_vit(pred.model)
    dense_trunk = apply_baseline(vit, PRESETS[args.preset])
    assert dense_trunk is vit                              # B1 is eager
    trunk = make_trunk(vit, args.config, compiled=args.compile)
    if args.compile:                                       # every graph compiled before anything is timed
        trunk.warm(torch.zeros(1, 3, 1008, 1008, device="cuda"))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if args.no_output_cache:
        trunk.cache_frames = 0
    prov["epvit"]["output_cache_frames"] = trunk.cache_frames
    host = HostTimers(trunk)

    if args.sample:
        run_sample(args, pred, vit, trunk, host)
        return

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
    pdir = paths.SACO_PRIORS / args.enc
    priors = {}

    def priors_for(name):
        if name not in priors:
            priors.clear()
            priors[name] = load_priors(pdir / f"{name}.npz")
        return priors[name]

    with installed(pred.model, trunk) as inst, torch.inference_mode():
        # install.py's init_state wrapper takes (*a, **kw); the predictor's start_session filters its kwargs by
        # inspect.signature(model.init_state), so expose the original signature (behaviour unchanged)
        functools.update_wrapper(pred.model.init_state, type(pred.model).init_state.__get__(pred.model))
        timer = StepTimer(pred, vit, trunk_module=trunk)  # registered after install: excludes its frame-id lookup
        v0, p0 = todo[0]                                  # warm-up session, discarded
        trunk.set_video(v0["video_name"], priors_for(v0["video_name"]))
        host.attach()
        run_session(pred, timer, str(warmup_dir(v0, args.warmup_frames)), p0["noun_phrase"])
        timer.take(); host.take(); trunk.log.clear()
        sampler = GpuStateSampler(env.physical_gpu(), period_s=5.0).__enter__()
        seg_start = time.strftime("%Y%m%d-%H%M%S")
        for i, (v, p) in enumerate(todo):
            f = out_dir / "pairs" / f"{p['id']}.json"
            if f.exists():
                continue
            torch.cuda.reset_peak_memory_stats()
            folder, ann_idx = saco_data.frame_dir_24fps(v)
            frame_map = {m: j for j, m in enumerate(ann_idx)}
            pr = priors_for(v["video_name"])
            assert len(pr["ftype"]) == ann_idx[-1] + 1, (v["video_name"], len(pr["ftype"]), ann_idx[-1] + 1)
            trunk.set_video(v["video_name"], pr)          # resets the reuse state and the output cache
            host.attach()
            trunk.log.clear()
            resets0 = inst["resets"]
            outs, prompt_ms, prop_ms, n_yield, status = run_session(pred, timer, str(folder), p["noun_phrase"])
            log = list(trunk.log)
            state_mib = trunk.state_bytes() / 2**20
            modes = {}
            for r in log:
                modes[r["mode"]] = modes.get(r["mode"], 0) + 1
            if status == "ok":
                assert modes.get("dense_no_prior", 0) == 0, f"frames not identified: {modes}"
            macs = [frame_macs(trunk, r) for r in log]
            rec = dict(pair_id=p["id"], status=status, video=v["video_name"], video_id=v["id"],
                       category_id=p["category_id"], noun_phrase=p["noun_phrase"], num_masklets_gt=p["num_masklets"],
                       frames=v["length"], frames_yielded=n_yield, prompt_wall_ms=prompt_ms, propagate_wall_ms=prop_ms,
                       peak_mib=torch.cuda.max_memory_allocated() / 2**20, hotstart_delay=pred.model.hotstart_delay,
                       objects_per_frame=[len(outs[k]["out_obj_ids"]) for k in sorted(outs)],
                       timing=timer.take(), adapter=None, predictions=predictions_from(outs, v, p, frame_map),
                       fps=24, model_frames=len(outs),
                       epvit=dict(config=args.config, enc=args.enc, resets=inst["resets"] - resets0, modes=modes,
                                  frames_in=len(log), tokens_recomputed=int(sum(r["M"] for r in log)),
                                  trunk_macs=float(sum(macs)), state_mib=state_mib, host=host.take(),
                                  log=[[r["fid"], r["mode"][0], r["M"]] for r in log]))
            env.write_json(f, rec)
            st = [s["ms"] for s in rec["timing"]["steps"] if s["phase"] == "propagate"]
            print(f"[{i + 1}/{len(todo)}] {v['video_name']} '{p['noun_phrase']}': {status}, {n_yield} frames, "
                  f"step median {np.median(st) if st else float('nan'):.1f} ms, modes {modes}, "
                  f"{len(rec['predictions'])} objects", flush=True)
        sampler.__exit__()
        env.write_json(out_dir / "gpu_state" / f"segment_{seg_start}.json",
                       dict(summary=sampler.summary(), samples=sampler.samples, provenance=prov))

    records = [json.loads((out_dir / "pairs" / f"{p['id']}.json").read_text()) for _, p in todo]
    metrics = finalize(out_dir, gt, [r for r in records if r["status"] == "ok"])
    summary = summarize_run(records, metrics, args, prov)
    fin = sum(r["epvit"]["frames_in"] for r in records)
    summary["epvit"] = dict(
        config=args.config, spec=CONFIGS[args.config], enc=args.enc, k=trunk.cfg.k, schedule=trunk.cfg.as_dict(),
        trunk_class=type(trunk).__name__, frames_in=fin,
        modes={m: sum(r["epvit"]["modes"].get(m, 0) for r in records) for m in ("flush", "sparse", "cached")},
        tokens_recomputed_per_frame_in=sum(r["epvit"]["tokens_recomputed"] for r in records) / max(fin, 1),
        trunk_tmac_per_frame_in=sum(r["epvit"]["trunk_macs"] for r in records) / max(fin, 1) / 1e12,
        sched_host_ms=summarize(x for r in records for x in r["epvit"]["host"]["sched_ms"]),
        plan_host_ms=summarize(x for r in records for x in r["epvit"]["host"]["plan_ms"]),
        state_mib_max=max(r["epvit"]["state_mib"] for r in records))
    segs = [json.loads(f.read_text()) for f in sorted((out_dir / "gpu_state").glob("segment_*.json"))]
    allsamp = [x for sg in segs for x in sg["samples"]]
    summary["gpu_state"] = dict(segments=len(segs), samples=len(allsamp),
                                **{k: summarize(x[k] for x in allsamp) for k in ("sm_mhz", "temp_c", "power_w")}
                                if allsamp else {})
    env.write_json(out_dir / "summary.json", summary)
    write_md(out_dir / "summary.md", summary)
    print(json.dumps(summary["key_metrics"], indent=1))
    print(json.dumps({k: v for k, v in summary["epvit"].items() if k != "schedule"}, indent=1))


if __name__ == "__main__":
    main()
