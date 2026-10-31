"""SA-Co/VEval (SA-V val) helpers: annotations, per-subset frame folders, filtered GT files.

Layout (per sam3/scripts/eval/veval/README.md), under data/:
  saco_sav/annotation/saco_veval_sav_val.json         (downloaded from HF facebook/SACo-VEval)
  saco_sav/JPEGImages_24fps/<video>/<nnnnn>.jpg       (see download_saco_eval.sh)
  saco_sav/frames/<video>/<i>.jpg                     (symlinks to the annotated 6 fps frames, made here)
  saco_sav/frames24/<video>/<i>.jpg                   (symlinks to native frames 0 .. last annotated, made here)

CLI:
  python data/saco_data.py --download-annotations
  python data/saco_data.py --check          # which subset videos have frames, NP pairs per video
  python data/saco_data.py --list           # all SA-V val videos with length / #NPs (to pick subsets)
"""
import argparse
import json
from pathlib import Path

import yaml

DATA = Path(__file__).resolve().parent
SACO = DATA / "saco_sav"
GT_FILE = SACO / "annotation/saco_veval_sav_val.json"
MEDIA = SACO / "JPEGImages_24fps"
FRAMES = SACO / "frames"
SUBSETS_FILE = DATA / "subsets.yaml"
EVAL_PAIRS_FILE = DATA / "eval_pairs.json"     # the frozen protocol: 20 videos, 198 attempted and 189 scored pairs


def download_annotations():
    from huggingface_hub import hf_hub_download

    return hf_hub_download("facebook/SACo-VEval", "annotation/saco_veval_sav_val.json",
                           repo_type="dataset", local_dir=str(SACO))


def load_gt(path=GT_FILE):
    with open(path) as fh:
        return json.load(fh)


def load_subsets(path=SUBSETS_FILE):
    with open(path) as fh:
        s = yaml.safe_load(fh) or {}
    return {k: list(v or []) for k, v in s.items()}


def load_eval_pairs(path=EVAL_PAIRS_FILE):
    with open(path) as fh:
        return json.load(fh)


def video_by_name(gt):
    return {v["video_name"]: v for v in gt["videos"]}


def pairs_for_video(gt, video_id, max_nps=None):
    """Video-NP pairs for one video: positives first (most masklets first), then negatives; by id."""
    ps = [p for p in gt["video_np_pairs"] if p["video_id"] == video_id]
    ps.sort(key=lambda p: (p["num_masklets"] == 0, -p["num_masklets"], p["id"]))
    return ps[:max_nps] if max_nps else ps


def frame_dir(video):
    """Folder with only the annotated frames, renamed 0.jpg..T-1.jpg (what the predictor loads)."""
    out = FRAMES / video["video_name"]
    names = video["file_names"]
    if out.is_dir() and len(list(out.glob("*.jpg"))) == len(names):
        return out
    out.mkdir(parents=True, exist_ok=True)
    for i, fn in enumerate(names):
        src = (MEDIA / fn).resolve()
        if not src.exists():
            raise FileNotFoundError(f"missing SA-V frame {src} (run bash data/download_saco_eval.sh)")
        dst = out / f"{i}.jpg"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    return out


FRAMES24 = SACO / "frames24"


def frame_dir_24fps(video):
    """Native 24 fps frames 0 .. last annotated frame, renamed 0.jpg..N-1.jpg (symlinks into MEDIA), and the 24 fps
    index of every annotated frame (annotated frame j is 24 fps frame int(stem of file_names[j])). Frames after the
    last annotated one are left out: online, forward-only processing cannot use them for any scored frame."""
    idx = [int(Path(n).stem) for n in video["file_names"]]
    out = FRAMES24 / video["video_name"]
    n = idx[-1] + 1
    if not (out.is_dir() and len(list(out.glob("*.jpg"))) == n):
        out.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            src = (MEDIA / video["video_name"] / f"{i:05d}.jpg").resolve()
            if not src.exists():
                raise FileNotFoundError(f"missing SA-V 24 fps frame {src}")
            dst = out / f"{i}.jpg"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
    return out, idx


def truncated_frame_dir(video, n):
    """First n annotated frames (for profiler runs)."""
    src = frame_dir(video)
    out = FRAMES / f"{video['video_name']}__first{n}"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(min(n, video["length"])):
        dst = out / f"{i}.jpg"
        if not dst.exists():
            dst.symlink_to((src / f"{i}.jpg").resolve())
    return out


def filtered_gt(gt, video_ids, pair_keys, path):
    """GT json restricted to the given videos and (video_id, category_id) pairs, for VEvalEvaluator."""
    vids = set(video_ids)
    keys = set(pair_keys)
    sub = dict(
        info=gt["info"],
        videos=[v for v in gt["videos"] if v["id"] in vids],
        annotations=[a for a in gt["annotations"] if (a["video_id"], a["category_id"]) in keys],
        video_np_pairs=[p for p in gt["video_np_pairs"] if (p["video_id"], p["category_id"]) in keys],
    )
    cats = {c for _, c in keys}
    sub["categories"] = [c for c in gt["categories"] if c["id"] in cats]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(sub, fh)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-annotations", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.download_annotations:
        print(download_annotations())
    gt = load_gt()
    byname = video_by_name(gt)
    if a.list:
        for v in sorted(gt["videos"], key=lambda v: v["video_name"]):
            ps = pairs_for_video(gt, v["id"])
            print(f"{v['video_name']}  frames={v['length']:4d}  {v['width']}x{v['height']}  "
                  f"NPs={len(ps):3d} (pos {sum(p['num_masklets'] > 0 for p in ps)})")
    if a.check:
        for name, vids in load_subsets().items():
            print(f"[{name}] {len(vids)} videos")
            for vn in vids:
                v = byname.get(vn)
                if v is None:
                    print(f"  {vn}: NOT in saco_veval_sav_val")
                    continue
                have = sum((MEDIA / f).exists() for f in v["file_names"])
                print(f"  {vn}: {have}/{v['length']} frames on disk, "
                      f"{len(pairs_for_video(gt, v['id']))} NPs")


if __name__ == "__main__":
    main()
