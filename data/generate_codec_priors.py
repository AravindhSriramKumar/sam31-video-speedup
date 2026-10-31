"""H.264 codec priors for EP-ViT (motion vectors, residuals, intra share, frame types on the 72x72 token grid; CPU).

    python data/generate_codec_priors.py --sample                # data/sample_priors/crf23_g192_iffix/sample_video.npz
    python data/generate_codec_priors.py --eval                  # data/saco_sav/priors/crf23_g192_iffix/<video>.npz
    python data/generate_codec_priors.py --eval --videos sav_017083

--eval encodes the frames the 24 fps protocol feeds SAM (native frames 0 .. last annotated) with epvit/codec/encode.py
and extracts priors with epvit/codec/extract.py. --sample extracts from data/sample_video.mp4 directly, since that
file already is this H.264 encode of sav_017083.

"_iffix": extract.py reads `frame.pict_type.name`, which PyAV 19 does not have (pict_type is an int), so every frame
after frame 0 comes out labelled 'P' and EP-ViT would never flush on a later I-frame. `ftype` is therefore replaced by
ffprobe's frame types here; the other arrays are unchanged because the scheduler flushes before reading them.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

import paths  # noqa: E402
from data import saco_data  # noqa: E402
from epvit.codec.encode import enc_tag, encode, frame_types  # noqa: E402
from epvit.codec.extract import extract  # noqa: E402

CRF, GOP = 23, 192
TAG = f"{enc_tag(CRF, GOP)}_iffix"


def fix_ftype(arrays, video):
    ff = np.array(frame_types(video))
    assert len(ff) == len(arrays["ftype"]) and set(ff) <= {"I", "P"}, video
    n = int((ff != arrays["ftype"]).sum())
    arrays["ftype"] = ff
    return n


def save(arrays, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def sample():
    from experiments.harness.sample import probe

    w, h, n = probe()
    out = paths.SAMPLE_PRIORS / TAG / "sample_video.npz"
    arrays, _ = extract(paths.SAMPLE_VIDEO, (w, h))
    assert len(arrays["ftype"]) == n, (len(arrays["ftype"]), n)
    relabelled = fix_ftype(arrays, paths.SAMPLE_VIDEO)
    save(arrays, out)
    print(f"sample: {n} frames, {relabelled} relabelled P -> I -> {out}")


def evaluation(names, workers):
    gt = saco_data.load_gt()
    by_name = saco_data.video_by_name(gt)
    subsets = saco_data.load_subsets()
    names = names or subsets["normal"] + subsets["high_motion"]

    def one(name):
        out = paths.SACO_PRIORS / TAG / f"{name}.npz"
        if out.exists():
            return print(f"  {name}: already present", flush=True)
        v = by_name[name]
        folder, ann_idx = saco_data.frame_dir_24fps(v)
        frames = [folder / f"{i}.jpg" for i in range(ann_idx[-1] + 1)]
        mp4 = paths.SACO_VIDEOS / enc_tag(CRF, GOP) / f"{name}.mp4"
        encode(frames, mp4, crf=CRF, gop=GOP, fps=24, threads=2)
        arrays, _ = extract(mp4, (v["width"], v["height"]))
        assert len(arrays["ftype"]) == len(frames), (name, len(arrays["ftype"]), len(frames))
        relabelled = fix_ftype(arrays, mp4)
        save(arrays, out)
        print(f"  {name}: {len(frames)} frames, {relabelled} relabelled P -> I", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, names))
    print(f"{len(names)} videos -> {paths.SACO_PRIORS / TAG} ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true")
    g.add_argument("--eval", action="store_true")
    ap.add_argument("--videos", nargs="*", default=None, help="--eval: only these videos")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    sample() if a.sample else evaluation(a.videos, a.workers)


if __name__ == "__main__":
    main()
