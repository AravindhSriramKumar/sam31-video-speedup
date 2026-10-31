"""Clips for the trunk harness.

A clip is named `<name>@<regime>`:
  * `sav_XXXXXX@6fps`  - the SA-Co/VEval annotated frames (every 4th frame of the 24 fps source). This is what
                         the task metric is computed on.
  * `sav_XXXXXX@24fps` - the same SA-V video, consecutive 24 fps frames (more redundancy between frames).
  * `0001`             - SAM's sample clip (270 frames, camera motion); `shibuya_720p` etc. from paths.VIDEOS.

Frames are read with SAM's own loader (`sam3.model.io_utils.load_video_frames`): resized to 1008x1008 without
keeping the aspect ratio, normalised with mean = std = 0.5, stored as float16 on the CPU. The pipeline casts
them to float32 before the trunk (`sam3_image.py::_get_img_feats`), and `to_trunk_input` does the same.
"""
from pathlib import Path

import torch

from eventful.env import paths

CACHE = paths.DATA / "frames" / "_eventful"


def _numbered(folder):
    return sorted((p for p in Path(folder).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")),
                  key=lambda p: int(p.stem))


def clip_dir(spec, max_frames=None):
    """Folder of frames named 0.jpg..T-1.jpg (or the source folder if it already has integer names)."""
    name, _, regime = spec.partition("@")
    if name.startswith("sav_"):
        from data import saco_data

        if regime in ("", "6fps"):
            gt = saco_data.load_gt()
            folder = saco_data.frame_dir(saco_data.video_by_name(gt)[name])
        elif regime == "24fps":
            folder = saco_data.MEDIA / name
        else:
            raise ValueError(f"unknown regime in {spec}")
    elif name == "0001":
        folder = paths.SAM3_SAMPLE_VIDEO
    else:
        folder = paths.VIDEOS / name
    if not Path(folder).is_dir():
        raise FileNotFoundError(f"{spec}: {folder} does not exist")
    if max_frames is None:
        return Path(folder)
    files = _numbered(folder)[:max_frames]
    out = CACHE / f"{name}@{regime or 'native'}_first{len(files)}"
    out.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(files):
        dst = out / f"{i}{src.suffix}"
        if not dst.exists():
            dst.symlink_to(src.resolve())
    return out


def load_clip(spec, max_frames=None):
    """Returns (frames [T,3,1008,1008] float16 CPU, info dict)."""
    from sam3.model.io_utils import load_video_frames

    folder = clip_dir(spec, max_frames)
    images, h, w = load_video_frames(str(folder), image_size=1008, offload_video_to_cpu=True)
    images = images if isinstance(images, torch.Tensor) else torch.stack(list(images))
    return images, dict(spec=spec, folder=str(folder), frames=int(images.shape[0]), orig_h=h, orig_w=w)


def to_trunk_input(frames_fp16, device="cuda"):
    """What the pipeline feeds the trunk: float32 on the GPU (`image.to(dtype=torch.float32, device=...)`)."""
    return frames_fp16.to(device=device, dtype=torch.float32)
