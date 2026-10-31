"""Install EPViT as the ViT trunk of the complete SAM 3.1 video pipeline (detector + multiplex tracker).

SAM 3.1 (Sam3MultiplexTracking) owns one ViT, held as `.trunk` by the tri-neck that feeds both the detector and the
tracker, so replacing that attribute puts EPViT under every image encode. Nothing else changes.

Three hooks, all on the model instance (SAM's classes are untouched):
  * `reset_state` also resets EPViT: every prompt session starts with a flush and an empty output cache;
  * `init_state` records a fingerprint of the video's preprocessed frames;
  * a forward pre-hook on EPViT matches each incoming frame against that fingerprint to get its frame index, which
    keys the codec priors. The pipeline indexes frames with GPU tensors, so this costs one small device-to-host read
    per trunk call. The hook runs before any timing hook registered after installation.
"""
from contextlib import contextmanager

import torch

from sam3.model.vitdet import ViT

FP_STRIDE = 7919


def shared_trunk(model):
    vits = {id(m): m for m in model.modules() if isinstance(m, ViT)}
    assert len(vits) == 1, f"expected one shared ViT trunk, found {len(vits)}"
    vit = next(iter(vits.values()))
    holders = [m for m in model.modules() if getattr(m, "trunk", None) is vit]
    assert holders, "no module holds the ViT as .trunk"
    return vit, holders


def _fingerprint(x):
    return x.flatten(1)[:, ::FP_STRIDE].float()


@contextmanager
def installed(model, trunk):
    """Use `trunk` (an EPViT wrapping this model's ViT, or any module with the trunk interface) inside `model`."""
    vit, holders = shared_trunk(model)
    assert getattr(trunk, "vit", None) is vit, "the replacement must wrap this model's own trunk"
    saved = {k: model.__dict__.get(k) for k in ("reset_state", "init_state")}
    orig_reset, orig_init = model.reset_state, model.init_state
    state = {"resets": 0, "video_fp": None, "frames": None}

    def reset_state(inference_state, *a, **kw):
        trunk.reset()
        state["resets"] += 1
        return orig_reset(inference_state, *a, **kw)

    def init_state(*a, **kw):
        st = orig_init(*a, **kw)
        frames = st["input_batch"].img_batch.tensors
        state["video_fp"] = _fingerprint(frames).to(frames.device)
        state["frames"] = frames.shape[0]
        return st

    def identify(m, args):
        if state["video_fp"] is None or not hasattr(m, "set_frame_ids") or getattr(m, "sched", None) is None:
            return                                    # no priors: frame ids are not needed (and cost a sync)
        x = getattr(args[0], "tensors", args[0])
        fp = _fingerprint(x).to(state["video_fp"].device)
        eq = (fp[:, None, :] == state["video_fp"][None]).all(-1)          # (B, T)
        found, idx = eq.any(1), eq.int().argmax(1)
        ids = torch.where(found, idx, torch.full_like(idx, -1)).tolist()   # the one device-to-host read
        m.set_frame_ids([i if i >= 0 else None for i in ids])

    for h in holders:
        h.trunk = trunk
    model.reset_state, model.init_state = reset_state, init_state
    hook = trunk.register_forward_pre_hook(identify)
    trunk.reset()
    try:
        yield state
    finally:
        hook.remove()
        for h in holders:
            h.trunk = vit
        for k, v in saved.items():
            if v is None:
                model.__dict__.pop(k, None)
            else:
                setattr(model, k, v)
        if hasattr(trunk, "release"):
            trunk.release()
