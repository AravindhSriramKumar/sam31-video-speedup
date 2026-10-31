"""Tests 3 and 4: the EventfulBlock wrapper is numerically identical to SAM's Block.

Uses real frames (2 annotated frames of SA-V sav_030331 and 2 of sav_018743) and real block inputs captured
with forward pre-hooks during a stock trunk run. Same kernels on the same inputs => expected bitwise equality.
Tests run in order: first under B0 (stock), then the trunk is switched to B1 (RoPE compiled + bf16 weights)
and the same checks are repeated.

    python tests/eventful/test_block_equivalence.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from eventful import env, frames, metrics  # noqa: E402
from eventful.baseline import cast_linear_weights_bf16, set_rope_mode  # noqa: E402
from eventful.block import EventfulBlock  # noqa: E402
from eventful.controller import EventfulController  # noqa: E402
from eventful.tolerances import SAME_KERNELS_REL  # noqa: E402

S = {}


def _setup():
    if S:
        return
    env.setup_runtime()
    torch.inference_mode().__enter__()
    S["vit"] = env.load_trunk()
    a, _ = frames.load_clip("sav_030331@24fps", max_frames=2)
    b, _ = frames.load_clip("sav_018743@24fps", max_frames=2)
    S["x"] = frames.to_trunk_input(torch.cat([a, b]))


def _block_inputs(vit, x):
    """Input of every block for one frame (B=1), captured from a stock forward."""
    caps, hooks = {}, []
    for i, blk in enumerate(vit.blocks):
        hooks.append(blk.register_forward_pre_hook(lambda m, args, i=i: caps.__setitem__(i, args[0].clone())))
    vit(x)
    for h in hooks:
        h.remove()
    return caps


def _check_blocks():
    vit, rows = S["vit"], []
    for f in range(S["x"].shape[0]):
        caps = _block_inputs(vit, S["x"][f:f + 1])
        for i, blk in enumerate(vit.blocks):
            ref = blk(caps[i])
            out = EventfulBlock(blk, i)(caps[i])
            st = metrics.feature_stats(out, ref, channel_dim=-1)
            rows.append(dict(frame=f, block=i, window=blk.window_size, bitwise=st["bitwise_equal"],
                             rel_err=st["rel_err"]))
    bad = [r for r in rows if not (r["bitwise"] or r["rel_err"] <= SAME_KERNELS_REL)]
    assert not bad, f"{len(bad)} block/frame pairs differ: {bad[:3]}"
    return dict(checked=len(rows), bitwise=sum(r["bitwise"] for r in rows),
                max_rel_err=max(r["rel_err"] for r in rows))


def _check_trunk():
    vit, x = S["vit"], S["x"]
    rows = []
    for f in range(x.shape[0]):
        ref = vit(x[f:f + 1])[0]
        c = EventfulController(vit)
        out = vit(x[f:f + 1])[0]
        c.restore()
        st = metrics.feature_stats(out, ref)
        rows.append(dict(frame=f, bitwise=st["bitwise_equal"], rel_err=st["rel_err"]))
    assert all(r["bitwise"] or r["rel_err"] <= SAME_KERNELS_REL for r in rows), rows
    return dict(frames=rows)


def test_1_block_wrapper_bitwise_B0():
    _setup()
    set_rope_mode("stock")
    return _check_blocks()


def test_2_trunk_wrapped_bitwise_B0():
    _setup()
    return _check_trunk()


def test_3_block_wrapper_bitwise_B1():
    _setup()
    set_rope_mode("compile")
    cast_linear_weights_bf16(S["vit"])
    return _check_blocks()


def test_4_trunk_wrapped_bitwise_B1():
    _setup()
    return _check_trunk()


if __name__ == "__main__":
    from tests.eventful._runner import run

    run(globals(), "test_block_equivalence")
