"""The profiler labels are measurement-only.

1. Trunk B1 with every trunk label (eventful/annotate.py::annotate_trunk: trunk, blk{i}, block parts, rope, sdpa),
   emitted both as NVTX ranges and as record_functions, gives a bitwise identical output to the unlabelled trunk,
   on 2 real frames. After `remove()`, every hook is gone and the patched functions are SAM's originals again.
2. Same for eventful/nvtx.py (the labels inside our own modules): EventfulBlock-wrapped trunk with labels on in both
   forms is bitwise equal to labels off.
3. A method label on an attribute the instance already overrides restores that override on remove(); labels that
   were attached but never opened are reported.

    python tests/eventful/test_annotate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

import sam3.model.vitdet as vitdet  # noqa: E402
from eventful import env, frames, nvtx  # noqa: E402
from eventful.annotate import Annotator, annotate_trunk  # noqa: E402
from eventful.baseline import PRESETS, apply_baseline  # noqa: E402
from eventful.controller import EventfulController  # noqa: E402

S = {}


def _setup():
    if S:
        return
    env.setup_runtime()
    torch.inference_mode().__enter__()
    S["vit"] = env.load_trunk()
    S["trunk"], _ = apply_baseline(S["vit"], PRESETS["B1"])
    x, _ = frames.load_clip("sav_012075@24fps", max_frames=2)
    S["x"] = frames.to_trunk_input(x)


def _hook_count(mod):
    return sum(len(m._forward_pre_hooks) + len(m._forward_hooks) for m in mod.modules())


def _outputs(trunk, x):
    return [trunk(x[i:i + 1])[-1].clone() for i in range(x.shape[0])]


def test_1_trunk_labels_bitwise_and_removable():
    _setup()
    vit, trunk, x = S["vit"], S["trunk"], S["x"]
    hooks0, rope0, sdpa0 = _hook_count(vit), vitdet.apply_rotary_enc_real, vitdet.F.scaled_dot_product_attention
    ref = _outputs(trunk, x)
    ann = Annotator(nvtx=True, record_functions=True)
    annotate_trunk(ann, vit)
    added = _hook_count(vit) - hooks0
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        lab = _outputs(trunk, x)
    seen = {e.name for e in prof.events()}
    ann.remove()
    after = _outputs(trunk, x)
    equal = [bool(torch.equal(a, b)) for a, b in zip(ref, lab)]
    restored = dict(hooks=_hook_count(vit) == hooks0, rope=vitdet.apply_rotary_enc_real is rope0,
                    sdpa=vitdet.F.scaled_dot_product_attention is sdpa0)
    assert all(equal), equal
    assert all(torch.equal(a, b) for a, b in zip(ref, after))
    assert all(restored.values()), restored
    expect = {"trunk", "blk0", "blk31", "blk7.qkv", "blk7.attn", "blk7.mlp", "rope", "sdpa"}
    assert expect <= seen, expect - seen
    return dict(frames=len(ref), hooks_added=added, bitwise_equal=equal, restored=restored,
                labels_seen_by_torch_profiler=sorted(l for l in seen if l in expect))


def test_2_eventful_block_labels_bitwise():
    _setup()
    vit, trunk, x = S["vit"], S["trunk"], S["x"]
    ctl = EventfulController(vit)                     # swaps vit.blocks for EventfulBlock wrappers
    try:
        nvtx.enable(False)
        ref = _outputs(trunk, x)
        nvtx.enable(True, record_functions=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            lab = _outputs(trunk, x)
        seen = {e.name for e in prof.events()}
    finally:
        nvtx.enable(False)
        ctl.restore()
    equal = [bool(torch.equal(a, b)) for a, b in zip(ref, lab)]
    assert all(equal), equal
    assert {"blk0", "blk0.qkv", "blk0.mlp"} <= seen
    return dict(frames=len(ref), bitwise_equal=equal)


def test_3_method_label_restores_instance_wrappers():
    """`Annotator.method` on an attribute the instance already overrides (SAM wraps some `forward`s per instance)
    must put that override back on remove(), and a never-called label must be reported by never_opened()."""
    class M(torch.nn.Module):
        def forward(self, x):
            return x + 1

    m = M()
    own = lambda x: x * 2  # noqa: E731   an instance-level wrapper, as SAM installs
    m.forward = own
    plain = M()
    ann = Annotator(nvtx=False, record_functions=True)
    ann.method(m, "forward", "a")
    ann.method(plain, "forward", "b")
    out = m.forward(torch.tensor(3.0))
    never = ann.never_opened()
    ann.remove()
    assert float(out) == 6.0, out                 # the label wrapped SAM's override, not the class method
    assert m.__dict__.get("forward") is own        # override restored
    assert "forward" not in plain.__dict__         # no stray attribute where there was none
    assert never == ["b"], never
    return dict(labelled_output=float(out), never_opened=never)


if __name__ == "__main__":
    from _runner import run

    run(globals(), "test_annotate")
