"""SequentialTrunkAdapter + FrameIndexTap inside the full SAM 3.1 predictor.

On the first 12 annotated frames of sav_030331 (first noun phrase), one session = add_prompt on frame 0 +
forward propagation. Checks:
  1. grounding batch 1: the trunk sees frames 0, 1, 0, 1, ..., 11 (14 encodes) and the adapter re-primes
     exactly twice (at both frame-0 encodes);
  2. grounding batch 8: same frame sequence, trunk calls of batch 1, 1, 8, 4, still 14 encodes / 2 re-primes;
  3. with the adapter installed (dense, no controller), every output (object ids, probabilities, binary masks)
     is identical to the predictor without the adapter.

    python tests/eventful/test_pipeline_adapter.py      (~3 min; builds the predictor once)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from data import saco_data  # noqa: E402
from eventful import env  # noqa: E402
from eventful.pipeline import PipelineHandle  # noqa: E402

S = {}
N_FRAMES = 12


def _setup():
    if S:
        return
    S["pred"] = env.build_predictor(grounding_batch=1, postprocess_batch=1)
    S["gt"] = saco_data.load_gt()
    video = saco_data.video_by_name(S["gt"])["sav_030331"]
    S["dir"] = str(saco_data.truncated_frame_dir(video, N_FRAMES))
    S["np"] = saco_data.pairs_for_video(S["gt"], video["id"])[0]["noun_phrase"]


def _session(folder=None, np_text=None):
    p = S["pred"]
    folder, np_text = folder or S["dir"], np_text or S["np"]
    sid = p.handle_request(dict(type="start_session", resource_path=folder, offload_video_to_cpu=True))["session_id"]
    outs = {0: p.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0, text=np_text))["outputs"]}
    for r in p.handle_stream_request(dict(type="propagate_in_video", session_id=sid, propagation_direction="forward")):
        outs[r["frame_index"]] = r["outputs"]
    p.handle_request(dict(type="close_session", session_id=sid))
    return outs


def _order_check(grounding_batch):
    _setup()
    p = S["pred"]
    p.model.batched_grounding_batch_size = grounding_batch
    h = PipelineHandle(p)
    calls = []   # batch size of every call the neck makes into the adapter (= what SAM passes to its trunk)
    hook = h.adapter.register_forward_pre_hook(
        lambda m, a: calls.append(int(getattr(a[0], "tensors", a[0]).shape[0])))
    try:
        _session()
        sess = h.last_session()
    finally:
        hook.remove()
        h.uninstall()
        p.model.batched_grounding_batch_size = 1
    ids = [fid for fid, _ in sess["trace"]]
    reprime_at = [i for i, (_, r) in enumerate(sess["trace"]) if r]
    expected = [0, 1] + list(range(N_FRAMES))
    assert ids == expected, ids
    assert sess["encodes"] == N_FRAMES + 2 and sess["reprimes"] == 2 and reprime_at == [0, 2], sess
    return dict(frame_ids=ids, reprime_positions=reprime_at, encodes=sess["encodes"], trunk_call_batches=calls)


def test_1_call_order_grounding_batch_1():
    r = _order_check(1)
    assert r["trunk_call_batches"] == [1] * (N_FRAMES + 2), r
    return r


def test_2_call_order_grounding_batch_8():
    r = _order_check(8)
    assert r["trunk_call_batches"] == [1, 1, 8, N_FRAMES - 8], r
    return r


def _same(a, b):
    if sorted(a.keys()) != sorted(b.keys()):
        return False
    for f in a:
        x, y = a[f], b[f]
        if not np.array_equal(x["out_obj_ids"], y["out_obj_ids"]):
            return False
        if not np.array_equal(x["out_probs"], y["out_probs"]):
            return False
        if not np.array_equal(x["out_binary_masks"], y["out_binary_masks"]):
            return False
    return True


def test_3_adapter_outputs_identical():
    """Uses the first (video, noun phrase) of the smoke subset whose dense output contains objects, so that the
    comparison covers real masks, not empty outputs."""
    _setup()
    chosen = None
    for name in saco_data.load_subsets()["smoke"]:
        video = saco_data.video_by_name(S["gt"])[name]
        folder = str(saco_data.truncated_frame_dir(video, N_FRAMES))
        for pair in saco_data.pairs_for_video(S["gt"], video["id"])[:2]:
            ref = _session(folder, pair["noun_phrase"])
            if any(len(o["out_obj_ids"]) for o in ref.values()):
                chosen = (name, pair["noun_phrase"], folder)
                break
        if chosen:
            break
    assert chosen, "no smoke pair produced objects in its first frames"
    h = PipelineHandle(S["pred"])
    try:
        out = _session(chosen[2], chosen[1])
    finally:
        h.uninstall()
    assert _same(ref, out), "outputs differ with the adapter installed"
    n_obj = [len(ref[f]["out_obj_ids"]) for f in sorted(ref)]
    return dict(video=chosen[0], noun_phrase=chosen[1], frames=len(ref), objects_per_frame=n_obj,
                mask_pixels_total=int(sum(o["out_binary_masks"].sum() for o in ref.values())), identical=True)


if __name__ == "__main__":
    from tests.eventful._runner import run

    torch.set_grad_enabled(False)
    run(globals(), "test_pipeline_adapter")
