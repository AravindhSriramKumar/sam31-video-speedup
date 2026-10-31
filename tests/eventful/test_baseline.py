"""Tests 1 and 2: the baseline fixes keep the math the same.

1. RoPE compiled vs stock: on real q, k (QKV of real norm1 outputs) for a windowed block (0) and a global block
   (7), both implementations are within 1 bf16 ulp of an fp32 reference, with the ulp taken at the norm of each
   rotated (real, imag) pair (an element-wise ulp is ill-defined where a*cos - b*sin cancels; it is recorded too).
2. bf16 linear weights: trunk output vs stock passes the "different kernels" rule (noise floor x 1.5).
3. The trunk-only loader (eventful.env.load_trunk) gives the same weights and the same output as the trunk
   inside the full predictor.

    python tests/eventful/test_baseline.py        (needs results/eventful/reference/noise_floor.json for test 2)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

import sam3.model.vitdet as vitdet  # noqa: E402
from eventful import env, frames, metrics, tolerances  # noqa: E402
from eventful.baseline import _ORIG_ROPE, cast_linear_weights_bf16, set_rope_mode  # noqa: E402
from eventful.controller import find_vit  # noqa: E402

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


def _real_qk(block_idx, frame):
    """q, k exactly as Attention.forward builds them (strided views of the qkv output) for one real frame."""
    vit = S["vit"]
    blk = vit.blocks[block_idx]
    cap = {}
    h = blk.norm1.register_forward_hook(lambda m, a, out: cap.__setitem__("n1", out.clone()))
    vit(S["x"][frame:frame + 1])
    h.remove()
    x = cap["n1"]
    if blk.window_size > 0:
        x, _ = vitdet.window_partition(x, blk.window_size)
    B, H, W, C = x.shape
    a = blk.attn
    qkv = a.qkv(x).reshape(B, H * W, 3, a.num_heads, -1)
    q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    return q, k, a.freqs_cis_real, a.freqs_cis_imag


def test_1_rope_compile_within_1ulp():
    _setup()
    compiled = torch.compile(_ORIG_ROPE, dynamic=False, fullgraph=True)
    out = {}
    for blk_idx, tag in ((0, "windowed"), (7, "global")):
        for frame in (0, 2):
            q, k, fr, fi = _real_qk(blk_idx, frame)
            rq, rk = _ORIG_ROPE(q.float(), k.float(), freqs_cis_real=fr, freqs_cis_imag=fi)  # fp32 reference
            sq, sk = _ORIG_ROPE(q, k, freqs_cis_real=fr, freqs_cis_imag=fi)
            cq, ck = compiled(q, k, freqs_cis_real=fr, freqs_cis_imag=fi)
            nq, nk = tolerances.pair_norm(rq), tolerances.pair_norm(rk)
            r = dict(shape=list(q.shape),
                     # asserted: error in ulps at the rotated pair's norm (see tolerances.within_ulps)
                     stock_q=tolerances.within_ulps(sq, rq, scale=nq), stock_k=tolerances.within_ulps(sk, rk, scale=nk),
                     compiled_q=tolerances.within_ulps(cq, rq, scale=nq),
                     compiled_k=tolerances.within_ulps(ck, rk, scale=nk),
                     # recorded only: error in ulps at each element's own magnitude
                     elementwise_ulps=dict(stock_q=tolerances.within_ulps(sq, rq), compiled_q=tolerances.within_ulps(cq, rq),
                                           stock_k=tolerances.within_ulps(sk, rk), compiled_k=tolerances.within_ulps(ck, rk)),
                     compiled_vs_stock_max_abs=max((cq.float() - sq.float()).abs().max().item(),
                                                   (ck.float() - sk.float()).abs().max().item()),
                     compiled_vs_stock_bitwise=bool(torch.equal(cq, sq) and torch.equal(ck, sk)))
            for key in ("stock_q", "stock_k", "compiled_q", "compiled_k"):
                assert r[key]["frac_within"] == 1.0, (tag, frame, key, r[key])
            out[f"{tag}_frame{frame}"] = r
    return out


def test_2_bf16_weights_within_noise_floor():
    _setup()
    set_rope_mode("stock")
    vit, x = S["vit"], S["x"]
    refs = [vit(x[f:f + 1])[0] for f in range(x.shape[0])]
    cast_linear_weights_bf16(vit)
    rows = []
    for f in range(x.shape[0]):
        st = metrics.feature_stats(vit(x[f:f + 1])[0], refs[f])
        ok, info = tolerances.check_different_kernels(st)
        rows.append(dict(frame=f, ok=ok, bitwise=st["bitwise_equal"], **info))
    assert all(r["ok"] for r in rows), rows
    return dict(frames=rows)


def test_3_trunk_loader_matches_predictor():
    """Slow (~2 min): builds the full predictor."""
    _setup()
    vit_alone = env.load_trunk()                      # fresh fp32 copy (S["vit"] was cast by test 2)
    pred = env.build_predictor()
    vit_pred = find_vit(pred.model)
    sa, sp = vit_alone.state_dict(), vit_pred.state_dict()
    assert sa.keys() == sp.keys()
    diff = [k for k in sa if not torch.equal(sa[k], sp[k].to(sa[k].device))]
    assert not diff, diff[:5]
    set_rope_mode("stock")
    x = S["x"][:1]
    out_a, out_p = vit_alone(x)[0], vit_pred(x)[0]
    assert torch.equal(out_a, out_p)
    return dict(tensors_compared=len(sa), output_bitwise_equal=True)


if __name__ == "__main__":
    from tests.eventful._runner import run

    run(globals(), "test_baseline")
