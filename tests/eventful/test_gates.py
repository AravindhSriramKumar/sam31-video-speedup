"""The gate mechanics are exact.

1. K alignment: requests >= N give exactly N; others round half-up to a multiple of 128.
2. Score kernel: Triton absolute / relative L2 equals the torch reference (rtol 1e-5) on random and real
   activations, fp32 and bf16 inputs.
3. GatedOp bookkeeping, on real MLP inputs of two consecutive frames: the K selected rows of the output equal
   op(c[selected]) (same kernels, M = K) bit for bit; every other row equals the previous output; ref equals c on
   the selected rows and the old ref elsewhere; the selected rows are exactly the K largest scores.
4. K = N: the gated trunk (MLP + qkv + proj gates) equals the dense trunk bit for bit on every frame of a real clip.
5. Frame 0 after reset(): bitwise equal to the dense trunk, at K = 1280.
6. Static video (one frame repeated 12 times): every later output equals frame 0 within the "different kernels"
   rule, for every gate set at K = 1280 (all scores are 0; recomputed rows may use another GEMM kernel, M = K).
7. Last-sent semantics (threshold policy, tiny linear op): a token drifting by 0.1 per frame is not selected at
   threshold 0.15 on frame 1 and is selected on frame 2.
8. No CPU-GPU synchronisation in steady state (top-K): torch.cuda.set_sync_debug_mode("error") raises nothing.
9. reset() between clips: clip A then clip B equals clip B alone, bit for bit.
10. refresh_every = 3: frames 3, 6, 9 equal the dense trunk bit for bit; frames in between do not have to.
11. Compiled path: K = N compiled gated trunk within the compiled-vs-eager envelope of dense on every frame; no
    per-frame recompilation; compiled vs eager gated error at K = 2560 recorded.

    python tests/eventful/test_gates.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from eventful import env, frames, metrics, tolerances  # noqa: E402
from eventful.baseline import PRESETS, apply_baseline  # noqa: E402
from eventful.config import EventfulConfig, resolve_k_schedule  # noqa: E402
from eventful.controller import EventfulController  # noqa: E402
from eventful.gates import GatedOp  # noqa: E402
from eventful.kernels.gate_score import gate_score  # noqa: E402

S = {}
ALL3 = dict(gate_mlp=True, gate_qkv=True, gate_proj=True)


def _setup():
    if S:
        return
    env.setup_runtime()
    torch.inference_mode().__enter__()
    S["vit"] = env.load_trunk()
    S["trunk"], _ = apply_baseline(S["vit"], PRESETS["B1"])
    a, _ = frames.load_clip("sav_012075@24fps", max_frames=10)
    b, _ = frames.load_clip("sav_018743@24fps", max_frames=4)
    S["a"], S["b"] = frames.to_trunk_input(a), frames.to_trunk_input(b)


def _dense(x):
    return [S["trunk"](x[i:i + 1])[-1].clone() for i in range(x.shape[0])]


def _gated(cfg, x):
    ctl = EventfulController(S["vit"], cfg)
    try:
        return [ctl.run(x[i:i + 1])[-1].clone() for i in range(x.shape[0])], ctl
    finally:
        ctl.restore()


def _mlp_inputs(x, block):
    """Real MLP-gate inputs (norm2 output rows) of `block` for each frame of x, from a dense run."""
    caps = []
    blk = S["vit"].blocks[block]
    h = blk.norm2.register_forward_hook(lambda m, a, o: caps.append(o.reshape(-1, o.shape[-1]).clone()))
    _dense(x)
    h.remove()
    return caps


def test_1_k_alignment():
    cfg = lambda s: resolve_k_schedule(EventfulConfig(k_schedule=s))["mlp"]  # noqa: E731
    got = dict(full=cfg("uniform:5184")[0], over=cfg("uniform:9999")[0], k1280=cfg("uniform:1280")[0],
               k1300=cfg("uniform:1300")[0], k1344=cfg("uniform:1344")[0], k10=cfg("uniform:10")[0],
               linear=cfg("linear:1280:3840")[::31])
    assert got["full"] == 5184 and got["over"] == 5184 and got["k1280"] == 1280, got
    assert got["k1300"] == 1280 and got["k1344"] == 1408 and got["k10"] == 128, got   # 1344/128 = 10.5 -> 11
    assert got["linear"] == [1280, 3840], got
    from eventful.config import parse_spec
    base, c, mode = parse_spec("B1:mlp:k2560:b24-31:c")
    got["spec_blocks"] = [c.gated_block(i) for i in (0, 23, 24, 31)]
    assert (base, mode, c.blocks) == ("B1", "compile", tuple(range(24, 32))) and got["spec_blocks"] == [False, False, True, True]
    return got


def test_2_score_kernel():
    _setup()
    real = _mlp_inputs(S["a"][:2], 7)
    cases = dict(random=(torch.randn(5184, 1024, device="cuda"), torch.randn(5184, 1024, device="cuda")),
                 real=(real[1], real[0]))
    out = {}
    for name, (c, r) in cases.items():
        for dt in (torch.float32, torch.bfloat16):
            for rel in (False, True):
                cc, rr = c.to(dt).contiguous(), r.to(dt).contiguous()
                t, ref = gate_score(cc, rr, "triton", rel), gate_score(cc, rr, "torch", rel)
                err = ((t - ref).abs() / ref.abs().clamp_min(1e-12)).max().item()
                out[f"{name}_{str(dt)[6:]}_{'rel' if rel else 'abs'}"] = err
                assert torch.allclose(t, ref, rtol=1e-5, atol=1e-6), (name, dt, rel, err)
    return dict(max_rel_diff=out)


def test_3_gatedop_bookkeeping():
    _setup()
    c0, c1 = _mlp_inputs(S["a"][:2], 7)
    op = S["vit"].blocks[7].mlp
    g = GatedOp(op, "t.mlp", k=1280)
    out0 = g(c0).clone()                      # flush
    ref0 = g.ref.clone()
    scores = gate_score(c1, g.ref)
    out1 = g(c1)
    idx = scores.topk(1280, sorted=False).indices
    sel = torch.zeros(5184, dtype=torch.bool, device="cuda")
    sel[idx] = True
    y_sel = op(c1.index_select(0, idx))
    checks = dict(
        selected_rows_equal_op=bool(torch.equal(out1[idx], y_sel)),
        unselected_rows_equal_previous=bool(torch.equal(out1[~sel], out0[~sel])),
        ref_selected_equal_input=bool(torch.equal(g.ref[sel], c1[sel])),
        ref_unselected_unchanged=bool(torch.equal(g.ref[~sel], ref0[~sel])),
        selected_are_top_k=bool(scores[sel].min() >= scores[~sel].max()),
        flush_equals_dense_op=bool(torch.equal(out0, op(c0))),
    )
    assert all(checks.values()), checks
    return checks


def test_4_k_equals_n_bitwise():
    _setup()
    dense = _dense(S["a"])
    gated, _ = _gated(EventfulConfig(**ALL3, k_schedule="uniform:5184"), S["a"])
    eq = [bool(torch.equal(g, d)) for g, d in zip(gated, dense)]
    assert all(eq), eq
    return dict(frames=len(eq), bitwise_equal=eq)


def test_5_frame0_after_reset_bitwise():
    _setup()
    d0 = _dense(S["a"][:1])[0]
    gated, _ = _gated(EventfulConfig(**ALL3, k_schedule="uniform:1280"), S["a"][:1])
    assert torch.equal(gated[0], d0)
    return dict(bitwise_equal=True)


def test_6_static_video():
    _setup()
    x = S["a"][3:4].expand(12, -1, -1, -1).contiguous()
    lim = tolerances.different_kernels_limit()
    res = {}
    for name, kw in (("mlp", {}), ("mlp+qkv", dict(gate_qkv=True)), ("mlp+qkv+proj", dict(gate_qkv=True, gate_proj=True))):
        gated, _ = _gated(EventfulConfig(k_schedule="uniform:1280", **kw), x)
        st = [metrics.feature_stats(g, gated[0]) for g in gated[1:]]
        ok = [tolerances.check_different_kernels(s)[0] for s in st]
        res[name] = dict(max_rel_err=max(s["rel_err"] for s in st), min_cos_median=min(s["cos_median"] for s in st),
                         all_within_rule=all(ok), bitwise_equal_frames=sum(s["bitwise_equal"] for s in st))
        assert all(ok), (name, res[name])
    return dict(limit_rel_err=lim[0], limit_cos_median=lim[1], per_gate_set=res)


def test_7_last_sent_semantics():
    lin = torch.nn.Linear(4, 4, device="cuda")
    g = GatedOp(lin, "t.lin", k=1, policy="threshold", threshold=0.15, score="torch")
    base = torch.zeros(2, 4, device="cuda")
    step = torch.tensor([[0.1, 0, 0, 0], [0, 0, 0, 0]], device="cuda")
    g(base)                                         # flush: ref = base
    sel = []
    for f in (1, 2):
        c = base + f * step
        scores = gate_score(c, g.ref, "torch")
        sel.append((scores > 0.15).nonzero().squeeze(1).tolist())
        g(c)
    assert sel == [[], [0]], sel
    return dict(selected_per_frame=sel)


def test_8_no_sync_in_steady_state():
    _setup()
    ctl = EventfulController(S["vit"], EventfulConfig(**ALL3, k_schedule="uniform:1280"))
    try:
        ctl.run(S["a"][0:1])                        # flush frame (eager, allowed to sync)
        ctl.run(S["a"][1:2])
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            ctl.run(S["a"][2:3])
            ctl.run(S["a"][3:4])
        finally:
            torch.cuda.set_sync_debug_mode(0)
    finally:
        ctl.restore()
    return dict(steady_frames_without_sync=2)


def test_9_reset_between_clips():
    _setup()
    cfg = EventfulConfig(**ALL3, k_schedule="uniform:1280")
    ctl = EventfulController(S["vit"], cfg)
    try:
        for i in range(4):
            ctl.run(S["a"][i:i + 1])
        ctl.reset()
        ab = [ctl.run(S["b"][i:i + 1])[-1].clone() for i in range(S["b"].shape[0])]
    finally:
        ctl.restore()
    b_alone, _ = _gated(cfg, S["b"])
    eq = [bool(torch.equal(x, y)) for x, y in zip(ab, b_alone)]
    assert all(eq), eq
    return dict(bitwise_equal=eq)


def test_10_refresh():
    _setup()
    dense = _dense(S["a"])
    gated, _ = _gated(EventfulConfig(**ALL3, k_schedule="uniform:1280", refresh_every=3), S["a"])
    eq = [bool(torch.equal(g, d)) for g, d in zip(gated, dense)]
    assert eq[0] and eq[3] and eq[6] and eq[9], eq
    return dict(bitwise_equal_per_frame=eq)


def test_11_compiled_path():
    """Compiled steady frames (controller.attach_compiled, mode "default"): with K = N the compiled gated trunk stays
    within the compiled-vs-eager envelope (eventful.tolerances: 1.5 x the stored B2 vs B1 difference) of
    the dense eager trunk on every frame, and torch.compile does not recompile from frame to frame (one compiled
    graph for all steady frames). Fused kernels round differently, so the eager "different kernels" rule does not
    apply to a compiled trunk."""
    _setup()
    import torch._dynamo.utils as du

    dense = _dense(S["a"][:6])
    res = {}
    for k in (5184, 2560):
        cfg = EventfulConfig(k_schedule=f"uniform:{k}")
        ctl = EventfulController(S["vit"], cfg)
        ctl.attach_compiled(torch.compile(S["vit"], mode="default", dynamic=False))
        try:
            before = dict(du.counters["stats"])
            outs = [ctl.run(S["a"][i:i + 1])[-1].clone() for i in range(6)]
            graphs = du.counters["stats"].get("unique_graphs", 0) - before.get("unique_graphs", 0)
        finally:
            ctl.restore()
        eager, _ = _gated(cfg, S["a"][:6])
        st = [metrics.feature_stats(o, d) for o, d in zip(outs, dense)]
        vs_eager = [metrics.feature_stats(o, e)["rel_err"] for o, e in zip(outs, eager)]
        res[k] = dict(rel_err_vs_dense=[x["rel_err"] for x in st], rel_err_vs_eager_gated=vs_eager,
                      new_graphs=graphs, within_rule_vs_dense=[tolerances.check_compiled_vs_eager(x)[0] for x in st],
                      cos_median_vs_dense=[x["cos_median"] for x in st])
    assert all(res[5184]["within_rule_vs_dense"]), res[5184]
    assert res[5184]["new_graphs"] <= 2 and res[2560]["new_graphs"] <= 2, res
    lim = tolerances.compiled_vs_eager_limit()
    return {str(k): v for k, v in res.items()} | dict(limit_rel_err=lim[0], limit_cos_median=lim[1], graph_breaks={str(k): v for k, v in du.counters["graph_break"].items()})


if __name__ == "__main__":
    from _runner import run

    run(globals(), "test_gates")
