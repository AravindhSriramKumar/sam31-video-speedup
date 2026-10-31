"""Measurement-only labels for profilers. Nothing here changes what the model computes.

Every label is emitted in two forms at once, so that both profilers see exactly the same names:
  * an NVTX range (torch.cuda.nvtx.range_push / range_pop), which Nsight Systems records;
  * a torch.profiler record_function, which torch.profiler records as a "user_annotation".
Either form can be switched off (e.g. NVTX only when running under nsys, to keep overhead minimal).

Ways to attach a label:
  module(mod, label)          forward pre-hook opens the range, forward hook closes it
  method(obj, name, label)    wraps a bound method on that instance (instance attribute; the class is untouched).
                              Use it instead of `module` when the caller invokes `mod.forward(...)` directly, which
                              bypasses nn.Module hooks (SAM does this for the neck).
  function(module, name, label)  wraps a module-level function, e.g. sam3.model.vitdet.apply_rotary_enc_real
  range(label)                context manager, for code we call ourselves (e.g. one frame)
`remove()` restores everything.

A note on what a label covers: a range is open on the CPU while the Python code inside it runs. GPU work is
asynchronous, so the kernels that range *launches* may execute after it has closed. The trace analysis
(the trace analysis) therefore attributes each kernel to the range that was open when the kernel was
launched, not to the range open while it ran.
"""
from contextlib import contextmanager

import torch
from torch.autograd.profiler import record_function


class Annotator:
    def __init__(self, nvtx=True, record_functions=True):
        self.nvtx, self.rf = nvtx, record_functions
        self._hooks, self._restore, self._stack = [], [], []
        self.attached, self.counts = [], {}     # labels attached, and how often each was opened (a label that is
                                                # attached but never opened points at code that bypasses it)

    # ------------------------------------------------------------------ primitives
    def _open(self, label):
        self.counts[label] = self.counts.get(label, 0) + 1
        if self.nvtx:
            torch.cuda.nvtx.range_push(label)
        rf = None
        if self.rf:
            rf = record_function(label)
            rf.__enter__()
        self._stack.append(rf)

    def _close(self):
        rf = self._stack.pop()
        if rf is not None:
            rf.__exit__(None, None, None)
        if self.nvtx:
            torch.cuda.nvtx.range_pop()

    @contextmanager
    def range(self, label):
        self._open(label)
        try:
            yield
        finally:
            self._close()

    # ------------------------------------------------------------------ attachers
    def module(self, mod, label):
        self.attached.append(label)
        self._hooks.append(mod.register_forward_pre_hook(lambda m, a: self._open(label)))
        self._hooks.append(mod.register_forward_hook(lambda m, a, o: self._close()))

    def method(self, obj, name, label):
        self.attached.append(label)
        orig = getattr(obj, name)
        had_own = name in obj.__dict__          # SAM itself sometimes sets instance-level wrappers (e.g. forward)
        prev = obj.__dict__.get(name)

        def wrapped(*args, **kwargs):
            with self.range(label):
                return orig(*args, **kwargs)

        setattr(obj, name, wrapped)
        self._restore.append(lambda: setattr(obj, name, prev) if had_own else obj.__dict__.pop(name, None))

    def function(self, module, name, label):
        self.attached.append(label)
        orig = getattr(module, name)

        def wrapped(*args, **kwargs):
            with self.range(label):
                return orig(*args, **kwargs)

        setattr(module, name, wrapped)
        self._restore.append(lambda: setattr(module, name, orig))

    def never_opened(self):
        """Attached labels that were never opened."""
        return sorted({l for l in self.attached if not self.counts.get(l)})

    def remove(self):
        for h in self._hooks:
            h.remove()
        for r in reversed(self._restore):
            r()
        self._hooks, self._restore = [], []


# ---------------------------------------------------------------------- standard label sets
TRUNK_BLOCK_PARTS = (("norm1", "norm1"), ("attn.qkv", "qkv"), ("attn.proj", "proj"), ("attn", "attn"),
                     ("norm2", "norm2"), ("mlp", "mlp"))


def annotate_trunk(ann, vit, blocks=True, parts=True, rope=True, sdpa=True):
    """Labels inside the ViT trunk.

    trunk                      the whole ViT forward
    blk{i}                     one block (its exclusive time = window partition/unpartition copies + residual adds)
    blk{i}.norm1 / .qkv / .attn / .proj / .norm2 / .mlp   the block's submodules (attn's exclusive time = head
                               reshapes; rope and sdpa are labelled separately inside it)
    rope, sdpa                 the RoPE function and scaled-dot-product attention calls
    Do not attach block-level labels to a trunk that is torch.compile'd (hooks inside a compiled region).
    """
    ann.module(vit, "trunk")
    eventful = any(hasattr(b, "blk") for b in vit.blocks)
    if eventful:
        # EventfulBlock re-states Attention.forward, so hooks on attn / qkv / proj / mlp would not describe it. It
        # labels blk{i}.qkv / .attn / .proj / .mlp itself, and each GatedOp labels its steps (blk{i}.mlp.score,
        # .select, .gather, .op, .scatter, .flush), through eventful/nvtx.py: switch those on in the same form.
        from eventful import nvtx
        nvtx.enable(ann.nvtx, record_functions=ann.rf)
        ann._restore.append(lambda: nvtx.enable(False))
    if blocks:
        for i, blk in enumerate(vit.blocks):
            if not eventful:                    # EventfulBlock opens its own blk{i} range (eventful/block.py)
                ann.module(blk, f"blk{i}")
            if eventful:
                if parts:
                    ann.module(blk.blk.norm1, f"blk{i}.norm1")
                    ann.module(blk.blk.norm2, f"blk{i}.norm2")
                continue
            if parts:
                for attr, short in TRUNK_BLOCK_PARTS:
                    mod = blk
                    for a in attr.split("."):
                        mod = getattr(mod, a)
                    ann.module(mod, f"blk{i}.{short}")
    if rope:
        import sam3.model.vitdet as vitdet
        ann.function(vitdet, "apply_rotary_enc_real", "rope")
    if sdpa:
        import sam3.model.vitdet as vitdet
        ann.function(vitdet.F, "scaled_dot_product_attention", "sdpa")


def annotate_pipeline(ann, predictor, vit, trunk_module=None):
    """Labels for one frame of the SAM 3.1 pipeline (names in parentheses are SAM methods / modules).

    step            one frame's model step                       (model._det_track_one_frame)
      detect        backbone + detection                         (model.run_backbone_and_detection)
        grounding   detector on the frame's chunk                (detector.forward_grounding)
          neck      tri-neck incl. trunk                         (Sam3TriViTDetNeck module)
            trunk   ViT trunk                                    (ViT module)
          det.encoder / det.decoder / det.seg_head               (detector._run_encoder / _run_decoder /
                                                                  _run_segmentation_heads)
      track         tracker propagation                          (model.run_tracker_propagation)
      plan          association, hotstart, heuristics            (model.run_tracker_update_planning_phase)
        memory_encoder                                           (model._tracker_update_memories)
      execute       apply the update plan                        (model.run_tracker_update_execution_phase)
      outputs       build per-object outputs                     (model.build_outputs)
    postprocess     output post-processing (outside the step)    (model._postprocess_output)
    """
    m = predictor.model
    det = m.detector
    ann.method(m, "_det_track_one_frame", "step")
    ann.method(m, "run_backbone_and_detection", "detect")
    ann.method(det, "forward_grounding", "grounding")
    t = trunk_module if trunk_module is not None else vit
    necks = [mod for mod in m.modules() if getattr(mod, "trunk", None) is t]
    assert len(necks) == 1, f"expected one module holding the trunk, found {len(necks)}"
    ann.method(necks[0], "forward", "neck")     # SAM calls vision_backbone.forward(...) directly: hooks would not fire
    # With the Eventful frame-order adapter installed (neck.trunk = adapter), "trunk" labels the adapter: it is eager
    # code, so the label stays valid when the ViT inside it is torch.compile'd (hooks there would be traced away).
    ann.module(t, "trunk")
    ann.method(det, "_run_encoder", "det.encoder")
    ann.method(det, "_run_decoder", "det.decoder")
    ann.method(det, "_run_segmentation_heads", "det.seg_head")
    ann.method(m, "run_tracker_propagation", "track")
    ann.method(m, "run_tracker_update_planning_phase", "plan")
    ann.method(m, "_tracker_update_memories", "memory_encoder")
    ann.method(m, "run_tracker_update_execution_phase", "execute")
    ann.method(m, "build_outputs", "outputs")
    ann.method(m, "_postprocess_output", "postprocess")
