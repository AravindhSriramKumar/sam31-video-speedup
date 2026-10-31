"""Measurement-only instrumentation of the full pipeline. It never changes what the model computes.

StepTimer records CUDA events around:
  * every per-frame model step: `model._det_track_one_frame` (sam3_multiplex_base.py:433), which runs the
    detector (and the trunk, when the frame's chunk is not cached yet), tracker propagation, planning,
    memory encoding and output building for one frame. It does NOT include output post-processing
    (`_postprocess_output`) or the generator's hotstart buffering.
  * every trunk call: forward pre-hook / hook on the ViT, or on `trunk_module` when given. Pass the
    SequentialTrunkAdapter when it is installed: with a torch.compile'd trunk (Eventful ':c'), hooks on the ViT are
    traced into the compiled graph and their event.record() calls never run. Timed at the adapter, a trunk call
    also includes the adapter's per-frame split and torch.cat of the outputs.
Events are resolved once per session (`take()`), so timing adds no syncs inside the run.
"""
import torch


class StepTimer:
    def __init__(self, predictor, vit, trunk_module=None):
        self.model = predictor.model
        self.phase = "unset"
        self._steps, self._trunk = [], []
        self._orig_step = self.model._det_track_one_frame

        def timed_step(*args, **kwargs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = self._orig_step(*args, **kwargs)
            e.record()
            self._steps.append((self.phase, kwargs.get("frame_idx"), s, e))
            return out

        self.model._det_track_one_frame = timed_step      # instance attribute; removed in remove()
        self._t_start = None
        m = trunk_module if trunk_module is not None else vit
        self._hooks = [m.register_forward_pre_hook(self._pre), m.register_forward_hook(self._post)]

    def _pre(self, module, args):
        x = getattr(args[0], "tensors", args[0])
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._t_start = (ev, int(x.shape[0]))

    def _post(self, module, args, out):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._trunk.append((self.phase, self._t_start[1], self._t_start[0], ev))

    def take(self):
        """Resolve and clear: dict(steps=[{phase, frame, ms}], trunk=[{phase, batch, ms}])."""
        torch.cuda.synchronize()
        steps = [dict(phase=p, frame=f, ms=s.elapsed_time(e)) for p, f, s, e in self._steps]
        trunk = [dict(phase=p, batch=b, ms=s.elapsed_time(e)) for p, b, s, e in self._trunk]
        self._steps, self._trunk = [], []
        return dict(steps=steps, trunk=trunk)

    def remove(self):
        self.model.__dict__.pop("_det_track_one_frame", None)
        for h in self._hooks:
            h.remove()
