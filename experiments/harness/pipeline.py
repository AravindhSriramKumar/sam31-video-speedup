"""Running the trunk inside the full SAM 3.1 pipeline with frame-order awareness.

Dense B2d runs through SequentialTrunkAdapter with controller=None, so it encodes one frame per call. The `controller`
argument is the plug-in point for a stateful trunk.

Problem: Eventful state is a function of the *previous frame*, but the pipeline does not call the trunk in
temporal order. In one session the trunk sees frames 0, 1 (add_prompt encodes frame 0 and prefetches frame 1
with batched grounding switched off), then 0, 1, 2, ... again during propagation (one frame per call at
grounding batch 1, chunks of 8 at batch 8).

Solution, in two small pieces:
  * FrameIndexTap  - an instance-level wrapper on `detector._get_img_feats` (sam3_image.py:116). That is the
                     only place that turns frame indices into trunk input. When it is about to encode, it
                     records which frame indices (in the order they are stacked) the next trunk call holds.
  * SequentialTrunkAdapter - installed as `neck.trunk`. Runs the batch one frame at a time and calls
                     `controller.reset()` (re-prime: next frame is dense) whenever a frame is not the
                     successor of the last frame it saw.

Both are removable (`PipelineHandle.uninstall()`); SAM's files are not touched.
"""
from collections import deque

import torch
import torch.nn as nn


def find_vit(model):
    """The single sam3.model.vitdet.ViT inside `model` (the predictor deletes the tracker's own backbone)."""
    from sam3.model.vitdet import ViT

    found = [m for m in model.modules() if isinstance(m, ViT)]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one ViT, found {len(found)}")
    return found[0]


class FrameIndexTap:
    def __init__(self, detector):
        self.detector = detector
        self._orig = detector._get_img_feats          # bound method of the instance
        self.pending = deque()
        detector._get_img_feats = self._wrapped         # instance attribute shadows the class method

    def _wrapped(self, backbone_out, img_ids):
        if "backbone_fpn" not in backbone_out:          # the call will run the backbone on these frames
            # Same order SAM uses to stack the images: torch.unique (sorted) when several, else the single id.
            # SAM syncs here itself (sam3_image.py:141/150/152), so .tolist() adds no new stall.
            ids = torch.unique(img_ids).tolist() if img_ids.numel() > 1 else [int(img_ids.reshape(-1)[0])]
            self.pending.append(ids)
        return self._orig(backbone_out, img_ids)

    def pop_ids(self, n):
        if self.pending and len(self.pending[0]) == n:
            return self.pending.popleft()
        self.pending.clear()                            # out of step: caller treats frames as unknown
        return None

    def uninstall(self):
        if self.detector.__dict__.get("_get_img_feats") is not None:
            del self.detector._get_img_feats


class SequentialTrunkAdapter(nn.Module):
    """Installed as neck.trunk. Splits a (B,3,1008,1008) chunk into frames and runs them in temporal order."""

    def __init__(self, vit, controller=None, tap=None, trunk=None):
        super().__init__()
        self.vit = vit
        # callable for dense frames: the ViT itself, or a torch.compile'd ViT (plain attribute, not a submodule)
        self.__dict__["dense_trunk"] = trunk if trunk is not None else vit
        self.controller = controller
        self.tap = tap
        self.new_session()

    @property
    def channel_list(self):                             # read by the neck at construction; kept for safety
        return self.vit.channel_list

    def new_session(self):
        self.last_id = None
        self.encodes = 0
        self.reprimes = 0
        self.trace = []                                 # (frame_id or None, reprimed) per encoded frame

    def forward(self, x):
        x = getattr(x, "tensors", x)                    # accept NestedTensor like ViT.forward does
        ids = self.tap.pop_ids(x.shape[0]) if self.tap is not None else None
        outs = []
        for i in range(x.shape[0]):
            fid = None if ids is None else ids[i]
            reprime = fid is None or self.last_id is None or fid != self.last_id + 1
            if reprime:
                if self.controller is not None:
                    self.controller.reset()
                self.reprimes += 1
            outs.append(self.controller.run(x[i:i + 1]) if self.controller is not None else self.dense_trunk(x[i:i + 1]))
            self.last_id = fid
            self.encodes += 1
            self.trace.append((fid, reprime))
        return [torch.cat(o, dim=0) for o in zip(*outs)]


class PipelineHandle:
    """Install the adapter into a built predictor; uninstall restores the original objects."""

    def __init__(self, predictor, controller=None, trunk=None):
        self.predictor = predictor
        self.vit = find_vit(predictor.model)
        necks = [m for m in predictor.model.modules() if getattr(m, "trunk", None) is self.vit]
        if len(necks) != 1:
            raise RuntimeError(f"expected one neck owning the ViT, found {len(necks)}")
        self.neck = necks[0]
        self.tap = FrameIndexTap(predictor.model.detector)
        self.adapter = SequentialTrunkAdapter(self.vit, controller, self.tap, trunk=trunk)
        self.neck.trunk = self.adapter
        self.sessions = []
        orig = predictor.handle_request

        def handle_request(request, _orig=orig):
            if request.get("type") in ("start_session", "reset_session", "close_session"):
                self._end_session()
            return _orig(request)

        predictor.handle_request = handle_request       # instance attribute; removed in uninstall()

    def _end_session(self):
        a = self.adapter
        if a.encodes:
            self.sessions.append(dict(encodes=a.encodes, reprimes=a.reprimes, trace=a.trace))
        a.new_session()
        if a.controller is not None:
            a.controller.reset()
        self.tap.pending.clear()

    def last_session(self):
        self._end_session()
        return self.sessions[-1] if self.sessions else None

    def uninstall(self):
        self.neck.trunk = self.vit
        self.tap.uninstall()
        self.predictor.__dict__.pop("handle_request", None)
