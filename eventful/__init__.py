"""Eventful token gating for the SAM 3.1 ViT trunk.

Module map (one idea per file):
  env.py         runtime setup (autocast/TF32 like the predictor), trunk/predictor loaders, provenance
  frames.py      clip loading for the trunk harness (6 fps annotated / 24 fps / demo clips)
  baseline.py    baseline fixes (RoPE compile, bf16 weights) and named presets B0, B1, B2, B3
  block.py       EventfulBlock: Block re-stated with qkv / proj / MLP hooks
  controller.py  find_vit, EventfulController: swaps vit.blocks, resets gate state
  pipeline.py    FrameIndexTap + SequentialTrunkAdapter: frame-order-aware trunk inside the full predictor
  instrument.py  measurement-only CUDA-event timing of pipeline steps and trunk calls
  timing.py      timing primitives;  metrics.py feature-error metrics;  test limits
  nvtx.py        optional labels inside our own modules (NVTX and/or torch.profiler record_function)
  annotate.py    measurement-only labels attached to SAM modules/methods from outside (same two forms)
  traceutil.py   kernel-name families shared by the trunk harness and the trace analysis
"""
from eventful import env as _env  # noqa: F401  (first: sets CUDA_DEVICE_ORDER and sys.path before anything else)
from eventful.baseline import PRESETS, BaselineConfig, apply_baseline
from eventful.controller import EventfulController, find_vit
from eventful.pipeline import PipelineHandle, SequentialTrunkAdapter

__all__ = ["PRESETS", "BaselineConfig", "apply_baseline", "EventfulController", "find_vit", "PipelineHandle",
           "SequentialTrunkAdapter"]
