# Eventful

Eventful token gating for the SAM 3.1 ViT trunk. A gate around each block's MLP recomputes only the tokens that changed and reuses cached outputs for the rest. SAM's own files are not edited.

| File | Contents |
|---|---|
| `gates.py` | One gate and its cache around a per-token operation |
| `block.py` | Drop-in wrapper around a SAM 3.1 ViT block |
| `controller.py` | Swaps the trunk's blocks and sets K for each frame |
| `config.py` | Gating configuration and the spec strings |
| `pipeline.py` | Runs the trunk in frame order inside the full pipeline |
| `baseline.py` | Baseline fixes shared with dense, as named presets |
| `kernels/` | Change score, top-K selection and the fused kernel wrapper |
| `env.py`, `frames.py` | Runtime setup and clips for the trunk harness |
| `annotate.py`, `nvtx.py`, `instrument.py`, `timing.py`, `traceutil.py`, `stats.py`, `metrics.py` | Profiling labels, timing, statistics and feature-error metrics |
