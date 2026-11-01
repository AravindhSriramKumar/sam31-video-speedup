# Experiments

Runners for the dense, EP-ViT and Eventful runs and the script that scores and compares them. Each runner takes `--sample` for the included clip or `--subset` for the full evaluation.

| File | Contents |
|---|---|
| `run_dense.py` | Dense SAM 3.1 baseline (B2d) |
| `run_epvit.py` | EP-ViT |
| `run_eventful_three_gates.py` | Eventful at its best configuration: gates on qkv, proj and the MLP |
| `run_eventful.py` | Eventful with the MLP gate only |
| `run_eventful_fused.py` | Eventful with the MLP gate only and the fused kernel |
| `analyze_results.py` | Scores the runs, or checks the committed tables with `--committed` |
| `epvit_compiled.py` | EP-ViT under `torch.compile` |
| `harness/` | Shared runtime setup, timing, sessions and scoring |
