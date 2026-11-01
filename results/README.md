# Results

Saved metrics and result tables for all five runs. New runs go to `results/runs/`, which is git-ignored.

| Path | Contents |
|---|---|
| `compiled_results.csv` | Accuracy and timing per method, for all 20 videos and each subset |
| `per_video_results.csv` | The same measures for each video |
| `controlled_e2e.csv` | EP-ViT and dense run side by side for end-to-end timing |
| `dense/`, `epvit/`, `eventful/`, `eventful_fused/`, `eventful_three_gates/` | `metrics.json` (evaluator output) and `summary.json` for `normal` and `high_motion` |
| `eventful_three_gates/bootstrap_*.md` | Paired bootstrap of three gates against the MLP gate only |
| `sample_demo/` | Comparison video and results table for the sample clip |

`summary.json` and `metrics.json` under `eventful_three_gates/` cover every pair that completed (72 `normal`, 124 `high_motion`). The tables use the 70 and 119 frozen pairs, so their values differ.
