# Results

Saved metrics and result tables for all four methods. New runs go to `results/runs/`, which is git-ignored.

| Path | Contents |
|---|---|
| `compiled_results.csv` | Accuracy and timing per method, for all 20 videos and each subset |
| `per_video_results.csv` | The same measures for each video |
| `controlled_e2e.csv` | EP-ViT and dense run side by side for end-to-end timing |
| `dense/`, `epvit/`, `eventful/`, `eventful_fused/` | `metrics.json` (evaluator output) and `summary.json` for `normal` and `high_motion` |
| `sample_demo/` | Comparison video and results table for the sample clip |
