# Vendored SAM 3 / SAM 3.1

This directory is a vendored copy of https://github.com/facebookresearch/sam3 (no `.git`).
All four methods in this repository run on this copy of SAM 3.1.

| | |
|---|---|
| Upstream base | `2345a4a` "Evict streamed SAM3 frame outputs" (Maxime Causeret, 2026-09-18) |
| Local patch   | `fa34c5b` "fix(sam3_base_predictor): filter init_state kwargs via inspect in start_session()" (kuihao), cherry-picked on branch `sam31-fix-multiplex-start-session` |
| Local edit    | `examples/saco_gold_silver_eval_example.ipynb` (uncommitted change at vendoring time) |
| Removed       | `assets/*.gif` (git-ignored), `sam3.egg-info/` |

The original clone's git history is kept outside the repo at `Project_SYSDL/sam3.git.bak`.

Install (editable, from the repo root): `pip install -e third_party/sam3`.
Treat this tree as read-only; record any further patch here and commit it on `main`.
