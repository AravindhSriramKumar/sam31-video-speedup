#!/usr/bin/env bash
# SA-Co/VEval evaluation data: annotations and the 24 fps frames of the 20 protocol videos -> data/saco_sav/.
# Safe to re-run: finished steps and videos are skipped, an interrupted video resumes from its part file.
#
# Needs approved access to the gated Hugging Face dataset facebook/SACo-VEval (`hf auth login` first).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GT=data/saco_sav/annotation/saco_veval_sav_val.json
WANT=$(python -c "import json; print(json.load(open('data/eval_pairs.json'))['annotation_sha256'])")

if [ -f "$GT" ] && [ "$(sha256sum "$GT" | cut -d' ' -f1)" = "$WANT" ]; then
    echo "annotations already present"
else
    python data/saco_data.py --download-annotations || {
        echo "annotation download failed: request access at https://huggingface.co/datasets/facebook/SACo-VEval," \
             "then run hf auth login" >&2
        exit 1
    }
fi

python data/prepare_saco_eval.py --fetch
