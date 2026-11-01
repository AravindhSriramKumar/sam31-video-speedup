"""Eventful SAM 3.1 with the fused CUTLASS kernel for the MLP gate's GEMMs, on the 24 fps protocol or the included sample.

    sh kernels/fused_gemm/build.sh                       # once: needs the CUDA toolkit, git and network access
    CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_fused.py --sample
    CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_fused.py --subset normal
    CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_fused.py --subset high_motion

The MLP-only Eventful configuration of run_eventful.py, with fc1 reading and fc2 writing rows by index inside the kernel
(`fgfull`). Full runs write results/runs/eventful_fused/<subset>__<spec>__gb1_pp1__adapter__24fps/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths  # noqa: E402
from experiments import run_eventful  # noqa: E402

SPEC = run_eventful.SPEC.replace(":tk:cg", ":tk:fgfull:cg")

if __name__ == "__main__":
    if not (paths.PROJECT / "kernels/fused_gemm/libfused_gemm.so").exists():
        raise SystemExit("kernels/fused_gemm/libfused_gemm.so is missing: run sh kernels/fused_gemm/build.sh")
    run_eventful.main(SPEC, "eventful_fused")
