# Fused GEMM kernel

One CUTLASS kernel for the two GEMMs of the Eventful MLP gate. It reads and writes rows by index, replacing the separate gather and scatter steps.

| File | Contents |
|---|---|
| `fused_gemm.cu` | The kernel source |
| `build.sh` | Downloads CUTLASS v3.8.0 and builds `libfused_gemm.so` |

Only `experiments/run_eventful_fused.py` needs it. Its Python wrapper is `eventful/kernels/fused_gemm.py`.
