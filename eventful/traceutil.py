"""Shared helpers for profiler traces (used by the profiling tools)."""


def kernel_family(name):
    """Coarse family of a GPU activity, decided from its name. One rule for every tool, so counts are comparable."""
    n = name.lower()
    if name.startswith("## "):   # torch.compile's "## Call CompiledFxGraph ... ##" GPU-timeline annotation, not a kernel
        return "annotation"
    if "memcpy" in n:
        return "memcpy"
    if "memset" in n:
        return "memset"
    if "flash" in n or "fmha" in n or "attention" in n:
        return "attention"
    if "gemm" in n or "cutlass" in n or "cublas" in n or "sm80_xmma" in n:
        return "gemm"
    if "layer_norm" in n:
        return "layernorm"
    if "triton" in n:
        return "triton (compiled/fused)"
    if "conv" in n or "cudnn" in n:
        return "conv/cudnn"
    if "elementwise" in n or "copy" in n or "reduce" in n or "cat" in n or "index" in n:
        return "elementwise/copy"
    return "other"
