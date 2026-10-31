"""CUTLASS kernels for the MLP gate's two GEMMs (kernels/fused_gemm/fused_gemm.cu), as compiled-graph custom ops.

    fused_fc1(W1, b1, x, ii, h, gather, shape)      h = gelu(x[ii] @ W1^T + b1) if gather, else gelu(x @ W1^T + b1)
    fused_fc2(W2, b2, h, d, ii, scatter, shape)     d[ii] = h @ W2^T + b2       if scatter, else d = h @ W2^T + b2

Reading or writing rows by index is a switch on the stock CUTLASS GEMM. Tokens are columns of B and D, as in the kernels
cuBLAS picks for the same call. torch.compile sees each kernel as one sealed op, launched on the current stream, so a CUDA
graph records it like any other launch. Indices are int32.
"""
import ctypes
from pathlib import Path

import torch
from torch.library import custom_op

_LIB = Path(__file__).resolve().parents[2] / "kernels/fused_gemm/libfused_gemm.so"
_lib = ctypes.CDLL(str(_LIB))
for _f in (_lib.fc1_gathered, _lib.fc2_scattered, _lib.fc1_plain, _lib.fc2_plain):
    _f.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]

# fastest tile per K, measured on real rows (results/eventful_fused/stage2_run1.json): 0 = 256x128 x3, 2 = 128x128 x3,
# 3 = 128x128 x5. fc1 uses 128x128 x3 at every K.
FC1_SHAPE = 2
FC2_SHAPE = {512: 3, 1024: 3, 1536: 0, 2048: 0}


def fc2_shape(k):
    return FC2_SHAPE.get(k, 2)


def _launch(f, shape, A, B, C, D, ii, K, H, Cw):
    r = f(shape, A.data_ptr(), B.data_ptr(), C.data_ptr(), D.data_ptr(), ii.data_ptr(), K, H, Cw, torch.cuda.current_stream().cuda_stream)
    if r != 0:
        raise RuntimeError(f"fused GEMM launch failed with status {r}")


@custom_op("eventful::fused_fc1", mutates_args={"h"})
def fused_fc1(W: torch.Tensor, b: torch.Tensor, x: torch.Tensor, ii: torch.Tensor, h: torch.Tensor, gather: bool, shape: int) -> None:
    _launch(_lib.fc1_gathered if gather else _lib.fc1_plain, shape, W, x, b, h, ii, h.shape[0], W.shape[0], W.shape[1])


@fused_fc1.register_fake
def _fc1_fake(W, b, x, ii, h, gather, shape):
    return None


@custom_op("eventful::fused_fc2", mutates_args={"d"})
def fused_fc2(W: torch.Tensor, b: torch.Tensor, h: torch.Tensor, d: torch.Tensor, ii: torch.Tensor, scatter: bool, shape: int) -> None:
    _launch(_lib.fc2_scattered if scatter else _lib.fc2_plain, shape, W, h, b, d, ii, h.shape[0], W.shape[1], W.shape[0])


@fused_fc2.register_fake
def _fc2_fake(W, b, h, d, ii, scatter, shape):
    return None
