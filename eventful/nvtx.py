"""Labels used inside our own modules (e.g. `rng("blk3.mlp.score")` in GatedOp). A no-op unless `enable()` was
called. Like eventful/annotate.py, a label can be emitted as an NVTX range (seen by Nsight Systems), as a
torch.profiler record_function (seen by torch.profiler), or both, so that both tools see the same names.

When labels are off, `rng()` returns one shared null context, so a disabled label costs a function call and a
global lookup, no generator or allocation (the gated trunk issues ~10 labelled steps per gate per frame, and its
CPU issue time is a measured quantity). `counts` records how often each label was opened while enabled.
"""
from contextlib import contextmanager, nullcontext

import torch
from torch.autograd.profiler import record_function

_NVTX = False
_RF = False
_NULL = nullcontext()
counts = {}


def enable(on=True, record_functions=False):
    """enable(True) -> NVTX ranges; enable(True, record_functions=True) -> NVTX and record_function;
    enable(False, record_functions=True) -> record_function only; enable(False) -> nothing."""
    global _NVTX, _RF
    _NVTX, _RF = bool(on), bool(record_functions)


def active():
    return _NVTX or _RF


def rng(name):
    if not (_NVTX or _RF):
        return _NULL
    return _range(name)


@contextmanager
def _range(name):
    counts[name] = counts.get(name, 0) + 1
    if _NVTX:
        torch.cuda.nvtx.range_push(name)
    rf = record_function(name) if _RF else None
    if rf is not None:
        rf.__enter__()
    try:
        yield
    finally:
        if rf is not None:
            rf.__exit__(None, None, None)
        if _NVTX:
            torch.cuda.nvtx.range_pop()
