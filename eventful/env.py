"""Runtime setup and provenance shared by every eventful script and test.

Two jobs:
  1. Reproduce the numeric environment SAM 3.1 runs in: TF32 on, a process-wide bf16 autocast that is never
     exited (this is what `Sam3MultiplexVideoPredictor.__init__` does), no autograd.
  2. Record where every number came from (`provenance()`), so each result file says which commit, GPU,
     clocks and command produced it.
"""
import datetime
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

# Make nvidia-smi indices and CUDA indices agree (must be set before CUDA initialises).
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import paths  # noqa: E402

_AUTOCAST = None


def setup_runtime():
    """Mirror the predictor's process state. Safe to call more than once."""
    global _AUTOCAST
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    if _AUTOCAST is None:
        _AUTOCAST = torch.autocast("cuda", dtype=torch.bfloat16)
        _AUTOCAST.__enter__()


def load_trunk(device="cuda"):
    """The SAM 3.1 ViT trunk alone, built with the same factory and weights the predictor uses.

    The predictor builds its trunk with `_create_vit_backbone(compile_mode=None, use_fa3=False,
    use_rope_real=True)` (via `_create_multiplex_tri_backbone`) and loads the tensors under
    `detector.backbone.vision_backbone.trunk.`. Building only the trunk takes seconds instead of ~100 s.
    """
    from sam3.model_builder import _create_vit_backbone

    vit = _create_vit_backbone(compile_mode=None, use_fa3=False, use_rope_real=True)
    sd = torch.load(paths.CKPT, map_location="cpu", mmap=True, weights_only=True)
    sd = sd.get("model", sd)
    pre = "detector.backbone.vision_backbone.trunk."
    res = vit.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=False)
    # freqs_cis_real / freqs_cis_imag are derived RoPE buffers, not stored in the checkpoint
    assert not res.unexpected_keys and all("freqs_cis_" in k for k in res.missing_keys), res
    return vit.to(device).eval()


def build_predictor(grounding_batch=1, postprocess_batch=1, hotstart_delay=None):
    """Full SAM 3.1 multiplex predictor in the project's configuration."""
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    pred = build_sam3_multiplex_video_predictor(
        checkpoint_path=str(paths.CKPT), use_fa3=False, use_rope_real=True, compile=False,
        async_loading_frames=False)
    pred.model.batched_grounding_batch_size = grounding_batch
    pred.model.postprocess_batch_size = postprocess_batch
    if hotstart_delay is not None:
        pred.model.hotstart_delay = hotstart_delay
    return pred


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def physical_gpu():
    """nvidia-smi index of the GPU this process uses (CUDA_VISIBLE_DEVICES must name exactly one)."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return v.split(",")[0] if v else "0"


def gpu_state():
    idx = physical_gpu()
    q = ("name,uuid,driver_version,clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,power.draw,"
         "power.limit,utilization.gpu,memory.used,clocks_throttle_reasons.active")
    vals = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader", "-i", idx]).split(", ")
    state = dict(zip(q.split(","), vals))
    uuid = state.get("uuid", "")
    apps = [l for l in _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                             "--format=csv,noheader"]).splitlines() if uuid and l.startswith(uuid)]
    state["other_compute_processes"] = [a for a in apps if a.split(", ")[1] != str(os.getpid())]
    state["physical_index"] = idx
    return state


def provenance(extra=None):
    """Everything needed to trace a number back to the code and machine that produced it."""
    git = lambda *a: _run(["git", "-C", str(ROOT), *a])  # noqa: E731
    dirty = [l for l in git("status", "--porcelain").splitlines()
             if any(p in l for p in ("eventful", "kernels/fused_gemm"))]
    out = dict(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        host=socket.gethostname(),
        argv=sys.argv,
        git_head=git("rev-parse", "--short", "HEAD"),
        eventful_files_uncommitted=dirty,
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=gpu_state() if torch.cuda.is_available() else None,
    )
    if extra:
        out.update(extra)
    return out


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    return path
