"""Offline checks of the repository itself: no GPU, checkpoint or dataset needed.

    python tests/test_protocol.py
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import paths  # noqa: E402
from data import saco_data  # noqa: E402

BUDGET_MB = 50


def test_frozen_protocol():
    p = saco_data.load_eval_pairs()
    s = saco_data.load_subsets()
    assert list(p["videos"]) == s["normal"] + s["high_motion"] and len(p["videos"]) == 20
    assert sum(p["attempted"].values()) == 198 and sum(len(v) for v in p["pairs"].values()) == 189
    assert p["native_frames_fed"] == 7616


def test_no_eventful_dependency():
    """The Dense and EP-ViT code must import nothing from the Eventful implementation."""
    for f in [*(p for p in ROOT.glob("experiments/**/*.py") if not p.name.startswith("run_eventful")),
              *ROOT.glob("epvit/**/*.py"), *ROOT.glob("data/*.py")]:
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n.split(".")[0] in ("eventful", "eventful_sam31") for n in names), f


def test_sample_video():
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height",
                          "-of", "json", str(paths.SAMPLE_VIDEO)], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    assert (s["codec_name"], s["width"], s["height"]) == ("h264", 480, 848), s
    assert paths.SAMPLE_VIDEO.stat().st_size < 10 << 20


def test_tracked_files_are_small_and_clean():
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True, text=True).stdout.split()
    forbidden = (".pt", ".pth", ".npz", ".log")
    for name in tracked:
        f = ROOT / name
        if not f.is_file():
            continue
        assert f.stat().st_size < BUDGET_MB << 20, name
        assert not name.endswith(forbidden), name


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"passed   {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAILED   {name}: {e}")
    sys.exit(1 if fails else 0)
