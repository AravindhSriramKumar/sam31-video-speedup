"""Encode a frame sequence to H.264 the way a camera would, so its bitstream can supply EP-ViT's priors.

Settings (fixed, recorded in the log):
  libx264, yuv420p, native resolution (padded to even), constant frame rate
  -bf 0            I- and P-frames only (as in the paper)
  -refs 1          every P-frame predicts from the previous frame only, so a motion vector always points into t-1
                   (the MV side data does not say which reference a block used)
  -g G -keyint_min G -sc_threshold 0   an I-frame exactly every G frames, no scene-cut I-frames
  -crf C
The input frames are piped as JPEG bytes in order, so any list of files (e.g. the annotated every-4th frames) works.
"""
import json
import subprocess
import time
from pathlib import Path


def enc_tag(crf, gop):
    return f"crf{crf}_g{gop}"


def encode(frames, out, crf=23, gop=48, fps=6, preset="medium", threads=4):
    """frames: ordered list of JPEG paths. Writes `out` (.mp4) and returns a log dict."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(fps), "-i", "-",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
           "-bf", "0", "-refs", "1", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
           "-threads", str(threads), "-an", str(tmp)]
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(Path(f).read_bytes())
    proc.stdin.close()
    err = proc.stderr.read().decode()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out}: {err}")
    seconds = time.perf_counter() - t0
    tmp.replace(out)
    ftypes = frame_types(out)
    if len(ftypes) != len(frames):
        raise RuntimeError(f"{out}: {len(ftypes)} encoded frames for {len(frames)} inputs")
    return {"video": out.stem, "frames": len(frames), "crf": crf, "gop": gop, "fps": fps, "preset": preset,
            "encode_s": seconds, "encode_ms_per_frame": 1e3 * seconds / len(frames),
            "bytes": out.stat().st_size, "kbps": 8 * out.stat().st_size / 1e3 / (len(frames) / fps),
            "frame_types": "".join(ftypes), "cmd": " ".join(cmd)}


def frame_types(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pict_type",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return [t.strip().rstrip(",") for t in out.split() if t.strip()]


def write_log(path, records):
    path = Path(path)
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update({r["video"]: r for r in records})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(old, indent=1))
    tmp.replace(path)
