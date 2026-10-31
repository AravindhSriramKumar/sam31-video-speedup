"""Fetch and validate the evaluation frames of the frozen 24 fps protocol (see data/eval_pairs.json).

    python data/prepare_saco_eval.py --fetch      # used by download_saco_eval.sh: the 20 videos' frames
    python data/prepare_saco_eval.py              # validate only; exit 1 on any mismatch

Frames come from the SA-Co/VEval SA-V frames hosted by Roboflow (linked from third_party/sam3/scripts/eval/veval/
README.md). The full archive is 32.6 GB; only the byte ranges of the 20 videos are read from it, so about 2.3 GB is
transferred. If the host stops honouring range requests, the whole archive is downloaded (resumable) instead.
Every extracted frame is checked against the archive's CRC, and each video against the hash in eval_pairs.json, so
a different dataset cannot pass silently.
"""
import argparse
import hashlib
import shutil
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import saco_data  # noqa: E402

ZIP_URL = "https://sa-co.roboflow.com/veval/saco_sav.zip"
CACHE = saco_data.SACO / ".cache"
CHUNK = 64 << 20
ENTRY_MARK = "JPEGImages_24fps/"


class RangeUnavailable(RuntimeError):
    pass


def http_range(start, end):
    """Bytes [start, end] through curl. The CDN sometimes answers with the whole file, so --max-filesize refuses it."""
    want = end - start + 1
    for attempt in range(8):
        r = subprocess.run(["curl", "-sS", "-f", "--connect-timeout", "20", "--max-time", "600",
                            "--max-filesize", str(want), "-r", f"{start}-{end}", ZIP_URL], capture_output=True)
        if r.returncode == 0 and len(r.stdout) == want:
            return r.stdout
        time.sleep(2 ** attempt)
    raise RangeUnavailable(f"bytes {start}-{end}: curl exit {r.returncode}, {len(r.stdout)} bytes")


def remote_size():
    out = subprocess.run(["curl", "-sSIL", "--connect-timeout", "20", ZIP_URL], capture_output=True, text=True,
                         check=True).stdout
    sizes = [int(l.split(":")[1]) for l in out.splitlines() if l.lower().startswith("content-length:")]
    return sizes[-1]


class RemoteFile:
    """Read-only seekable view of the remote zip, enough for zipfile to read its central directory."""

    def __init__(self):
        self.size, self.pos = remote_size(), 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        b = http_range(self.pos, self.pos + n - 1)
        self.pos += n
        return b


def member_video(info):
    """'.../JPEGImages_24fps/sav_017083/00000.jpg' -> ('sav_017083', '00000.jpg')."""
    if ENTRY_MARK not in info.filename or info.is_dir():
        return None
    parts = info.filename.split(ENTRY_MARK, 1)[1].split("/")
    return (parts[0], parts[1]) if len(parts) == 2 else None


def write_frame(video, name, data, info):
    if zlib.crc32(data) != info.CRC:
        raise RuntimeError(f"CRC mismatch: {info.filename}")
    dst = saco_data.MEDIA / video / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(dst)


def fetch_span(video, members):
    """Download one video's contiguous archive span to CACHE (resumes from the part file), extract, check CRCs."""
    start = min(i.header_offset for i, _ in members)
    last = max(members, key=lambda m: m[0].header_offset)[0]
    end = last.header_offset + 30 + len(last.filename.encode()) + 256 + last.compress_size   # past the last entry
    total = sum(i.compress_size for i, _ in members)
    if end - start > 2 * total + (1 << 20):
        raise RangeUnavailable(f"{video}: entries are not contiguous")
    part = CACHE / f"{video}.part"
    with open(part, "ab") as fh:
        while start + fh.tell() < end:
            pos = start + fh.tell()
            fh.write(http_range(pos, min(pos + CHUNK, end) - 1))
    buf = part.read_bytes()
    for info, name in members:
        o = info.header_offset - start
        sig, fn_len, ex_len = struct.unpack("<I22x2H", buf[o:o + 30])
        assert sig == 0x04034b50, info.filename
        raw = buf[o + 30 + fn_len + ex_len:o + 30 + fn_len + ex_len + info.compress_size]
        write_frame(video, name, zlib.decompress(raw, -15) if info.compress_type == zipfile.ZIP_DEFLATED else raw, info)
    part.unlink()


def fetch_ranged(todo):
    with zipfile.ZipFile(RemoteFile()) as z:
        infos = z.infolist()
    print(f"archive index: {len(infos)} entries")
    by_video = {}
    for i in infos:
        m = member_video(i)
        if m and m[0] in todo:
            by_video.setdefault(m[0], []).append((i, m[1]))
    missing = todo - set(by_video)
    if missing:
        raise RuntimeError(f"videos not in the archive: {sorted(missing)}")
    for k, video in enumerate(sorted(todo)):
        fetch_span(video, by_video[video])
        (CACHE / f"{video}.done").touch()
        print(f"[{k + 1}/{len(todo)}] {video}: {len(by_video[video])} frames", flush=True)


def fetch_full(todo):
    zpath = CACHE / "saco_sav.zip"
    print("range requests unavailable: downloading the whole 32.6 GB archive (resumable)", flush=True)
    subprocess.run(["curl", "-L", "-f", "-C", "-", "--retry", "20", "--retry-delay", "10", "-o", str(zpath), ZIP_URL],
                   check=True)
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            m = member_video(info)
            if m and m[0] in todo:
                write_frame(m[0], m[1], z.read(info), info)
    for video in todo:
        (CACHE / f"{video}.done").touch()
    zpath.unlink()


def fetch():
    videos = saco_data.load_eval_pairs()["videos"]
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = {v for v in videos if not (CACHE / f"{v}.done").exists()}
    print(f"{len(videos) - len(todo)} of {len(videos)} videos already present")
    if not todo:
        return
    need = 3 << 30
    if shutil.disk_usage(saco_data.SACO).free < need:
        raise SystemExit(f"need {need >> 30} GB free in {saco_data.SACO}")
    try:
        fetch_ranged(todo)
    except RangeUnavailable as e:
        print(f"ranged fetch stopped: {e}")
        fetch_full({v for v in todo if not (CACHE / f"{v}.done").exists()})


def video_digest(folder):
    """SHA-256 over the frames' names and bytes, in frame order."""
    h = hashlib.sha256()
    files = sorted(folder.glob("*.jpg"))
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest(), len(files)


def validate():
    frozen = saco_data.load_eval_pairs()
    errs = []

    def check(cond, msg):
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            errs.append(msg)

    h = hashlib.sha256(saco_data.GT_FILE.read_bytes()).hexdigest() if saco_data.GT_FILE.exists() else "missing"
    check(h == frozen["annotation_sha256"], f"annotations {saco_data.GT_FILE.name} sha256 {h[:16]}...")
    if h == "missing":
        return errs
    gt = saco_data.load_gt()
    subsets = saco_data.load_subsets()
    by_name = saco_data.video_by_name(gt)
    names = subsets["normal"] + subsets["high_motion"]
    check(names == list(frozen["videos"]), f"{len(names)} videos, same ids and order as the protocol")
    check(len(names) == 20 and len(set(names)) == 20, "20 distinct videos")

    native = 0
    for name in names:
        v = by_name[name]
        idx = [int(Path(n).stem) for n in v["file_names"]]
        n = idx[-1] + 1
        native += n
        folder = saco_data.MEDIA / name
        fed = [folder / f"{i:05d}.jpg" for i in range(n)]
        ok_frames = all(f.is_file() and f.stat().st_size > 0 for f in fed)
        ok_order = idx == sorted(idx) and all(i % 4 == 0 for i in idx)
        digest, count = video_digest(folder) if folder.is_dir() else ("missing", 0)
        check(ok_frames and ok_order and digest == frozen["videos"][name]["sha256"]
              and count == frozen["videos"][name]["frames"], f"{name}: {count} frames, {n} fed, hash {digest[:12]}")
    check(native == frozen["native_frames_fed"], f"{native} native frames fed at 24 fps")

    attempted, scored = 0, 0
    for s in ("normal", "high_motion"):
        ids = {p["id"] for n in subsets[s] for p in saco_data.pairs_for_video(gt, by_name[n]["id"])}
        attempted += len(ids)
        scored += len(frozen["pairs"][s])
        check(len(ids) == frozen["attempted"][s], f"{s}: {len(ids)} pairs attempted")
        check(set(frozen["pairs"][s]) <= ids, f"{s}: {len(frozen['pairs'][s])} scored pairs are a subset of them")
    check(attempted == 198, f"{attempted} pairs attempted in total")
    check(scored == 189, f"{scored} pairs scored in total")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download the frames first")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    errs = validate()
    print(f"\n{'VALID' if not errs else f'{len(errs)} CHECKS FAILED'}: 20-video, 24 fps, 189-pair protocol")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
