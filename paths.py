"""Project paths, all relative to the repo root. Import from scripts instead of hard-coding."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

SAM3_REPO = PROJECT / "third_party/sam3"
CKPT = PROJECT / "checkpoints/sam3.1_multiplex.pt"
CKPT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"

DATA = PROJECT / "data"
SAMPLE_VIDEO = DATA / "sample_video.mp4"
SAMPLE_FRAMES = DATA / "sample_frames"          # decoded from sample_video.mp4 (git-ignored)
SAMPLE_PRIORS = DATA / "sample_priors"          # <enc_tag>/sample_video.npz (git-ignored)
SACO_PRIORS = DATA / "saco_sav/priors"          # <enc_tag>/<video>.npz (git-ignored)
SACO_VIDEOS = DATA / "saco_sav/videos"          # <enc_tag>/<video>.mp4 (git-ignored)

RESULTS = PROJECT / "results"
RUNS = RESULTS / "runs"                         # output of new runs (git-ignored); committed results sit beside it


if __name__ == "__main__":
    for name, p in sorted(globals().items()):
        if isinstance(p, Path):
            print(f"{'ok ' if p.exists() else 'MISSING'} {name:14s} {p}")
