# SAM 3.1 Video Speedup

**Accelerating SAM 3.1 Video Inference Through Temporal Computation Reuse**

Aravindh Sriram Kumar and Shaik Sadik. CS6886, IIT Madras.

SAM 3.1 re-encodes every video frame with a large Vision Transformer, even when little has changed. This repository measures how much of that work can be reused. It currently holds the dense baseline and EP-ViT, a codec-guided encoder that recomputes only the tokens H.264 motion vectors and residuals mark as changed.

> **Status:** Dense B2d and EP-ViT are included and validated. **The Eventful implementation and fused scatter-gather kernels are pending.**

## Results

SA-Co/VEval SA-V val, 20 videos at 24 fps, 189 (video, noun phrase) pairs, official evaluator. Measured on an NVIDIA RTX A5000 (24 GB). The full tables are in [`results/`](results/).

| Method | pHOTA | ΔpHOTA | Encoder ms/frame | Encoder speedup | Trunk TMAC/frame |
|---|---:|---:|---:|---:|---:|
| Dense SAM 3.1 (B2d, compiled) | 48.88 | 0 | 89.9 | 1.00× | 2.699 |
| EP-ViT 75% (`D_ep75`, compiled) | 48.27 | −0.61 | 68.3 | 1.32× | 2.052 |

pHOTA by subset (Dense → EP-ViT): `normal` 55.45 → 54.25, `high_motion` 38.35 → 38.88.

### End-to-end latency

End-to-end (E2E) is the median per-frame model step. Two measurements exist, and they answer different questions:

| Measurement | Pairs | Dense ms | EP-ViT ms | E2E speedup | Use it for |
|---|---:|---:|---:|---:|---|
| Controlled: run side by side on idle GPUs | 10 `normal` | 187.5 | 163.6 | **1.15×** | Speedup claims |
| Controlled: run side by side on idle GPUs | 10 `high_motion` | 188.1 | 165.8 | **1.13×** | Speedup claims |
| Full sweep: runs days apart on a shared host | 189 | 175.0 | 137.8 | 1.27× (not reliable) | Record of the sweep only |

The 1.27× is a host-load artefact. EP-ViT doesn't touch the rest of the step, yet that part differed between the sweep runs. Use the controlled numbers.

### Reading these numbers

- `D_ep75` is reported because it has the best balance of accuracy and speed among the configurations with controlled E2E data. `F_ep50` is faster and loses more accuracy; it is not part of this release's results.
- No significance testing was run on the differences above.
- cgF1 is in the tables, but per-video values are noisy, so pHOTA is the main accuracy measure.
- 198 pairs are attempted. 189 completed in every run and are scored; the other 9 stop on an upstream SAM 3.1 tracker bug or on GPU memory.

## Installation

Requires an NVIDIA GPU (24 GB for the full evaluation), CUDA 12.8 and `ffmpeg` with libx264 on `PATH`.

```bash
git clone https://github.com/AravindhSriramKumar/sam31-video-speedup.git
cd sam31-video-speedup
conda env create -f environment.yml
conda activate sam31
pip install -e third_party/sam3
```

Tested with Python 3.12, PyTorch 2.10.0, PyAV 19.0.2, ffmpeg 4.4.2 and CUDA 12.8. Exact package versions are in `requirements.txt`. `third_party/sam3/VENDORED.md` records the vendored SAM 3.1 commit and its local patch.

## Checkpoint

Both methods use the same weights. See [`checkpoints/README.md`](checkpoints/README.md) for access, download and the SHA-256. Run `python paths.py` to see which inputs are present.

## Quick start: included sample

`data/sample_video.mp4` is committed, so this needs only the checkpoint. It's a smoke test: it proves the code runs and reports timing, but there's **no ground truth and no accuracy score**.

```bash
python data/generate_codec_priors.py --sample
CUDA_VISIBLE_DEVICES=0 python experiments/run_dense.py --sample
CUDA_VISIBLE_DEVICES=0 python experiments/run_epvit.py --sample --config D_ep75 --compile
```

Masks and timings go to `results/runs/sample/`. Change the text prompt with `--prompt`.

### About the sample

It is SA-V video `sav_017083`, 480×848, 24 fps, 577 frames (24 s), 6.2 MB, copied unchanged from this project's `crf23_g192` encode. It is also one of the 20 evaluation videos.

Motion below is the share of 72×72 token cells whose mean absolute luma change between consecutive frames exceeds 8.

| Interval | Motion | Character |
|---|---:|---|
| 3–5 s | 0.88–0.90 | High |
| 10–13 s | 0.81–0.86 | High |
| 7–9 s | 0.44–0.46 | Moderate |
| 14–15 s | 0.16 | Low |

Only 14–15 s is truly low-motion, and most of the clip is fairly busy. It has the widest motion range of the 20 evaluation encodes. At the 4-frame spacing used to choose the `normal` and `high_motion` subsets, the same clip reads high throughout (0.76–0.97).

## Evaluation data

```bash
hf auth login                      # needs approved access to facebook/SACo-VEval
bash data/download_saco_eval.sh
```

- **Annotations:** `facebook/SACo-VEval` on Hugging Face (gated, manual approval), checked against a SHA-256.
- **Frames:** the SA-V frames hosted by Roboflow, the ready-to-use source linked from SAM 3's [VEval instructions](third_party/sam3/scripts/eval/veval/README.md). They come from `https://sa-co.roboflow.com/veval/saco_sav.zip`. This is not a Meta-hosted URL; Meta's SA-V page was unavailable when this was built.
- **Size:** the script reads only the byte ranges of the 20 videos (2.2 GB) from the 32.6 GB archive. If the host stops honouring range requests, it downloads the whole archive instead, resumably.
- **Re-running** skips finished videos and resumes an interrupted one.

Output goes to `data/saco_sav/JPEGImages_24fps/`. Validate it at any time:

```bash
python data/prepare_saco_eval.py
```

This checks the annotation hash, the 20 video ids and order, each video's frame hash and order, the 7,616 frames fed at 24 fps, and the 198 and 189 pair counts. The expected values are in `data/eval_pairs.json`. It exits non-zero on any mismatch.

## Codec priors

EP-ViT needs H.264 motion vectors, residuals, intra shares and frame types on the token grid. They are generated, not stored in Git.

```bash
python data/generate_codec_priors.py --sample   # data/sample_priors/
python data/generate_codec_priors.py --eval     # data/saco_sav/priors/, CPU only
```

The encoding is libx264, CRF 23, GOP 192, I and P frames only, one reference frame (`crf23_g192`). `_iffix` means frame types come from ffprobe, because `epvit/codec/extract.py` mislabels later I-frames under PyAV 19 and EP-ViT would then never flush on them. Use `--videos sav_017083` to process a subset.

## Reproducing the results

Each method processes 79,794 model steps, so expect several hours per method on one GPU. Runs resume from `pairs/` if interrupted.

```bash
# Check the committed tables (no data or GPU needed)
python experiments/analyze_results.py --committed

# Dense B2d
CUDA_VISIBLE_DEVICES=0 python experiments/run_dense.py --subset normal
CUDA_VISIBLE_DEVICES=0 python experiments/run_dense.py --subset high_motion

# EP-ViT
python data/generate_codec_priors.py --eval
CUDA_VISIBLE_DEVICES=0 python experiments/run_epvit.py --subset normal --config D_ep75 --compile
CUDA_VISIBLE_DEVICES=0 python experiments/run_epvit.py --subset high_motion --config D_ep75 --compile

# Score both on the 189 pairs and compare with results/compiled_results.csv
python experiments/analyze_results.py
```

Accuracy should match the committed tables. Latency depends on your GPU and host load, so compare methods within one machine, ideally side by side.

To try one pair, pass a JSON list of pair ids with `--pair-ids`.

## Tests

```bash
python tests/test_protocol.py            # repository checks, offline
python tests/test_epvit.py               # EP-ViT CPU tests
EPVIT_GPU=0 python tests/test_epvit.py   # adds GPU tests (needs checkpoint and data)
```

## Repository layout

```text
checkpoints/          checkpoint instructions
data/                 sample video, dataset download and validation, codec priors, frozen protocol
epvit/                EP-ViT encoder and codec code
experiments/          runners, analysis
experiments/harness/  shared runtime, baselines, timing, sessions, scoring
report/               technical report (draft)
results/              validated results; new runs go to results/runs/ (git-ignored)
tests/                tests
third_party/sam3/     vendored SAM 3.1
```

## Adding Eventful and fused kernels

The pending work fits the existing layout, using the same checkpoint, data and protocol:

| Item | Location |
|---|---|
| Eventful implementation, with one code path for standard PyTorch and fused backends | `eventful/` |
| Fused scatter-gather kernels | `kernels/` |
| Runners | `experiments/run_eventful.py`, `experiments/run_eventful_fused.py` |
| Results | `results/eventful/`, `results/eventful_fused/` |

`experiments/harness/pipeline.py` already accepts a stateful trunk through the `controller` argument of `PipelineHandle`.

## References

- Carion et al., *SAM 3: Segment Anything with Concepts*, Meta. [Code](https://github.com/facebookresearch/sam3)
- Qiu et al., *EP-ViT: Leveraging Encoding Priors to Eliminate Spatio-Temporal Redundancy in Vision Transformer*, ECAI 2025.
- Dutson et al., *Eventful Transformers: Leveraging Temporal Redundancy in Vision Transformers*, ICCV 2023. [Code](https://github.com/WISION-Lab/eventful-transformer)
- Ravi et al., *Segment Anything in Images and Videos* (SAM 2, SA-V dataset), Meta.

## Licences and acknowledgements

This project uses SAM 3.1 (SAM Materials). `third_party/sam3` and the checkpoint are covered by the SAM License in `third_party/sam3/LICENSE`.

The SA-Co/VEval SA-V domain is listed as CC-BY-NC 4.0. Evaluation data and `data/sample_video.mp4`, a re-encode of SA-V video `sav_017083`, are for non-commercial use with attribution to Meta's SA-V and SA-Co datasets.

No licence has been chosen yet for the project's own code.
