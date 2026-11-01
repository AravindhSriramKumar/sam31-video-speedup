# SAM 3.1 Video Speedup: Accelerating SAM 3.1 Video Inference Through Temporal Computation Reuse

Authors: Aravindh Sriram Kumar and Shaik Sadik. CS6886, IIT Madras.

## Introduction

This repository contains the code, evaluation protocol and results for our course project on making SAM 3.1 video inference faster.

It includes dense SAM 3.1, EP-ViT and Eventful, each at its best configuration, plus two more Eventful runs for comparison: the MLP gate only, and the MLP gate only with a fused kernel.

- [Overview](#overview)
- [Create Environment](#create-environment)
- [Prepare Models](#prepare-models)
- [Prepare Data](#prepare-data)
- [Usage](#usage)
    - [Inference](#inference)
    - [Evaluation](#evaluation)
- [Main Results](#main-results)
- [Acknowledgments](#acknowledgments)

## Overview

### Motivation

SAM 3.1 re-encodes every video frame with a large Vision Transformer, even when little has changed. This project measures how much of that work can be reused between frames.

### Method

**Dense** is the baseline: SAM 3.1 with a compiled trunk and bf16 linear weights (`B2d`).

**EP-ViT** (`D_ep75`, compiled) is a codec-guided encoder. It recomputes only the tokens that H.264 motion vectors and residuals mark as changed.

**Eventful** reuses work inside the ViT trunk. Gates sit around the qkv projection, the attention output projection and the MLP of each block. Each frame a gate scores how much every token's input has changed, recomputes the K tokens that changed most, and reuses the cached output for the rest. The best configuration, chosen for speed, is `B1:mlp+qkv+proj:k3840:lag0.6:bf16ref:torchscore:fb4608:tk:cg`. It has the largest Eventful speedup, but gives up some accuracy:

| Option | Meaning |
|---|---|
| `B1` | Base trunk with bf16 linear weights and compiled RoPE |
| `mlp+qkv+proj` | Gate the MLP, qkv and proj of each block |
| `k3840` | Start at K = 3,840 of 5,184 tokens (75%) |
| `lag0.6` | Choose K per frame from how many tokens changed by more than 0.6 on the previous frame |
| `bf16ref` | Keep the reference cache in bf16 |
| `torchscore` | Score tokens with PyTorch ops, which the compiler fuses |
| `fb4608` | Run a dense frame when K reaches 4,608 |
| `tk` | Select the top K with a Triton kernel |
| `cg` | Compile with CUDA graphs |

**Eventful (MLP gate only)** uses the same options with `mlp` in place of `mlp+qkv+proj`. It is slower than three gates and more accurate.

**Eventful + fused kernel (MLP gate only)** adds `fgfull` to the MLP-only configuration: one CUTLASS kernel for the two MLP GEMMs, reading and writing rows by index.

### Performance

We evaluate on 20 SA-V validation videos from SA-Co/VEval at 24 fps, on an NVIDIA RTX A5000 (24 GB). One evaluation pair is one video with one noun phrase prompt.

The videos form two subsets, chosen by how much each one changes between frames. Motion is the share of cells on the 72×72 token grid whose mean absolute pixel change between consecutive annotated frames exceeds 8.

| Subset | Videos | Scored pairs | Motion |
|---|---:|---:|---:|
| `normal` | 10 | 70 | 0.06–0.18 |
| `high_motion` | 10 | 119 | 0.77–0.89 |

The table below covers all 189 pairs together. The values of metrics in this table are averages across all video pairs(normal +high motion both).

| Method | cgF1 | ΔcgF1 | pHOTA | ΔpHOTA | Encoder ms/frame | Encoder speedup |
|---|---:|---:|---:|---:|---:|---:|
| Dense SAM 3.1 (B2d, compiled) | 12.41 | 0 | 48.88 | 0 | 89.9 | 1.00× |
| EP-ViT 75% (`D_ep75`, compiled) | 12.26 | −0.16 | 48.27 | −0.61 | 68.3 | 1.32× |
| Eventful three gates | 11.25 | −1.17 | 47.48 | −1.40 | 68.7 | 1.31× |
| Eventful (MLP gate only) | 12.27 | −0.15 | 48.97 | +0.09 | 75.5 | 1.19× |
| Eventful + fused kernel (MLP gate only) | 11.55 | −0.86 | 48.57 | −0.32 | 75.2 | 1.20× |

Across all 20 videos, Eventful three gates matches EP-ViT's encoder speedup but has the lowest accuracy of the five, and Eventful (MLP gate only) stays closest to dense.

These overall values hide a large difference between the subsets. EP-ViT speeds up both by about the same amount, while Eventful's speedup depends on motion: three gates reaches 1.90× on `normal`, the fastest of any method, but only 1.11× on `high_motion`. On single videos it ranges from 0.97× to 2.23×. [Main Results](#main-results) gives each subset separately, along with end-to-end latency.

---

## Create Environment

### Prerequisites

The code requires an NVIDIA GPU (24 GB for the full evaluation), CUDA 12.8 and `ffmpeg` with libx264 on `PATH`.

```bash
git clone https://github.com/AravindhSriramKumar/sam31-video-speedup.git
cd sam31-video-speedup
conda env create -f environment.yml
conda activate sam31
pip install -e third_party/sam3
```

Tested with Python 3.12, PyTorch 2.10.0, PyAV 19.0.2, ffmpeg 4.4.2 and CUDA 12.8. Exact package versions are in `requirements.txt`. `third_party/sam3/VENDORED.md` records the vendored SAM 3.1 commit and its local patch.

### Fused kernel

Only the fused kernel runs need this step. It needs the NVIDIA CUDA toolkit, `git` and network access for the first CUTLASS download.

```bash
sh kernels/fused_gemm/build.sh
```

If `nvcc` isn't on `PATH`, set `NVCC` to its full path. `ARCH` sets the target and defaults to `sm_86`. `kernels/fused_gemm/` holds the kernel, and `eventful/kernels/` holds its wrapper and the top-K and scoring kernels.

## Prepare Models

All methods use the same weights, `sam3.1_multiplex.pt`, from the gated Hugging Face repository [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1). After your access is approved:

```bash
hf auth login
hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir checkpoints
```

See [`checkpoints/README.md`](checkpoints/README.md) for the SHA-256. Run `python paths.py` to see which inputs are present.

## Prepare Data

### Evaluation data

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

### Codec priors

EP-ViT needs H.264 motion vectors, residuals, intra shares and frame types on the token grid. They are generated, not stored in Git.

```bash
python data/generate_codec_priors.py --sample   # data/sample_priors/
python data/generate_codec_priors.py --eval     # data/saco_sav/priors/, CPU only
```

The encoding is libx264, CRF 23, GOP 192, I and P frames only, one reference frame (`crf23_g192`). `_iffix` means frame types come from ffprobe, because `epvit/codec/extract.py` mislabels later I-frames under PyAV 19 and EP-ViT would then never flush on them. Use `--videos sav_017083` to process a subset.

## Usage

### Inference

`data/sample_video.mp4` is committed, so this needs only the checkpoint. It's a smoke test: it proves the code runs and reports timing, but there's **no ground truth and no accuracy score**.

```bash
python data/generate_codec_priors.py --sample
CUDA_VISIBLE_DEVICES=0 python experiments/run_dense.py --sample
CUDA_VISIBLE_DEVICES=0 python experiments/run_epvit.py --sample --config D_ep75 --compile
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_three_gates.py --sample
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful.py --sample
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_fused.py --sample
```

Masks and timings go to `results/runs/sample/`. Change the text prompt with `--prompt`.

Results for the prompt `a paw`, with the four original runs back to back on one idle GPU and three gates measured later against a fresh Dense run on one idle GPU (87.5 ms encoder, 195.1 ms step, the baseline for its row):

| Method | Encoder ms/frame | Encoder speedup | Median step ms | E2E speedup | Mask agreement with Dense | pHOTA, `sav_017083` | cgF1, `sav_017083` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 87.9 | 1.00× | 197.1 | 1.00× | 1.000 | 30.68 | 1.80 |
| EP-ViT 75% | 67.8 | 1.30× | 176.9 | 1.11× | 0.765 | 27.45 | 1.19 |
| Eventful three gates| 74.7 | 1.17× | 185.2 | 1.05× | 0.701 | 29.70 | 2.72 |
| Eventful (MLP gate only) | 80.4 | 1.09× | 191.7 | 1.03× | 0.851 | 31.02 | 2.65 |
| Eventful + fused kernel (MLP gate only) | 80.5 | 1.09× | 191.0 | 1.03× | 0.851 | 30.49 | 2.39 |

- **The sample run has no ground truth,** so it gives no accuracy score of its own. pHOTA and cgF1 above come from the 17 scored evaluation pairs of the same video, `sav_017083`, in `results/per_video_results.csv`. They use other text prompts and the native frames, not the `a paw` run. One video is noisy, especially for cgF1.
- **Mask agreement** is the mean per-frame IoU with Dense on the `a paw` run. It shows how close the masks are to Dense, not which is right.
- **Eventful gains less here than on `normal` videos** because this clip is mostly high-motion, so most tokens are recomputed.

`results/sample_demo/` holds the saved table and video from this run. The masks of all five methods are side by side in [`comparison.mp4`](results/sample_demo/comparison.mp4). Colours mark object ids, which differ between runs, so a paw can change colour across panels.

The sample is SA-V video `sav_017083`, 480×848, 24 fps, 577 frames (24 s), 6.2 MB, copied unchanged from this project's `crf23_g192` encode. It is also one of the 20 evaluation videos. Motion below is the share of 72×72 token cells whose mean absolute luma change between consecutive frames exceeds 8.

| Interval | Motion | Character |
|---|---:|---|
| 3–5 s | 0.88–0.90 | High |
| 10–13 s | 0.81–0.86 | High |
| 7–9 s | 0.44–0.46 | Moderate |
| 14–15 s | 0.16 | Low |

Only 14–15 s is truly low-motion, and most of the clip is fairly busy. It has the widest motion range of the 20 evaluation encodes. At the 4-frame spacing used to choose the `normal` and `high_motion` subsets, the same clip reads high throughout (0.76–0.97).

### Evaluation

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

# Eventful three gates
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_three_gates.py --subset normal
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_three_gates.py --subset high_motion

# Eventful (MLP gate only)
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful.py --subset normal
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful.py --subset high_motion

# Eventful + fused kernel (MLP gate only)
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_fused.py --subset normal
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_fused.py --subset high_motion

# Score all five on the 189 pairs and compare with results/compiled_results.csv
python experiments/analyze_results.py
```

Accuracy should match the committed tables. Latency depends on your GPU and host load, so compare methods within one machine, ideally side by side.

To try one pair, pass a JSON list of pair ids with `--pair-ids`.

## Main Results

The full tables are in [`results/`](results/). The overall table is under [Performance](#performance). The same methods, scored separately on each subset:

| Method | cgF1 `normal` | cgF1 `high_motion` | pHOTA `normal` | pHOTA `high_motion` | Encoder ms/frame `normal` | Encoder ms/frame `high_motion` | Speedup `normal` | Speedup `high_motion` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 15.43 | 11.00 | 55.45 | 38.35 | 89.4 | 90.2 | 1.00× | 1.00× |
| EP-ViT 75% | 14.63 | 11.11 | 54.25 | 38.88 | 68.5 | 68.2 | 1.31× | 1.32× |
| Eventful three gates (best configuration) | 14.30 | 9.94 | 53.53 | 37.34 | 47.0 | 81.4 | 1.90× | 1.11× |
| Eventful (MLP gate only) | 15.61 | 10.78 | 55.38 | 38.61 | 57.8 | 85.9 | 1.55× | 1.05× |
| Eventful + fused kernel (MLP gate only) | 14.78 | 9.96 | 55.35 | 37.80 | 58.4 | 85.0 | 1.53× | 1.06× |

### End-to-end latency

End-to-end (E2E) is the median per-frame model step. Three measurements exist, and they answer different questions:

| Measurement | Scope | Dense ms | Method ms | E2E speedup | Use it for |
|---|---|---:|---:|---:|---|
| EP-ViT, controlled: run side by side on idle GPUs | 10 `normal` pairs | 187.5 | 163.6 | **1.15×** | Speedup claims |
| EP-ViT, controlled: run side by side on idle GPUs | 10 `high_motion` pairs | 188.1 | 165.8 | **1.13×** | Speedup claims |
| Eventful (MLP gate only), controlled, 40 frames | One `normal` video | 189.6 | 158.5 | **1.20×** | Speedup claims, this video only |
| Eventful (MLP gate only), controlled, 40 frames | One `high_motion` video | 188.0 | 189.8 | **0.99×** | Speedup claims, this video only |
| EP-ViT, full sweep on a shared host | 189 pairs | 175.0 | 137.8 | 1.27× (not reliable) | Record of the sweep only |
| Eventful three gates, full sweep on a shared host | 189 pairs | 175.0 | 144.2 | 1.21× (not reliable) | Record of the sweep only |
| Eventful (MLP gate only), full sweep on a shared host | 189 pairs | 175.0 | 158.1 | 1.11× (not reliable) | Record of the sweep only |
| Eventful + fused kernel (MLP gate only), full sweep on a shared host | 189 pairs | 175.0 | 155.2 | 1.13× (not reliable) | Record of the sweep only |

The sweep values are affected by host load. EP-ViT doesn't touch the rest of the step, yet that part differed between the sweep runs. The EP-ViT and Eventful controlled measurements use different pairs, so don't compare them directly. Eventful three gates and the fused kernel have no controlled E2E measurement.

### Reading these numbers

- **EP-ViT** gives the same speedup on every video and loses a little accuracy on `normal`. **Eventful three gates** is the fastest method on `normal` videos and matches EP-ViT's overall encoder speed, but has the lowest accuracy of the five: 1.02 cgF1 and 1.49 pHOTA below Eventful (MLP gate only). **Eventful (MLP gate only)** stays closest to dense. Both Eventful configurations gain little on `high_motion`, where most tokens change.
- `D_ep75` is reported because it has the best balance of accuracy and speed among the EP-ViT configurations with controlled E2E data.
- The fused kernel saves about 0.2 ms per frame in the trunk benchmark (74.7 → 74.5 ms). Reading rows by index slows `fc1` by about 10%, which cancels the copies it removes. So the fused kernel doesn't beat the MLP-only configuration, and it wasn't measured with three gates.
- Accuracy differences this small may be noise. The one significance test run is a paired bootstrap (200 draws) of three gates against MLP-only, saved in `results/eventful_three_gates/`. Only the `normal` pHOTA loss is outside noise (−1.85, 95% interval −3.65 to −0.23); the `normal` cgF1 and both `high_motion` intervals include zero. No other differences above were tested.
- cgF1 measures detection and segmentation; pHOTA also measures how well objects are tracked over time. Per-video cgF1 values are noisy, so compare the totals above, not single videos.
- 198 pairs are attempted. 189 completed in every run and are scored; the other 9 stop on an upstream SAM 3.1 tracker bug or on GPU memory.

## Acknowledgments

This project builds on [SAM 3](https://github.com/facebookresearch/sam3) (Carion et al., Meta), the SA-V dataset (Ravi et al., Meta), EP-ViT (Qiu et al., ECAI 2025) and [Eventful Transformers](https://github.com/WISION-Lab/eventful-transformer) (Dutson et al., ICCV 2023).

This project uses SAM 3.1 (SAM Materials). `third_party/sam3` and the checkpoint are covered by the SAM License in `third_party/sam3/LICENSE`.

The SA-Co/VEval SA-V domain is listed as CC-BY-NC 4.0. Evaluation data and `data/sample_video.mp4`, a re-encode of SA-V video `sav_017083`, are for non-commercial use with attribution to Meta's SA-V and SA-Co datasets.

No licence has been chosen yet for the project's own code.
