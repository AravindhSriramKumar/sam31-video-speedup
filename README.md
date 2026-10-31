# SAM 3.1 Video Speedup

**Accelerating SAM 3.1 Video Inference Through Temporal Computation Reuse**

Aravindh Sriram Kumar and Shaik Sadik. CS6886, IIT Madras.

SAM 3.1 re-encodes every video frame with a large Vision Transformer, even when little has changed. This repository measures how much of that work can be reused. It holds the dense baseline, EP-ViT, a codec-guided encoder that recomputes only the tokens H.264 motion vectors and residuals mark as changed, and Eventful, which recomputes only the tokens that changed most inside the trunk.

> **Status:** Dense, EP-ViT, Eventful and the fused kernel are all included, each at its best configuration.

## Results

SA-Co/VEval SA-V val, 20 videos at 24 fps, 189 (video, noun phrase) pairs, official evaluator. Measured on an NVIDIA RTX A5000 (24 GB). The full tables are in [`results/`](results/).

| Method | cgF1 | ΔcgF1 | pHOTA | ΔpHOTA | Encoder ms/frame | Encoder speedup | Trunk TMAC/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense SAM 3.1 (B2d, compiled) | 12.41 | 0 | 48.88 | 0 | 89.9 | 1.00× | 2.699 |
| EP-ViT 75% (`D_ep75`, compiled) | 12.26 | −0.16 | 48.27 | −0.61 | 68.3 | 1.32× | 2.052 |
| Eventful | 12.27 | −0.15 | 48.97 | +0.09 | 75.5 | 1.19× | not logged |
| Eventful + fused kernel | 11.55 | −0.86 | 48.57 | −0.32 | 75.2 | 1.20× | not logged |

The same methods by subset:

| Method | cgF1 `normal` | cgF1 `high_motion` | pHOTA `normal` | pHOTA `high_motion` | Encoder ms/frame `normal` | Encoder ms/frame `high_motion` | Speedup `normal` | Speedup `high_motion` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 15.43 | 11.00 | 55.45 | 38.35 | 89.4 | 90.2 | 1.00× | 1.00× |
| EP-ViT 75% | 14.63 | 11.11 | 54.25 | 38.88 | 68.5 | 68.2 | 1.31× | 1.32× |
| Eventful | 15.61 | 10.78 | 55.38 | 38.61 | 57.8 | 85.9 | 1.55× | 1.05× |
| Eventful + fused kernel | 14.78 | 9.96 | 55.35 | 37.80 | 58.4 | 85.0 | 1.53× | 1.06× |

### End-to-end latency

End-to-end (E2E) is the median per-frame model step. Three measurements exist, and they answer different questions:

| Measurement | Scope | Dense ms | Method ms | E2E speedup | Use it for |
|---|---|---:|---:|---:|---|
| EP-ViT, controlled: run side by side on idle GPUs | 10 `normal` pairs | 187.5 | 163.6 | **1.15×** | Speedup claims |
| EP-ViT, controlled: run side by side on idle GPUs | 10 `high_motion` pairs | 188.1 | 165.8 | **1.13×** | Speedup claims |
| Eventful, controlled, 40 frames | One `normal` video | 189.6 | 158.5 | **1.20×** | Speedup claims, this video only |
| Eventful, controlled, 40 frames | One `high_motion` video | 188.0 | 189.8 | **0.99×** | Speedup claims, this video only |
| EP-ViT, full sweep on a shared host | 189 pairs | 175.0 | 137.8 | 1.27× (not reliable) | Record of the sweep only |
| Eventful, full sweep on a shared host | 189 pairs | 175.0 | 158.1 | 1.11× (not reliable) | Record of the sweep only |
| Eventful + fused kernel, full sweep on a shared host | 189 pairs | 175.0 | 155.2 | 1.13× (not reliable) | Record of the sweep only |

The sweep values are affected by host load. EP-ViT doesn't touch the rest of the step, yet that part differed between the sweep runs. The EP-ViT and Eventful controlled measurements use different pairs, so don't compare them directly. The fused kernel has no controlled E2E measurement.

### Reading these numbers

- **EP-ViT** gives the same speedup on every video and loses a little accuracy on `normal`. **Eventful** stays closest to dense and is the fastest on `normal` videos, but gains little on `high_motion`, where most tokens change.
- `D_ep75` is reported because it has the best balance of accuracy and speed among the EP-ViT configurations with controlled E2E data.
- The fused kernel saves about 0.2 ms per frame in the trunk benchmark (74.7 → 74.5 ms). Reading rows by index slows `fc1` by about 10%, which cancels the copies it removes. So plain Eventful stays the best configuration.
- Accuracy differences this small may be noise. No significance testing was run on the 189-pair differences above.
- cgF1 measures detection and segmentation; pHOTA also measures how well objects are tracked over time. Per-video cgF1 values are noisy, so compare the totals above, not single videos.
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
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful.py --sample
sh kernels/fused_gemm/build.sh     # once: needs the CUDA toolkit, git and network access
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_fused.py --sample
python experiments/render_sample.py
```

Masks and timings go to `results/runs/sample/`. Change the text prompt with `--prompt`. `render_sample.py` draws the four runs side by side in `results/sample_demo/`.

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

# Eventful
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful.py --subset normal
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful.py --subset high_motion

# Eventful + fused kernel
sh kernels/fused_gemm/build.sh
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_fused.py --subset normal
CUDA_VISIBLE_DEVICES=0 python experiments/run_eventful_fused.py --subset high_motion

# Score all four on the 189 pairs and compare with results/compiled_results.csv
python experiments/analyze_results.py
```

Accuracy should match the committed tables. Latency depends on your GPU and host load, so compare methods within one machine, ideally side by side.

To try one pair, pass a JSON list of pair ids with `--pair-ids`.

## Eventful

Eventful reuses work between frames inside the ViT trunk. A gate sits around the MLP of each block. Each frame it scores how much every token's input has changed, recomputes the K tokens that changed most, and reuses the cached output for the rest.

The best configuration is `B1:mlp:k3840:lag0.6:bf16ref:torchscore:fb4608:tk:cg`:

| Option | Meaning |
|---|---|
| `B1` | Base trunk with bf16 linear weights and compiled RoPE |
| `mlp` | Gate the MLP only |
| `k3840` | Start at K = 3,840 of 5,184 tokens (75%) |
| `lag0.6` | Choose K per frame from how many tokens changed by more than 0.6 on the previous frame |
| `bf16ref` | Keep the reference cache in bf16 |
| `torchscore` | Score tokens with PyTorch ops, which the compiler fuses |
| `fb4608` | Run a dense frame when K reaches 4,608 |
| `tk` | Select the top K with a Triton kernel |
| `cg` | Compile with CUDA graphs |

The fused kernel run adds `fgfull`: one CUTLASS kernel for the two MLP GEMMs, reading and writing rows by index. `kernels/fused_gemm/` holds the kernel and `eventful/kernels/` holds its wrapper and the top-K and scoring kernels. It needs the NVIDIA CUDA toolkit, `git` and network access for the first CUTLASS download. If `nvcc` isn't on `PATH`, set `NVCC` to its full path. `ARCH` sets the target and defaults to `sm_86`. Everything else runs without it.

## Tests

```bash
python tests/test_protocol.py            # repository checks, offline
python tests/test_epvit.py               # EP-ViT CPU tests
EPVIT_GPU=0 python tests/test_epvit.py   # adds GPU tests (needs checkpoint and data)
for t in annotate baseline block_equivalence gates pipeline_adapter; do
  CUDA_VISIBLE_DEVICES=0 python tests/eventful/test_$t.py   # Eventful tests (need checkpoint and data)
done
```

## Repository layout

```text
checkpoints/          checkpoint instructions
data/                 sample video, dataset download and validation, codec priors, frozen protocol
epvit/                EP-ViT encoder and codec code
eventful/             Eventful gates, controller and kernels
experiments/          runners for each method, scoring and the sample renderer
experiments/harness/  shared runtime, baselines, timing, sessions, scoring
kernels/fused_gemm/   fused GEMM kernel and its build script
report/               technical report (draft)
results/              dense/, epvit/, eventful/, eventful_fused/, tables, sample_demo/; new runs go to results/runs/ (git-ignored)
tests/                tests
third_party/sam3/     vendored SAM 3.1
```

## References

- Carion et al., *SAM 3: Segment Anything with Concepts*, Meta. [Code](https://github.com/facebookresearch/sam3)
- Qiu et al., *EP-ViT: Leveraging Encoding Priors to Eliminate Spatio-Temporal Redundancy in Vision Transformer*, ECAI 2025.
- Dutson et al., *Eventful Transformers: Leveraging Temporal Redundancy in Vision Transformers*, ICCV 2023. [Code](https://github.com/WISION-Lab/eventful-transformer)
- Ravi et al., *Segment Anything in Images and Videos* (SAM 2, SA-V dataset), Meta.

## Licences and acknowledgements

This project uses SAM 3.1 (SAM Materials). `third_party/sam3` and the checkpoint are covered by the SAM License in `third_party/sam3/LICENSE`.

The SA-Co/VEval SA-V domain is listed as CC-BY-NC 4.0. Evaluation data and `data/sample_video.mp4`, a re-encode of SA-V video `sav_017083`, are for non-commercial use with attribution to Meta's SA-V and SA-Co datasets.

No licence has been chosen yet for the project's own code.
