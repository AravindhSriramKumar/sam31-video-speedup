# epvit: EP-ViT encoder for SAM 3.1

Replaces SAM 3.1's ViT image encoder with **EP-ViT** (Qiu et al., *EP-ViT: Leveraging Encoding Priors to Eliminate
Spatio-Temporal Redundancy in Vision Transformer*, ECAI 2025; no public code). H.264 motion vectors and residuals classify
every token as static, matchable or unmatchable, and only the tokens that need it are recomputed. The neck, detector,
tracker and memory are SAM 3.1's own and unchanged. It depends only on `third_party/sam3`.

`codec/` builds the H.264 priors and the per-frame token plan. `model/` holds the encoder and installs it into SAM 3.1.

## Pipeline

| step | code | output |
|---|---|---|
| 1. encode the protocol's frames of each video to H.264 (libx264, I/P only, `-refs 1`, fixed GOP) | `codec/encode.py` | `<video>.mp4` |
| 2. decode, export MVs, reconstruct the residual, map both to the 72×72 token grid (paper Eq. 12) | `codec/extract.py` | `<video>.npz` |
| 3. per-frame plan: Eq. 6 classes, Eq. 11 update link, budget or threshold mode | `codec/schedule.py` (online, on the host) | `FramePlan` |
| 4. the encoder: M-row forward, K/V buffers, query-subset attention, cache warp for matchable tokens | `model/epvit.py` | `EPViT.from_vit(sam_vit, cfg)` |
| 5. install into the SAM 3.1 video pipeline (frame ids, resets) | `model/install.py` | `installed(model, epvit)` |

Steps 1 and 2 run through `data/generate_codec_priors.py`. Evaluation is `experiments/run_epvit.py`.

## Mapping the paper onto SAM 3.1's ViT

* **Tokens:** 1008² input, patch 14, 72×72 tokens, 32 blocks. Windowed blocks use 24×24 windows; blocks 7, 15, 23 and
  31 are global. Each frame is stretched to 1008², so motion vectors are scaled by 72/W and 72/H.
* **Reuse sign convention:** `mv[t, i, j] = (dx, dy)` points to where token (i, j)'s content was in frame t−1. The
  reuse source is (i+dy, j+dx).
* **Redundancy-Aware Attention:** only the M recomputed query rows are kept downstream, so the paper's sparse QKᵀ/AV
  updates become the M queries attending to all buffered keys/values. This is exact for those rows.
* **RoPE:** K/V are buffered before RoPE, and RoPE is applied to the full key buffer, so warped keys are rotated for
  their new position.
* **Repeated frames:** the pipeline encodes frames 0 and 1 twice per prompt session. The second encode is served from
  a small output cache.

### Differences from the paper

* **Training-free.** The paper fine-tunes ViTDet.
* **Absolute position embedding.** SAM's tiled absolute position embedding is not re-applied to warped tokens.
* **Residual.** The residual is reconstructed on decoded luma, because the decoder does not export it.

## Finding so far

At the official 6 fps protocol, SA-V is mostly handheld footage with exposure changes. Over all 146 val videos
(14,907 frames, `crf23_g48`):

* 6% of P-frame tokens are static and 44% matchable;
* 32% are forced back to recompute by Eq. 11;
* the paper's threshold rule recomputes 83% of tokens (median video 90%).

The budget modes are where the compute savings come from.
