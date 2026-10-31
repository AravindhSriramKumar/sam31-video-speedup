"""EP-ViT for SAM 3.1: a drop-in replacement of SAM 3.1's ViT trunk that recomputes only the tokens H.264 motion
vectors and residuals mark as changed (Qiu et al., "EP-ViT: Leveraging Encoding Priors to Eliminate Spatio-Temporal
Redundancy in Vision Transformer", ECAI 2025). Depends only on third_party/sam3. See epvit/README.md.
"""
GRID = 72                          # SAM 3.1 ViT: 1008 / 14
PATCH = 14
IMG = GRID * PATCH
