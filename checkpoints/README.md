# Checkpoint

Download location and instructions for the SAM 3.1 weights.

Dense, EP-ViT, Eventful and Eventful + fused all use the same file. None of them has trained weights of its own. It is not stored in this repository.

| | |
|---|---|
| Source | Hugging Face [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) (gated, manual approval) |
| File | `sam3.1_multiplex.pt` (3.5 GB) |
| Place at | `checkpoints/sam3.1_multiplex.pt` |
| SHA-256 | `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6` |

1. Request access on the model page and wait for approval.
2. Log in and download:

   ```bash
   hf auth login
   hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir checkpoints
   ```

3. Verify:

   ```bash
   echo "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6  checkpoints/sam3.1_multiplex.pt" | sha256sum -c -
   ```

The weights are covered by the SAM License (`third_party/sam3/LICENSE`).
