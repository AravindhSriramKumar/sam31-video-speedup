# Data

The frozen evaluation pairs, the sample video and the scripts that fetch and prepare the data. The dataset frames and codec priors are generated here and are not stored in the repository.

| File | Contents |
|---|---|
| `eval_pairs.json` | The 189 scored (video, phrase) pairs of the 24 fps protocol |
| `subsets.yaml` | The 10 `normal` and 10 `high_motion` SA-V validation videos |
| `sample_video.mp4` | Short clip for trying a runner without the dataset |
| `download_saco_eval.sh` | Downloads the annotations and frames into `data/saco_sav/` |
| `prepare_saco_eval.py` | Fetches and validates the evaluation frames |
| `saco_data.py` | Helpers for annotations, frame folders and filtered ground truth |
| `generate_codec_priors.py` | Builds the H.264 priors that EP-ViT needs |
