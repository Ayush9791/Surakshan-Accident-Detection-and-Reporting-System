# Local training data

The ResNet-18 pipeline reads original videos from the ignored `vid_dataset/` directory and writes newly sampled images and manifests to the ignored `dataset/resnet18_frames/` directory. Source videos and pre-existing images are preserved.

## Audited video sources

- `vid_dataset/archive (1)/`: 981 videos with ACCIDENT event-time metadata. The trainer samples positive frames around each annotated event and may use pre-event frames as negatives only when they end at least two seconds before the annotated event.
- `vid_dataset/TAD-benchmark/`: 127 normal videos are used as genuine negative footage. Its 277 accident videos do not provide event timestamps, so they are excluded from frame-level supervision.
- `vid_dataset/traffic_eval_ccby/`: six reviewed normal time ranges are included in training only. Three videos without reviewed ranges are excluded.
- `vid_dataset/Accident-Bench/`: 29 air-space question-answer videos are not binary road-accident data and are excluded.
- `vid_dataset/TAD/`: empty.

The full audit inspected 1,423 videos (7.661 GiB). It found no exact duplicate videos, 17 near-duplicate candidates retained for review, and seven videos with a failed sampled read near the end. Frame-level extraction failures and quality filters are recorded in `dataset/resnet18_frames/skipped_frames.csv`; excluded videos and reasons are recorded in `dataset/resnet18_frames/preprocessing_exclusions.csv`.

## Preserved legacy images

The existing `dataset/train/`, `dataset/val/`, and `dataset/test/` contain 7,578 valid images. They remain untouched and are not used by the new training run: mapping their clip hashes back to source metadata found 11 original-video groups represented across multiple splits. The older `dataset/rebuilt_video/` output is also preserved; the new model uses only the newly generated `dataset/resnet18_frames/` manifest.

## Prepare and train

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe preprocess_video_dataset.py --workers 8
.\.venv\Scripts\python.exe train.py
```

The preprocessor splits source groups before sampling frames. It keeps TAD's official test videos in the test split, uses a fixed seed, samples at 2 FPS, letterboxes to 224×224, filters unusable and near-duplicate frames, and saves an audit manifest. The ResNet-18 trainer uses class- and source-group-balanced sampling, tunes the decision threshold on validation, and evaluates the held-out test split only after model selection.

The first training run downloads ImageNet ResNet-18 weights from TorchVision if they are not already cached. It does not load the legacy `surakshan_model.pth` checkpoint.
