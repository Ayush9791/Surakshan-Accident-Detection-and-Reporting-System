# Local training data

The active Keras training pipeline reads source videos from the ignored `vid_dataset/` directory and writes sampled frames plus manifests to `dataset/resnet18_frames/`. The output directory retains its historical name to preserve existing local data; it is not tied to the ResNet18 architecture.

## Audited video sources

- `vid_dataset/archive (1)/`: 981 videos with ACCIDENT event-time metadata. Accident frames are sampled around the annotated event. Pre-event frames are used as negatives only when they end at least two seconds before the annotated event.
- `vid_dataset/TAD-benchmark/`: 127 normal videos are used as negative footage. Its 277 accident videos do not provide event timestamps and are excluded from frame-level supervision.
- `vid_dataset/traffic_eval_ccby/`: six reviewed normal traffic ranges are included in training only. Three videos without reviewed ranges are excluded.
- `vid_dataset/Accident-Bench/`: 29 air-space question-answer videos are not binary road-accident data and are excluded.
- `vid_dataset/TAD/`: empty in the audited collection.

The previous audit inspected 1,423 videos (7.661 GiB), found no exact duplicate videos, flagged 17 near-duplicate candidates for review, and found seven videos with a failed sampled read near the end. These are historical audit results, not a claim that the current local files have been re-audited.

## Preserved older image folders

The existing `dataset/train/`, `dataset/val/`, and `dataset/test/` contain 7,578 valid images. They were kept separate because mapping their clip hashes to source metadata found 11 original-video groups represented across multiple splits. The older `dataset/rebuilt_video/` output is also preserved. The active training script reads only the `frames_manifest.csv` under the configured dataset root.

## Prepare and train

Run in PowerShell from the repository root:

```powershell
.\.venv\Scripts\python.exe preprocess_video_dataset.py --workers 8
.\.venv\Scripts\python.exe train.py
```

The preprocessor uses source-group-aware splits, retains TAD's official test assignment, samples and filters frames, and writes an audit manifest. The Keras trainer uses 128×128 grayscale inputs, class/source-group-balanced sample weights, validation-selected thresholding, and a held-out test evaluation after model selection.

The current configuration is `configs/keras_training.json`. The old ResNet18 and experimental temporal-model configurations are no longer part of the active pipeline.
