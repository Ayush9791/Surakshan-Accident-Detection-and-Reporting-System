# Training data

## Primary dataset: ACCIDENT

`ACCIDENT/` is a local directory junction to `C:\Users\AYUSH\Downloads\archive (1)`. The videos are not copied; keep the source folder in place. This avoids a second copy of about 15.5 GB.

The metadata files list:

- 2,027 real surveillance clips with `split_geo_aware` assignments in `metadata-real.csv`.
- 2,211 synthetic clips listed in `metadata-synthetic.csv`.
- At extraction time, 981 real videos matched metadata paths by filename. The other 1,046 real clips and all 2,211 synthetic videos were unavailable. The archive link under Downloads has since become unavailable.
- The frame folders use normalized 75:15:15 proportions (71.4:14.3:14.3). The extracted clips were reassigned by their saved path hashes because the source metadata is no longer accessible; accident and normal sequences from one clip remain together, but sibling segments from the same original video cannot be verified. This is not the official geo-aware test partition.

The trainer uses each clip's `accident_time` to sample a positive window and a pre-impact normal window where there is enough separation. It omits ambiguous early negatives. Very short clips remain positive examples; the trainer samples frames across their available duration. Project page and annotations: https://accidentbench.github.io/.

## Additional dataset researched: TAD

The closest additional public source found is [TAD: A Large-scale Benchmark for Traffic Accidents Detection from Video Surveillance](https://github.com/UnicomAI/UnicomBenchmark/tree/main/TADBench), which is specifically highway surveillance video and includes both accident and normal labels. Its repository links a Google Drive archive. The README's stated total (344) conflicts with its class counts (277 + 127 = 404); the paper reports 404 videos and an official development/test split. Check the archive's current contents and terms before combining it with ACCIDENT, and keep its official test set untouched. It has **not** been downloaded or added to training: this machine's download connection could not resolve `drive.google.com`, and the public README does not state a dataset license. Paper: https://arxiv.org/abs/2209.12386.

## Optional normal-traffic supplement

`traffic_eval_ccby/` contains a small subset of fixed-camera road-monitoring videos from Tiago Silva's [Zenodo dataset](https://doi.org/10.5281/zenodo.14041920), licensed CC BY 4.0. `verified_normal_videos.txt` specifies the reviewed file/time ranges. The footage was visually spot-checked for collisions and is training-only; it is not a held-out benchmark. Earlier training attempts with these clips did not improve the ACCIDENT geo-aware test score, so the trainer excludes them by default. To include them, pass `--include-reviewed-normal-data`.

## Folder layout

- `ACCIDENT/` is the required source dataset for the current video trainer.
- `train/`, `val/`, and `test/` hold image-only frame-classification data. `prepare_video_dataset.py` appends processed images to their existing `Accident` and `Non Accident` class folders and preserves existing images. The requested 75:15:15 proportions are normalized to approximately 71.4:14.3:14.3 because they sum to 105%. `python prepare_video_dataset.py --reassign-only` reapportions already extracted groups without needing the source videos; a full extraction requires the ACCIDENT metadata and videos. Counts and split details are in `frame_split_manifest.json`. Pre-existing videos and other non-image files are kept in the ignored project-level `source_videos_archive/`. `train.py` trains a temporal MobileNet-GRU from the grouped generated frames and ignores unrelated existing images in those folders.
- `traffic_eval_ccby/` is optional and used only with `--include-reviewed-normal-data`.
- `cache/` contains the active decoded-clip cache and extraction chunks. Keep it if you want extraction to resume without re-reading and decoding all videos. It is generated locally and ignored by Git.

## Start training

Prepare the frames once, then train from the project root:

```powershell
python prepare_video_dataset.py
.venv\Scripts\python.exe train.py
```

The trainer selects its checkpoint and threshold using validation sequences, evaluates the held-out test split once, and saves `surakshan_model.pth`. The checkpoint format is supported by the live predictor.
