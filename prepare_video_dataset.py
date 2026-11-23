"""Extract labeled frames into source-grouped train/val/test folders.

The requested 75:15:15 proportions are normalized to 5:1:1 (about 71.4%,
14.3%, 14.3%). Source videos and sibling segments stay in one split. Each
video contributes event frames; sufficiently long clips also contribute
pre-impact normal frames at least two seconds before impact.

Run from the project root:
    python prepare_video_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from video_preprocessing import letterbox_rgb_frame


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "dataset" / "ACCIDENT"
OUTPUT_ROOT = PROJECT_ROOT / "dataset"
STAGING_ROOT = OUTPUT_ROOT / ".accident_frames_build"
SEED = 42
IMAGE_SIZE = 224
EVENT_OFFSETS = (-0.8, -0.4, 0.0, 0.4, 0.8)
NORMAL_OFFSETS = (0.2, 0.6, 1.0, 1.4, 1.8)
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("Accident", "Non Accident")
SPLIT_WEIGHTS = {"train": 75, "val": 15, "test": 15}
GENERATED_FRAME = re.compile(r"^(real|synthetic)_([0-9a-f]{14})_([pn])(\d{2})\.jpg$", re.IGNORECASE)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def source_video_id(path: str) -> str:
    """Collapse segment suffixes so sibling clips cannot cross splits."""
    return re.sub(r"_\d+$", "", Path(path).stem)


def row_relative_path(row: dict[str, str], synthetic: bool) -> str:
    return row["rgb_path"] if synthetic else row["path"]


def resolve_video_path(source_root: Path, row: dict[str, str], synthetic: bool) -> Path:
    relative = row_relative_path(row, synthetic)
    direct = source_root / relative
    if direct.is_file():
        return direct
    # Some local ACCIDENT archives flatten the video files while retaining
    # paths with real_videos/ or synthetic_videos/ prefixes in the metadata.
    return source_root / Path(relative).name


def select_splits(source_root: Path, real_rows, synthetic_rows, seed: int):
    available = []
    missing = {"real": 0, "synthetic": 0}
    for row in real_rows:
        if resolve_video_path(source_root, row, False).is_file():
            available.append((row, False))
        else:
            missing["real"] += 1
    for row in synthetic_rows:
        if resolve_video_path(source_root, row, True).is_file():
            available.append((row, True))
        else:
            missing["synthetic"] += 1

    group_keys = sorted({
        ("synthetic" if synthetic else "real", source_video_id(row_relative_path(row, synthetic)))
        for row, synthetic in available
    })
    random.Random(seed).shuffle(group_keys)
    total_weight = sum(SPLIT_WEIGHTS.values())
    val_count = round(len(group_keys) * SPLIT_WEIGHTS["val"] / total_weight)
    test_count = round(len(group_keys) * SPLIT_WEIGHTS["test"] / total_weight)
    train_count = len(group_keys) - val_count - test_count
    group_split = {}
    for key in group_keys[:train_count]:
        group_split[key] = "train"
    for key in group_keys[train_count:train_count + val_count]:
        group_split[key] = "val"
    for key in group_keys[train_count + val_count:]:
        group_split[key] = "test"

    assigned = {split: [] for split in SPLITS}
    frame_split = {}
    for row, synthetic in available:
        relative = row_relative_path(row, synthetic)
        source_name = "synthetic" if synthetic else "real"
        group_key = (source_name, source_video_id(relative))
        split = group_split[group_key]
        assigned[split].append((row, synthetic))
        digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:14]
        frame_split[(source_name, digest)] = split

    group_counts = {split: sum(value == split for value in group_split.values()) for split in SPLITS}
    return assigned, frame_split, group_counts, missing


def _read_frame_at(capture, seconds: float, duration: float):
    seconds = min(max(0.0, seconds), max(0.0, duration - 0.02))
    capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
    ok, frame = capture.read()
    if not ok:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return letterbox_rgb_frame(rgb, IMAGE_SIZE)


def _save_frame(frame_rgb: np.ndarray, path: Path) -> bool:
    return cv2.imwrite(
        str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def _normal_window_start(impact_time: float) -> float | None:
    # The window is 2.4 seconds long, then at least 2 seconds of margin remains.
    # This matches the current video trainer's pre-impact negative rule.
    if impact_time < 4.4:
        return None
    return max(0.0, (impact_time - 4.4) / 2.0)


def extract_video(item):
    source_root, split, row, synthetic, staging_root = item
    relative = row_relative_path(row, synthetic)
    video_path = resolve_video_path(source_root, row, synthetic)
    if not video_path.is_file():
        return {"split": split, "relative": relative, "error": "missing"}

    try:
        duration = float(row["duration"])
        impact = float(row["accident_time"])
    except (KeyError, TypeError, ValueError):
        return {"split": split, "relative": relative, "error": "bad_metadata"}
    if not np.isfinite(duration) or not np.isfinite(impact) or duration <= 0:
        return {"split": split, "relative": relative, "error": "bad_duration_or_impact"}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return {"split": split, "relative": relative, "error": "unreadable"}

    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:14]
    safe_source = "synthetic" if synthetic else "real"
    saved = {"Accident": 0, "Non Accident": 0}
    failures = 0
    try:
        event_times = [min(max(0.0, impact + offset), max(0.0, duration - 0.02))
                       for offset in EVENT_OFFSETS]
        for frame_index, timestamp in enumerate(event_times):
            frame = _read_frame_at(capture, timestamp, duration)
            if frame is None:
                failures += 1
                continue
            output = staging_root / split / "Accident" / f"{safe_source}_{digest}_p{frame_index:02}.jpg"
            if _save_frame(frame, output):
                saved["Accident"] += 1
            else:
                failures += 1

        normal_start = _normal_window_start(impact)
        if normal_start is not None:
            normal_end = min(normal_start + 2.0, impact - 2.0, duration - 0.02)
            if normal_end >= normal_start:
                for frame_index, offset in enumerate(NORMAL_OFFSETS):
                    timestamp = min(normal_start + offset, normal_end)
                    frame = _read_frame_at(capture, timestamp, duration)
                    if frame is None:
                        failures += 1
                        continue
                    output = staging_root / split / "Non Accident" / f"{safe_source}_{digest}_n{frame_index:02}.jpg"
                    if _save_frame(frame, output):
                        saved["Non Accident"] += 1
                    else:
                        failures += 1
    finally:
        capture.release()

    return {
        "split": split,
        "relative": relative,
        "source": safe_source,
        "saved": saved,
        "failures": failures,
        "no_negative_window": _normal_window_start(impact) is None,
    }


def publish_staging(staging_root: Path):
    # Append processed images to the requested class folders without replacing
    # existing image files. Generated filenames are source-hash unique.
    added = {split: {name: 0 for name in CLASS_NAMES} for split in SPLITS}
    duplicates = 0
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            incoming = staging_root / split / class_name
            destination = OUTPUT_ROOT / split / class_name
            destination.mkdir(parents=True, exist_ok=True)
            if not incoming.is_dir():
                continue
            for source_file in incoming.iterdir():
                target = destination / source_file.name
                if target.exists():
                    duplicates += 1
                    continue
                shutil.move(str(source_file), str(target))
                added[split][class_name] += 1
    shutil.rmtree(staging_root, ignore_errors=True)
    return added, duplicates


def redistribute_existing_frames(frame_split):
    """Move previously generated frames to their new source-group split."""
    moved = 0
    unknown = 0
    for current_split in SPLITS:
        for class_name in CLASS_NAMES:
            folder = OUTPUT_ROOT / current_split / class_name
            if not folder.is_dir():
                continue
            for image_path in folder.glob("*.jpg"):
                match = GENERATED_FRAME.match(image_path.name)
                if not match:
                    continue
                source_name, digest, _, _ = match.groups()
                target_split = frame_split.get((source_name.lower(), digest.lower()))
                if target_split is None:
                    unknown += 1
                    continue
                if target_split == current_split:
                    continue
                target = OUTPUT_ROOT / target_split / class_name / image_path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite duplicate generated frame: {target}"
                    )
                shutil.move(str(image_path), str(target))
                moved += 1
    return moved, unknown


def summarize_generated_frames():
    frames = {split: {name: 0 for name in CLASS_NAMES} for split in SPLITS}
    clips = {split: {name: set() for name in CLASS_NAMES} for split in SPLITS}
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            folder = OUTPUT_ROOT / split / class_name
            if not folder.is_dir():
                continue
            for image_path in folder.glob("*.jpg"):
                match = GENERATED_FRAME.match(image_path.name)
                if not match:
                    continue
                source_name, digest, kind, _ = match.groups()
                frames[split][class_name] += 1
                clips[split][class_name].add((source_name.lower(), digest.lower(), kind.lower()))
    clip_counts = {
        split: {class_name: len(clips[split][class_name]) for class_name in CLASS_NAMES}
        for split in SPLITS
    }
    return frames, clip_counts


def reassign_existing_by_clip_hash(seed: int):
    """Rebalance already extracted groups when source metadata is unavailable."""
    grouped_files = defaultdict(list)
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            folder = OUTPUT_ROOT / split / class_name
            if not folder.is_dir():
                continue
            for image_path in folder.glob("*.jpg"):
                match = GENERATED_FRAME.match(image_path.name)
                if match:
                    source_name, digest, _, _ = match.groups()
                    grouped_files[(source_name.lower(), digest.lower())].append(image_path)

    group_keys = sorted(grouped_files)
    if not group_keys:
        raise RuntimeError("No extracted ACCIDENT frame groups were found to split")
    random.Random(seed).shuffle(group_keys)
    total_weight = sum(SPLIT_WEIGHTS.values())
    val_count = round(len(group_keys) * SPLIT_WEIGHTS["val"] / total_weight)
    test_count = round(len(group_keys) * SPLIT_WEIGHTS["test"] / total_weight)
    train_count = len(group_keys) - val_count - test_count
    assignments = {}
    for key in group_keys[:train_count]:
        assignments[key] = "train"
    for key in group_keys[train_count:train_count + val_count]:
        assignments[key] = "val"
    for key in group_keys[train_count + val_count:]:
        assignments[key] = "test"

    moved = 0
    for key, paths in grouped_files.items():
        target_split = assignments[key]
        for image_path in paths:
            current_split = image_path.parents[1].name
            if current_split == target_split:
                continue
            class_name = image_path.parent.name
            target = OUTPUT_ROOT / target_split / class_name / image_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite duplicate frame: {target}")
            shutil.move(str(image_path), str(target))
            moved += 1

    frames, clips = summarize_generated_frames()
    group_counts = {split: sum(value == split for value in assignments.values()) for split in SPLITS}
    total_groups = len(assignments)
    weight_total = sum(SPLIT_WEIGHTS.values())
    report = {
        "source": "previously extracted frame files; ACCIDENT metadata link unavailable",
        "seed": seed,
        "requested_split_ratio": SPLIT_WEIGHTS,
        "normalized_split_percent": {
            split: round(100 * SPLIT_WEIGHTS[split] / weight_total, 2)
            for split in SPLITS
        },
        "actual_grouping_unit": "metadata path hash; positive and pre-impact frames from the same clip stay together",
        "source_video_sibling_grouping": "unverified because the source metadata is unavailable",
        "clip_groups": group_counts,
        "actual_clip_group_percent": {
            split: round(100 * group_counts[split] / total_groups, 2)
            for split in SPLITS
        },
        "frames": frames,
        "clips": clips,
        "generated_frames_reassigned": moved,
    }
    write_manifest(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=DEFAULT_SOURCE,
                        help="ACCIDENT folder containing its metadata CSV files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent video readers (default: 4)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Seed for source-grouped split")
    parser.add_argument("--reassign-only", action="store_true",
                        help="Move existing generated frames to the configured ratio without decoding videos")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.reassign_only:
        report = reassign_existing_by_clip_hash(args.seed)
        print(json.dumps(report, indent=2), flush=True)
        return

    source_root = args.source.resolve()
    for filename in ("metadata-real.csv", "metadata-synthetic.csv"):
        if not (source_root / filename).is_file():
            raise FileNotFoundError(f"Missing {source_root / filename}")

    real_rows = read_rows(source_root / "metadata-real.csv")
    synthetic_rows = read_rows(source_root / "metadata-synthetic.csv")
    assigned, frame_split, group_counts, missing = select_splits(
        source_root, real_rows, synthetic_rows, args.seed
    )

    moved_frames, unknown_frames = redistribute_existing_frames(frame_split)

    if args.reassign_only:
        frames, clips = summarize_generated_frames()
        report = build_split_report(
            source_root, args.seed, group_counts, assigned, missing, frames, clips,
            moved_frames=moved_frames, unknown_frames=unknown_frames,
        )
        write_manifest(report)
        print(json.dumps(report, indent=2), flush=True)
        return

    if STAGING_ROOT.exists():
        shutil.rmtree(STAGING_ROOT)
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            (STAGING_ROOT / split / class_name).mkdir(parents=True, exist_ok=True)

    items = []
    for split, rows in assigned.items():
        for row, synthetic in rows:
            items.append((source_root, split, row, synthetic, STAGING_ROOT))
    report = build_split_report(
        source_root, args.seed, group_counts, assigned, missing,
        {split: {name: 0 for name in CLASS_NAMES} for split in SPLITS},
        {split: {name: 0 for name in CLASS_NAMES} for split in SPLITS},
        moved_frames=moved_frames, unknown_frames=unknown_frames,
    )

    print(f"Extracting {len(items):,} annotated videos into staged train/val/test folders...", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_video, item) for item in items]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if "error" in result:
                report["errors"].append({"split": result["split"], "path": result["relative"], "error": result["error"]})
            else:
                report["videos_without_clear_negative"] += int(result["no_negative_window"])
                for class_name, count in result["saved"].items():
                    report["frames"][result["split"]][class_name] += count
            if done % 100 == 0 or done == len(items):
                print(f"Processed {done:,}/{len(items):,} videos", flush=True)

    if report["errors"]:
        print(f"Warning: {len(report['errors'])} videos had missing or unreadable inputs; see summary after extraction.", flush=True)

    report["unreadable_videos"] = sum("error" in result for result in results)
    added, duplicates = publish_staging(STAGING_ROOT)
    frames, clips = summarize_generated_frames()
    report["frames"] = frames
    report["clips"] = clips
    report["new_frames_added"] = added
    report["duplicate_files_skipped"] = duplicates
    write_manifest(report)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Frame dataset ready: {OUTPUT_ROOT}", flush=True)


def build_split_report(source_root, seed, group_counts, assigned, missing,
                       frames, clips, moved_frames=0, unknown_frames=0):
    total_groups = sum(group_counts.values())
    weight_total = sum(SPLIT_WEIGHTS.values())
    return {
        "source": str(source_root),
        "seed": seed,
        "requested_split_ratio": SPLIT_WEIGHTS,
        "normalized_split_percent": {
            split: round(100 * SPLIT_WEIGHTS[split] / weight_total, 2)
            for split in SPLITS
        },
        "actual_source_video_groups": group_counts,
        "actual_source_video_group_percent": {
            split: round(100 * group_counts[split] / total_groups, 2) if total_groups else 0
            for split in SPLITS
        },
        "metadata_video_rows_assigned": {
            split: len(assigned[split]) for split in SPLITS
        },
        "missing_metadata_videos": missing,
        "image_size": IMAGE_SIZE,
        "frames_per_event_video": len(EVENT_OFFSETS),
        "frames_per_normal_video": len(NORMAL_OFFSETS),
        "frames": frames,
        "clips": clips,
        "errors": [],
        "generated_frames_reassigned": moved_frames,
        "generated_frames_without_matching_metadata": unknown_frames,
        "videos_without_clear_negative": sum(
            1 for split_rows in assigned.values()
            for row, synthetic in split_rows
            if _normal_window_start(float(row["accident_time"])) is None
        ),
    }


def write_manifest(report):
    manifest_path = OUTPUT_ROOT / "frame_split_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Extraction stopped; saved images remain in the split folders. Rerun to finish or rebalance them.", file=sys.stderr)
        raise
