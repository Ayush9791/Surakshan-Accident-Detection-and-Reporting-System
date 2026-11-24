"""Inspect and preprocess the labeled surveillance video collections."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".wmv", ".flv", ".webm",
    ".mpg", ".mpeg", ".3gp", ".ts",
}
SPLITS = ("train", "val", "test")
CLASS_NAMES = ("Non_Accident", "Accident")
TAD_PATH_PART = "tad-benchmark"
ACCIDENT_PATH_PART = "archive (1)"
TRAFFIC_PATH_PART = "traffic_eval_ccby"


def load_config(path: Path):
    config = json.loads(path.read_text(encoding="utf-8"))
    for section in ("paths", "preprocessing", "splits", "training"):
        if section not in config:
            raise ValueError(f"Config is missing section: {section}")
    return config


def _write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_jsonl(path: Path):
    if not path.is_file():
        return []
    items = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL {path}, line {line_number}") from error
    return items


def _source_video_id(path: str):
    """Merge numbered ACCIDENTBench segments from the same source upload."""
    return re.sub(r"_\d+$", "", Path(path).stem).lower()


def _read_accidentbench_metadata(video_root: Path):
    metadata_path = video_root / "archive (1)" / "metadata-real.csv"
    by_basename = {}
    missing_file = None
    if not metadata_path.is_file():
        return by_basename, f"Missing ACCIDENTBench labels: {metadata_path}"
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            filename = Path(row.get("path", "")).name.lower()
            if filename:
                by_basename[filename] = row
    return by_basename, missing_file


def _read_reviewed_traffic(video_root: Path):
    list_path = video_root / "traffic_eval_ccby" / "verified_normal_videos.txt"
    ranges = defaultdict(list)
    if not list_path.is_file():
        return ranges
    for line_number, raw_line in enumerate(list_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = raw_line.strip()
        if not text or text.startswith("#"):
            continue
        fields = [part.strip() for part in text.split(",")]
        if len(fields) < 3:
            raise ValueError(f"Malformed reviewed-normal entry at {list_path}:{line_number}")
        start, end = float(fields[1]), float(fields[2])
        if start < 0 or end <= start:
            raise ValueError(f"Invalid time range at {list_path}:{line_number}")
        ranges[fields[0].lower()].append((start, end))
    return ranges


def discover_video_records(video_root: Path):
    video_root = video_root.resolve()
    accident_rows, metadata_error = _read_accidentbench_metadata(video_root)
    reviewed_ranges = _read_reviewed_traffic(video_root)
    if metadata_error:
        raise FileNotFoundError(metadata_error)

    records = []
    for path in sorted(video_root.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        relative = path.relative_to(video_root).as_posix()
        parts = [part.casefold() for part in Path(relative).parts]
        record = {
            "path": path,
            "relative_path": relative,
            "filename": path.name,
            "source": "unresolved",
            "label": None,
            "source_group_id": None,
            "official_split": "",
            "accident_time": None,
            "normal_ranges": [],
            "label_confidence": "unresolved",
            "training_only": False,
            "metadata_type": "",
        }

        if TAD_PATH_PART in parts:
            label_dir = path.parent.name.casefold()
            if label_dir == "normal":
                record.update(source="TAD", label="Non_Accident", label_confidence="folder_label")
            elif label_dir.startswith("accident_") or label_dir == "accident":
                record.update(source="TAD", label="Accident", label_confidence="folder_label")
            else:
                record.update(source="TAD", label_confidence="unresolved_folder_label")
            record["official_split"] = "test" if "test" in parts else "train" if "train" in parts else ""
            record["source_group_id"] = f"tad:{path.stem.casefold()}"

        elif ACCIDENT_PATH_PART in parts:
            annotation = accident_rows.get(path.name.casefold())
            record.update(source="ACCIDENTBench", source_group_id=f"accidentbench:{_source_video_id(annotation['path'])}" if annotation else None)
            if annotation:
                try:
                    record.update(
                        label="Accident",
                        accident_time=float(annotation["accident_time"]),
                        label_confidence="annotated_accident_time",
                        metadata_type=annotation.get("type", ""),
                        official_split=annotation.get("split_geo_aware", ""),
                    )
                except (KeyError, TypeError, ValueError):
                    record["label_confidence"] = "invalid_metadata"

        elif TRAFFIC_PATH_PART in parts:
            record["source_group_id"] = f"traffic:{path.name.casefold()}"
            ranges = reviewed_ranges.get(path.name.casefold(), [])
            if ranges:
                record.update(
                    source="ReviewedTraffic",
                    label="Non_Accident",
                    normal_ranges=ranges,
                    label_confidence="reviewed_time_ranges",
                    training_only=True,
                )
            else:
                record.update(source="ReviewedTraffic", label_confidence="no_reviewed_range")

        if record["source_group_id"] is None and record["label"] is not None:
            record["source_group_id"] = f"{record['source'].casefold()}:{path.stem.casefold()}"
        records.append(record)
    return records, {
        "accidentbench_metadata_rows": len(accident_rows),
        "reviewed_traffic_files": len(reviewed_ranges),
        "unresolved_traffic_videos": sum(
            1 for item in records if item["source"] == "ReviewedTraffic" and item["label"] is None
        ),
    }


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phash64(frame):
    gray = cv2.cvtColor(cv2.resize(frame, (32, 32)), cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(gray.astype(np.float32))[:8, :8].flatten()[1:]
    median = float(np.median(dct))
    value = 0
    for bit in dct:
        value = (value << 1) | int(bit > median)
    return value


def _inspect_one(record):
    path = record["path"]
    item = dict(record)
    item.pop("path", None)
    item["file_size_bytes"] = path.stat().st_size
    item["sha256"] = _sha256(path)
    capture = cv2.VideoCapture(str(path))
    item["opened"] = bool(capture.isOpened())
    fps = capture.get(cv2.CAP_PROP_FPS) if item["opened"] else 0.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) if item["opened"] else 0.0
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH) if item["opened"] else 0.0
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) if item["opened"] else 0.0
    item["fps"] = float(fps) if math.isfinite(fps) else 0.0
    item["frame_count"] = int(frame_count) if math.isfinite(frame_count) and frame_count > 0 else 0
    item["width"] = int(width) if math.isfinite(width) and width > 0 else 0
    item["height"] = int(height) if math.isfinite(height) and height > 0 else 0
    item["duration_seconds"] = (
        item["frame_count"] / item["fps"] if item["fps"] > 0 else 0.0
    )
    sample_positions = sorted(set([
        0,
        max(0, item["frame_count"] // 2),
        max(0, int(item["frame_count"] * 0.90)),
    ]))
    sample_reads, signatures = [], []
    if item["opened"] and item["frame_count"]:
        for position in sample_positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = capture.read()
            sample_reads.append(bool(ok))
            if ok:
                signatures.append(_phash64(frame))
    capture.release()
    item["sample_reads"] = sample_reads
    item["signature"] = signatures
    item["read_status"] = "ok" if item["opened"] and sample_reads and all(sample_reads) else "sample_read_warning"
    if not item["opened"]:
        item["read_status"] = "cannot_open"
    elif item["fps"] <= 0 or item["frame_count"] <= 0 or not sample_reads or not sample_reads[0]:
        item["read_status"] = "unreadable_or_missing_video_stream"
    return item


def inspect_videos(records, workers=8, progress=True):
    inspected = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_inspect_one, record) for record in records]
        for done, future in enumerate(as_completed(futures), start=1):
            inspected.append(future.result())
            if progress and (done % 100 == 0 or done == len(futures)):
                print(f"Inspected {done:,}/{len(futures):,} videos", flush=True)
    inspected.sort(key=lambda item: item["relative_path"].casefold())
    return inspected


def _hamming(left, right):
    return (int(left) ^ int(right)).bit_count()


def mark_exact_duplicates(records):
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["sha256"]].append(record)
    duplicates = []
    for sha, group in by_hash.items():
        if len(group) <= 1:
            continue
        labels = {item["label"] for item in group}
        ordered = sorted(group, key=lambda item: item["relative_path"].casefold())
        for item in ordered:
            item["duplicate_sha256"] = sha
            item["duplicate_of"] = ordered[0]["relative_path"]
            item["duplicate_label_conflict"] = len(labels) > 1
            if len(labels) > 1:
                item["source_label_before_duplicate_review"] = item.get("label")
                item["label"] = None
                item["label_confidence"] = "exact_duplicate_label_conflict"
            elif item is not ordered[0]:
                item["source_group_id_before_duplicate_merge"] = item.get("source_group_id")
                item["source_group_id"] = ordered[0].get("source_group_id")
        duplicates.append({
            "sha256": sha,
            "files": [item["relative_path"] for item in ordered],
            "labels": sorted(str(value) for value in labels),
            "label_conflict": len(labels) > 1,
        })
    return duplicates


def near_duplicate_candidates(records, max_hamming=4):
    candidates = []
    eligible = [item for item in records if len(item.get("signature", [])) == 3]
    # Strict three-timepoint signature match to avoid flagging merely similar CCTV scenes.
    for index, left in enumerate(eligible):
        for right in eligible[index + 1:]:
            if left["sha256"] == right["sha256"]:
                continue
            distances = [_hamming(a, b) for a, b in zip(left["signature"], right["signature"])]
            if max(distances) <= max_hamming:
                candidates.append({
                    "path_a": left["relative_path"],
                    "path_b": right["relative_path"],
                    "source_a": left["source"],
                    "source_b": right["source"],
                    "frame_phash_hamming": json.dumps(distances),
                    "review_status": "candidate_only_not_removed",
                })
    return candidates


def _group_records(records):
    groups = defaultdict(list)
    for record in records:
        if record.get("label") is None or record.get("duplicate_label_conflict"):
            continue
        if record.get("duplicate_of") and record["duplicate_of"] != record["relative_path"]:
            continue
        groups[record["source_group_id"]].append(record)
    return groups


def assign_group_splits(records, split_config, seed=42):
    groups = _group_records(records)
    labels_by_group = {}
    for group_id, items in groups.items():
        labels = {item["label"] for item in items}
        if len(labels) != 1:
            for item in items:
                item["split"] = "unresolved"
                item["split_reason"] = "source_group_has_conflicting_labels"
            continue
        labels_by_group[group_id] = next(iter(labels))

    assignments = {}
    rng = random.Random(seed)
    lock_tad_test = bool(split_config.get("preserve_tad_official_test", True))
    for label in CLASS_NAMES:
        class_groups = [key for key, value in labels_by_group.items() if value == label]
        locked = []
        candidates = []
        for group_id in class_groups:
            items = groups[group_id]
            is_training_only = all(item.get("training_only", False) for item in items)
            is_locked = (
                lock_tad_test and any(
                    item["source"] == "TAD" and item["official_split"] == "test"
                    for item in items
                )
            )
            if is_locked:
                locked.append(group_id)
            elif not is_training_only:
                candidates.append(group_id)
            else:
                assignments[group_id] = "train"

        n_core = len(locked) + len(candidates)
        if not n_core:
            continue
        val_target = round(n_core * float(split_config["val"]))
        test_target = round(n_core * float(split_config["test"]))
        train_target = n_core - val_target - test_target
        test_target = max(test_target, len(locked))
        if train_target < 0:
            raise ValueError(f"Invalid split targets for class {label}")
        rng.shuffle(candidates)
        extra_test = test_target - len(locked)
        test_groups = locked + candidates[:extra_test]
        val_start = extra_test
        val_groups = candidates[val_start:val_start + val_target]
        train_groups = candidates[val_start + val_target:]
        if len(train_groups) + len(test_groups) + len(val_groups) != n_core:
            # Keep the realized groups intact and assign any rounding remainder to train.
            assigned = set(test_groups) | set(val_groups) | set(train_groups)
            train_groups.extend(key for key in candidates if key not in assigned)
        for group_id in train_groups:
            assignments[group_id] = "train"
        for group_id in val_groups:
            assignments[group_id] = "val"
        for group_id in test_groups:
            assignments[group_id] = "test"

    for record in records:
        group_id = record.get("source_group_id")
        if group_id in assignments and record.get("label") is not None:
            record["split"] = assignments[group_id]
            record["split_reason"] = (
                "TAD official test preserved" if record["source"] == "TAD"
                and record["official_split"] == "test" else "source_group_stratified"
            )
        elif record.get("label") is None:
            record["split"] = "unresolved"
            record["split_reason"] = record.get("label_confidence", "unresolved_label")
    return assignments, groups


def _window_specs(record, config):
    duration = float(record.get("duration_seconds") or 0.0)
    if duration <= 0:
        return []
    pre = float(config["accident_pre_seconds"])
    post = float(config["accident_post_seconds"])
    normal_length = float(config["normal_window_seconds"])
    margin = float(config["normal_impact_margin_seconds"])
    sequence_seconds = float(config["sequence_seconds"])

    if record["source"] == "ACCIDENTBench":
        impact = record.get("accident_time")
        if impact is None or not math.isfinite(float(impact)):
            return []
        impact = min(max(0.0, float(impact)), max(0.0, duration - 0.02))
        start = max(0.0, impact - pre)
        end = min(duration - 0.02, impact + post)
        windows = [{
            "kind": "accident_event", "class_name": "Accident",
            "start_seconds": start, "end_seconds": max(start + 0.02, end),
            "protected_positive": True,
        }]
        if impact >= normal_length + margin:
            normal_start = max(0.0, (impact - normal_length - margin) / 2.0)
            normal_end = min(normal_start + normal_length, impact - margin, duration - 0.02)
            if normal_end > normal_start:
                windows.append({
                    "kind": "pre_impact_normal", "class_name": "Non_Accident",
                    "start_seconds": normal_start, "end_seconds": normal_end,
                    "protected_positive": False,
                })
        return windows

    ranges = record.get("normal_ranges") or [(0.0, duration)]
    candidates = []
    for range_index, (range_start, range_end) in enumerate(ranges):
        start = max(0.0, float(range_start))
        end = min(duration - 0.02, float(range_end))
        if end <= start:
            continue
        span = end - start
        if span <= sequence_seconds:
            candidates.append((start, end, range_index))
            continue
        step = sequence_seconds
        position = start
        while position + 0.25 < end:
            window_end = min(position + sequence_seconds, end)
            candidates.append((position, window_end, range_index))
            position += step

    fps = float(config["sampling_fps"])
    frames_per_sequence = max(1, int(math.ceil(sequence_seconds * fps)))
    max_windows = max(1, int(config["max_frames_per_video"]) // frames_per_sequence)
    max_windows = min(max_windows, int(config["max_sequences_per_video"]))
    if len(candidates) > max_windows:
        positions = np.rint(np.linspace(0, len(candidates) - 1, max_windows)).astype(int)
        candidates = [candidates[index] for index in sorted(set(positions.tolist()))]

    output = []
    for index, (start, end, range_index) in enumerate(candidates):
        output.append({
            "kind": f"video_window_{range_index}_{index}",
            "class_name": record["label"],
            "start_seconds": start,
            "end_seconds": end,
            "protected_positive": record["label"] == "Accident",
        })
    return output


def _sample_times(start, end, fps):
    if end <= start:
        return [max(0.0, start)]
    count = max(1, int(math.ceil((end - start) * fps)))
    return [min(start + index / fps, end - 0.01) for index in range(count)]


def _letterbox(frame_bgr, size):
    height, width = frame_bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("empty decoded frame")
    scale = min(size / width, size / height)
    out_w = max(1, min(size, round(width * scale)))
    out_h = max(1, min(size, round(height * scale)))
    resized = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    x = (size - out_w) // 2
    y = (size - out_h) // 2
    canvas[y:y + out_h, x:x + out_w] = resized
    return canvas


def _append_skip(skips, record, sequence, time_seconds, reason, blur=None, distance=None):
    skips.append({
        "source_video": record["relative_path"],
        "video_id": record["video_id"],
        "sequence_id": sequence["sequence_id"],
        "time_seconds": round(float(time_seconds), 4),
        "reason": reason,
        "laplacian_variance": None if blur is None else round(float(blur), 4),
        "phash_distance": distance,
    })


def _process_sequence(capture, record, sequence, dataset_root, config):
    fps = float(config["sampling_fps"])
    size = int(config["image_size"])
    blur_threshold = float(config["min_laplacian_variance"])
    duplicate_distance = int(config["phash_duplicate_distance"])
    times = _sample_times(sequence["start_seconds"], sequence["end_seconds"], fps)
    decoded = []
    skips = []

    for frame_index, timestamp in enumerate(times):
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            _append_skip(skips, record, sequence, timestamp, "frame_decode_failed")
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.std())
        phash = _phash64(frame)
        decoded.append({
            "time": timestamp,
            "frame_index": frame_index,
            "frame": _letterbox(frame, size),
            "blur": blur,
            "brightness_std": brightness,
            "phash": phash,
        })

    if not decoded:
        return [], skips, "no_decodable_frames"

    kept = []
    previous = None
    for frame in decoded:
        if not sequence["protected_positive"] and frame["brightness_std"] < 1.5:
            _append_skip(skips, record, sequence, frame["time"], "near_blank_frame", frame["blur"])
            continue
        if not sequence["protected_positive"] and frame["blur"] < blur_threshold:
            _append_skip(skips, record, sequence, frame["time"], "below_blur_threshold", frame["blur"])
            continue
        distance = _hamming(frame["phash"], previous["phash"]) if previous is not None else None
        if (
            not sequence["protected_positive"]
            and previous is not None
            and frame["time"] - previous["time"] <= 1.01 / fps
            and distance <= duplicate_distance
        ):
            _append_skip(
                skips, record, sequence, frame["time"],
                "perceptual_duplicate_frame", frame["blur"], distance,
            )
            continue
        if sequence["protected_positive"] and (
            frame["brightness_std"] < 1.5 or frame["blur"] < blur_threshold
        ):
            frame["quality_note"] = "low_quality_protected_positive"
        frame["phash_distance_previous"] = distance
        kept.append(frame)
        previous = frame

    # If all negative frames were rejected by conservative quality filters,
    # retain the sharpest decoded frame rather than losing a labeled sequence.
    if not kept and decoded and not sequence["protected_positive"]:
        best = max(decoded, key=lambda item: item["blur"])
        best["quality_note"] = "retained_as_sharpest_fallback"
        kept = [best]
        skips = [
            item for item in skips
            if not (item["sequence_id"] == sequence["sequence_id"] and item["time_seconds"] == round(best["time"], 4))
        ]

    relative_paths = []
    out_dir = dataset_root / sequence["split"] / sequence["class_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in kept:
        filename = f"{record['video_id']}_{sequence['sequence_token']}_f{item['frame_index']:04d}.jpg"
        target = out_dir / filename
        if not target.exists():
            if not cv2.imwrite(
                str(target), item["frame"], [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                _append_skip(skips, record, sequence, item["time"], "jpeg_write_failed", item["blur"])
                continue
        relative_paths.append(target.relative_to(dataset_root).as_posix())
        item["relative_path"] = target.relative_to(dataset_root).as_posix()
        item["status"] = "kept"
    if not relative_paths:
        return [], skips, "no_frames_saved"

    frame_rows = [{
        "source_video": record["relative_path"],
        "video_id": record["video_id"],
        "source_group_id": sequence["source_group_id"],
        "sequence_id": sequence["sequence_id"],
        "split": sequence["split"],
        "class_name": sequence["class_name"],
        "time_seconds": round(float(item["time"]), 4),
        "frame_path": item["relative_path"],
        "laplacian_variance": round(float(item["blur"]), 4),
        "phash64": f"{item['phash']:016x}",
        "phash_distance_previous": item.get("phash_distance_previous"),
        "quality_note": item.get("quality_note", ""),
        "status": "kept",
    } for item in kept if item.get("relative_path")]
    status = "partial" if skips else "complete"
    return frame_rows, skips, status


def _video_fingerprint(record):
    return hashlib.sha256(
        f"{record['sha256']}|{record['source_group_id']}|{record['label']}|{record['split']}".encode()
    ).hexdigest()


def _sequence_records(record, split, config):
    specs = _window_specs(record, config)
    sequence_rows = []
    for index, spec in enumerate(specs):
        token = f"s{index:03d}"
        sequence_rows.append({
            **spec,
            "sequence_id": f"{record['video_id']}_{token}",
            "sequence_token": token,
            "video_id": record["video_id"],
            "source_group_id": record["source_group_id"],
            "source": record["source"],
            "split": split,
            "original_path": record["relative_path"],
            "frame_paths": [],
            "status": "pending",
            "skipped_frame_count": 0,
        })
    return sequence_rows


def _inventory_summary(records, duplicate_groups, near_duplicates, source_info):
    labels = Counter(item.get("label") or "Unresolved" for item in records)
    sources = Counter(item["source"] for item in records)
    valid = [item for item in records if item.get("duration_seconds", 0) > 0]
    durations = [item["duration_seconds"] for item in valid]
    duration_summary = {}
    if durations:
        duration_summary = {
            "min_seconds": round(float(min(durations)), 3),
            "median_seconds": round(float(np.median(durations)), 3),
            "max_seconds": round(float(max(durations)), 3),
            "total_hours": round(float(sum(durations) / 3600), 3),
        }
    return {
        "video_count": len(records),
        "total_bytes": sum(item["file_size_bytes"] for item in records),
        "total_gib": round(sum(item["file_size_bytes"] for item in records) / 1024**3, 3),
        "supported_extensions": dict(Counter(Path(item["relative_path"]).suffix.lower() for item in records)),
        "labels": dict(labels),
        "sources": dict(sources),
        "durations": duration_summary,
        "fps_top_15": dict(Counter(round(item["fps"], 2) for item in records if item["fps"] > 0).most_common(15)),
        "resolutions_top_15": {
            f"{w}x{h}": count for (w, h), count in Counter(
                (item["width"], item["height"]) for item in records if item["width"] and item["height"]
            ).most_common(15)
        },
        "opened_video_count": sum(bool(item["opened"]) for item in records),
        "metadata_or_sample_read_warnings": sum(item["read_status"] != "ok" for item in records),
        "exact_duplicate_groups": duplicate_groups,
        "near_duplicate_candidates_review_only": len(near_duplicates),
        "unresolved_label_video_count": labels.get("Unresolved", 0),
        "source_metadata": source_info,
    }


def _finalize_manifests(output_root: Path):
    videos = _read_jsonl(output_root / "videos_manifest.jsonl")
    sequences = _read_jsonl(output_root / "sequences_manifest.jsonl")
    frames = _read_jsonl(output_root / "frames_manifest.jsonl")
    skips = _read_jsonl(output_root / "skipped_frames.jsonl")
    video_fields = [
        "video_id", "original_filename", "original_path", "source", "assigned_class",
        "split", "source_group_id", "duration_seconds", "fps", "width", "height",
        "video_fingerprint", "preprocessing_status", "sequence_count", "extracted_frame_count",
        "skipped_frame_count", "extracted_frame_paths",
    ]
    seq_fields = [
        "sequence_id", "video_id", "source_group_id", "source", "class_name", "split",
        "original_path", "start_seconds", "end_seconds", "frame_paths", "extracted_frame_count",
        "skipped_frame_count", "status",
    ]
    frame_fields = [
        "source_video", "video_id", "source_group_id", "sequence_id", "split", "class_name", "time_seconds",
        "frame_path", "laplacian_variance", "phash64", "phash_distance_previous", "quality_note", "status",
    ]
    skip_fields = [
        "source_video", "video_id", "sequence_id", "time_seconds", "reason",
        "laplacian_variance", "phash_distance",
    ]
    def compact(rows, keys):
        unique = {}
        for row in rows:
            key = tuple(row.get(field) for field in keys)
            unique[key] = row
        return list(unique.values())
    _write_csv(output_root / "videos_manifest.csv", compact(videos, ["video_id"]), video_fields)
    _write_csv(output_root / "sequences_manifest.csv", compact(sequences, ["sequence_id"]), seq_fields)
    _write_csv(output_root / "frames_manifest.csv", compact(frames, ["frame_path"]), frame_fields)
    _write_csv(output_root / "skipped_frames.csv", compact(skips, ["source_video", "sequence_id", "time_seconds", "reason"]), skip_fields)


def _serialize_inventory(record):
    item = dict(record)
    item.pop("signature", None)
    item.pop("path", None)
    if item.get("normal_ranges") is not None:
        item["normal_ranges"] = json.dumps(item["normal_ranges"])
    if item.get("sample_reads") is not None:
        item["sample_reads"] = json.dumps(item["sample_reads"])
    return item


def prepare_video_dataset(config, config_path, project_root=PROJECT_ROOT, workers=8, force_audit=False):
    paths_config = config["paths"]
    prep = config["preprocessing"]
    split_config = config["splits"]
    video_root = (project_root / paths_config["video_root"]).resolve()
    output_root = (project_root / paths_config["dataset_root"]).resolve()
    results_root = (project_root / paths_config["results_root"]).resolve()
    if not video_root.is_dir():
        raise FileNotFoundError(f"Video dataset folder not found: {video_root}")
    if not (video_root / "TAD-benchmark").is_dir():
        raise FileNotFoundError(f"Expected the extracted TAD benchmark under {video_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "preprocessing": prep,
        "splits": split_config,
        "config_path": str(config_path),
    }
    config_hash = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    run_config_file = output_root / "preprocessing_config.json"
    if run_config_file.exists():
        previous = json.loads(run_config_file.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config_hash:
            raise RuntimeError(
                f"{output_root} already contains a different preprocessing run. "
                "Choose a new dataset_root in the config to preserve it."
            )
    else:
        _write_json(run_config_file, {**run_config, "config_hash": config_hash})

    records, source_info = discover_video_records(video_root)
    cache_file = results_root / "video_inventory.json"
    inventory = None
    if cache_file.is_file() and not force_audit:
        saved = json.loads(cache_file.read_text(encoding="utf-8"))
        signature = [
            (item["relative_path"], item["file_size_bytes"], item.get("mtime_ns"))
            for item in saved.get("records", [])
        ]
        current = [(item["relative_path"], item["path"].stat().st_size, item["path"].stat().st_mtime_ns) for item in records]
        if signature == current:
            inventory = saved
            print("Reusing cached video inspection; file size and modification times match.", flush=True)
    if inventory is None:
        inspected = inspect_videos(records, workers=workers)
        for item in inspected:
            source_path = video_root / item["relative_path"]
            item["mtime_ns"] = source_path.stat().st_mtime_ns
        exact_duplicates = mark_exact_duplicates(inspected)
        near_duplicates = near_duplicate_candidates(inspected)
        summary = _inventory_summary(inspected, exact_duplicates, near_duplicates, source_info)
        inventory = {
            "video_root": str(video_root),
            "summary": summary,
            "records": inspected,
            "exact_duplicate_details": exact_duplicates,
            "near_duplicate_candidates": near_duplicates,
        }
        _write_json(cache_file, inventory)
        _write_csv(
            results_root / "video_inventory.csv",
            [_serialize_inventory(item) for item in inspected],
            list(_serialize_inventory(inspected[0]).keys()) if inspected else [],
        )
        _write_csv(
            results_root / "duplicate_video_candidates.csv",
            near_duplicates,
            ["path_a", "path_b", "source_a", "source_b", "frame_phash_hamming", "review_status"],
        )
        _write_json(results_root / "video_inventory_summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    # Reattach non-serializable absolute paths and normal ranges from discovery.
    by_path = {item["relative_path"]: item for item in records}
    inspected = inventory["records"]
    for item in inspected:
        current = by_path[item["relative_path"]]
        item["path"] = current["path"]
        item["normal_ranges"] = current["normal_ranges"]

    eligible_records = []
    excluded_records = []
    for record in inspected:
        reason = None
        if record.get("duplicate_label_conflict"):
            reason = "exact_duplicate_has_conflicting_labels"
        elif record.get("label") is None:
            reason = record.get("label_confidence", "unresolved_label")
        elif record.get("source") == "TAD" and record.get("label") == "Accident":
            reason = "accident_video_has_no_event_timestamp"
        elif not record.get("opened") or record.get("read_status") == "cannot_open":
            reason = "video_cannot_be_opened"
        elif record.get("fps", 0) <= 0 or record.get("frame_count", 0) <= 0:
            reason = "video_has_no_usable_frame_metadata"
        elif record.get("duplicate_of") and record["duplicate_of"] != record.get("relative_path"):
            reason = "exact_duplicate_copy"

        if reason:
            excluded_records.append({**record, "exclusion_reason": reason})
        else:
            eligible_records.append(record)

    assignments, groups = assign_group_splits(eligible_records, split_config, int(prep["seed"]))
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            (output_root / split / class_name).mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "requested_proportions": split_config,
        "group_counts": dict(Counter(assignments.values())),
        "groups": {key: value for key, value in assignments.items()},
        "tad_test_groups_locked": sum(
            1 for group_id, items in groups.items()
            if assignments.get(group_id) == "test" and any(
                item["source"] == "TAD" and item["official_split"] == "test" for item in items
            )
        ),
    }
    _write_json(output_root / "split_manifest.json", split_manifest)
    exclusion_fields = [
        "relative_path", "source", "label", "label_confidence", "exclusion_reason",
        "read_status", "opened", "sample_reads",
    ]
    _write_csv(output_root / "preprocessing_exclusions.csv", [
        {**item, "sample_reads": json.dumps(item.get("sample_reads", []))}
        for item in excluded_records
    ], exclusion_fields)
    excluded_by_path = {item["relative_path"]: item["exclusion_reason"] for item in excluded_records}

    video_manifest_path = output_root / "videos_manifest.jsonl"
    sequence_manifest_path = output_root / "sequences_manifest.jsonl"
    previous_videos = _read_jsonl(video_manifest_path)
    done_by_id = {row["video_id"]: row for row in previous_videos if row.get("video_fingerprint")}
    sequence_rows = _read_jsonl(sequence_manifest_path)
    existing_sequence_ids = {row["sequence_id"] for row in sequence_rows}
    frame_log_path = output_root / "frames_manifest.jsonl"
    skip_log_path = output_root / "skipped_frames.jsonl"
    processed = 0

    for index, record in enumerate(inspected, 1):
        if record["relative_path"] in excluded_by_path:
            continue
        if record.get("label") is None or record.get("split") in (None, "unresolved"):
            if not record.get("unresolved_logged"):
                _append_jsonl(video_manifest_path, [{
                    "video_id": record["sha256"][:16],
                    "original_filename": record["filename"],
                    "original_path": record["relative_path"],
                    "source": record["source"],
                    "assigned_class": "Unresolved",
                    "split": "unresolved",
                    "source_group_id": record.get("source_group_id"),
                    "duration_seconds": record.get("duration_seconds", 0),
                    "fps": record.get("fps", 0),
                    "width": record.get("width", 0),
                    "height": record.get("height", 0),
                    "video_fingerprint": _video_fingerprint(record),
                    "preprocessing_status": "skipped_unresolved_label",
                    "sequence_count": 0,
                    "extracted_frame_count": 0,
                    "skipped_frame_count": 0,
                    "extracted_frame_paths": [],
                }])
            continue
        video_id = record["sha256"][:16]
        record["video_id"] = video_id
        record["split"] = record.get("split", "unresolved")
        fingerprint = _video_fingerprint(record)
        previous = done_by_id.get(video_id)
        if previous and previous.get("video_fingerprint") == fingerprint:
            paths = previous.get("extracted_frame_paths") or []
            if all((output_root / item).is_file() for item in paths) and previous.get("preprocessing_status") in ("complete", "partial"):
                continue

        sequences = _sequence_records(record, record["split"], prep)
        capture = cv2.VideoCapture(str(record["path"]))
        video_skips, video_frames, sequence_out = [], [], []
        if not capture.isOpened():
            status = "cannot_open"
        else:
            status = "complete"
            for seq in sequences:
                if seq["sequence_id"] in existing_sequence_ids:
                    # Existing sequence is resumable if every referenced file remains.
                    old = next((row for row in sequence_rows if row["sequence_id"] == seq["sequence_id"]), None)
                    if old and all((output_root / path).is_file() for path in old.get("frame_paths", [])):
                        sequence_out.append(old)
                        video_frames.extend(old.get("frame_paths", []))
                        continue
                rows, skips, sequence_status = _process_sequence(
                    capture, record, seq, output_root, prep
                )
                video_skips.extend(skips)
                frame_paths = [row["frame_path"] for row in rows]
                seq["frame_paths"] = frame_paths
                seq["extracted_frame_count"] = len(frame_paths)
                seq["skipped_frame_count"] = len(skips)
                seq["status"] = "ready" if frame_paths else sequence_status
                if frame_paths:
                    sequence_out.append(seq)
                    video_frames.extend(frame_paths)
                video_frames_log = rows
                if video_frames_log:
                    _append_jsonl(frame_log_path, video_frames_log)
                existing_sequence_ids.add(seq["sequence_id"])
                if sequence_status != "complete":
                    status = "partial"
            capture.release()

        _append_jsonl(skip_log_path, video_skips)
        _append_jsonl(sequence_manifest_path, sequence_out)
        manifest_record = {
            "video_id": video_id,
            "original_filename": record["filename"],
            "original_path": record["relative_path"],
            "source": record["source"],
            "assigned_class": record["label"],
            "split": record["split"],
            "source_group_id": record["source_group_id"],
            "duration_seconds": record.get("duration_seconds", 0),
            "fps": record.get("fps", 0),
            "width": record.get("width", 0),
            "height": record.get("height", 0),
            "video_fingerprint": fingerprint,
            "preprocessing_status": status,
            "sequence_count": len(sequence_out),
            "extracted_frame_count": len(set(video_frames)),
            "skipped_frame_count": len(video_skips),
            "extracted_frame_paths": sorted(set(video_frames)),
        }
        _append_jsonl(video_manifest_path, [manifest_record])
        done_by_id[video_id] = manifest_record
        processed += 1
        if processed % 25 == 0:
            print(f"Preprocessed {processed} newly processed videos ({index}/{len(inspected)})", flush=True)

    _finalize_manifests(output_root)
    final_videos = _read_jsonl(video_manifest_path)
    final_sequences = _read_jsonl(sequence_manifest_path)
    final_frames = _read_jsonl(frame_log_path)
    final_skips = _read_jsonl(skip_log_path)
    summary = {
        "inventory": inventory["summary"],
        "source_group_split_counts": split_manifest["group_counts"],
        "source_group_split_percent": {
            split: round(100 * count / max(1, sum(split_manifest["group_counts"].values())), 2)
            for split, count in split_manifest["group_counts"].items()
        },
        "tad_official_test_groups_preserved": split_manifest["tad_test_groups_locked"],
        "video_manifest_statuses": dict(Counter(item["preprocessing_status"] for item in final_videos)),
        "excluded_videos_by_reason": dict(Counter(item["exclusion_reason"] for item in excluded_records)),
        "eligible_source_video_count": len(eligible_records),
        "sequences_by_split_class": {
            split: dict(Counter(
                item["class_name"] for item in final_sequences if item.get("split") == split and item.get("status") == "ready"
            )) for split in SPLITS
        },
        "frames_by_split_class": {
            split: dict(Counter(
                item["class_name"] for item in final_frames if item.get("split") == split
            )) for split in SPLITS
        },
        "frames_extracted": sum(item.get("status") == "kept" for item in final_frames),
        "frames_skipped": len(final_skips),
        "skip_reasons": dict(Counter(item["reason"] for item in final_skips)),
        "processed_this_run": processed,
        "dataset_root": str(output_root),
        "results_root": str(results_root),
    }
    _write_json(output_root / "dataset_stats.json", summary)
    _write_json(results_root / "dataset_stats.json", summary)
    return summary
