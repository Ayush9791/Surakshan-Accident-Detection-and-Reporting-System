"""Verify the existing Keras accident classifier without training or saving weights."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_system import predict as inference  # Loads the existing weights once.

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_TO_INDEX = {"Accident": 0, "Non_Accident": 1}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Image file or directory")
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    parser.add_argument(
        "--manifest", type=Path,
        help="Optional frame manifest; evaluates the selected split and groups frames by annotated window",
    )
    parser.add_argument(
        "--split", choices=("val", "test"), default="test",
        help="Manifest split to evaluate (default: test)",
    )
    parser.add_argument("--threshold", type=float, help="Override the configured accident probability threshold")
    parser.add_argument("--metrics-output", type=Path, help="Optional JSON metrics output (requires --manifest)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--quiet", action="store_true", help="Write results to files without printing every prediction")
    return parser.parse_args()


def normalize_class(value):
    value = value.strip().casefold().replace(" ", "_").replace("-", "_")
    return {"accident": "Accident", "non_accident": "Non_Accident"}.get(value)


def collect_inputs(input_path: Path, manifest_path: Path | None, split: str):
    input_path = input_path.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {input_path.suffix}")
        files = [input_path]
    else:
        files = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.casefold() in SUPPORTED_EXTENSIONS
        )

    manifest_rows = {}
    if manifest_path:
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
        with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if row.get("status") != "kept" or row.get("split") != split:
                    continue
                class_name = row.get("class_name", "")
                if class_name not in CLASS_TO_INDEX:
                    raise ValueError(f"Unknown class in test manifest: {class_name}")
                path = (manifest_path.parent / row["frame_path"]).resolve()
                if input_path.is_dir():
                    try:
                        path.relative_to(input_path)
                    except ValueError:
                        continue
                elif path != input_path:
                    continue
                manifest_rows[path] = {
                    "ground_truth": class_name,
                    "sequence_id": row.get("sequence_id", ""),
                    "source_group_id": row.get("source_group_id", ""),
                    "source_video": row.get("source_video", ""),
                }
        files = sorted(manifest_rows)
        if not files:
            raise ValueError("No kept test frames in the manifest match --input")
    else:
        for path in files:
            class_name = normalize_class(path.parent.name)
            if class_name:
                manifest_rows[path] = {"ground_truth": class_name, "sequence_id": ""}

    if not files:
        raise ValueError(f"No supported images found in {input_path}")
    return files, manifest_rows


def calculate_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positive = labels == CLASS_TO_INDEX["Accident"]
    predicted_positive = probabilities >= float(threshold)
    tp = int(np.sum(positive & predicted_positive))
    tn = int(np.sum(~positive & ~predicted_positive))
    fp = int(np.sum(~positive & predicted_positive))
    fn = int(np.sum(positive & ~predicted_positive))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "samples": int(len(labels)),
        "accuracy": (tp + tn) / max(1, len(labels)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "threshold": float(threshold),
    }


def run():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.metrics_output and not args.manifest:
        raise ValueError("--metrics-output requires --manifest")

    files, labels_by_path = collect_inputs(args.input, args.manifest, args.split)
    model = inference.model
    threshold = float(inference.ACCIDENT_THRESHOLD if args.threshold is None else args.threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    print(f"Weights loaded: {inference.MODEL_PATH}")
    print(f"TensorFlow GPUs: {tf_gpu_list()}")
    print(f"Input shape: {model.input_shape}; output shape: {model.output_shape}")
    print(f"Parameters: {model.count_params():,}; threshold: {threshold:.6f}")

    output_rows = []
    valid_labels, valid_probabilities = [], []
    window_groups = defaultdict(list)
    for start in range(0, len(files), args.batch_size):
        batch_paths = files[start:start + args.batch_size]
        tensors, tensor_paths = [], []
        for path in batch_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                output_rows.append({"filename": path.name, "path": str(path), "error": "unreadable image"})
                continue
            tensors.append(inference._preprocess_frame(image)[0])
            tensor_paths.append(path)
        if not tensors:
            continue
        predictions = model(np.stack(tensors).astype(np.float32), training=False).numpy()
        for path, probabilities in zip(tensor_paths, predictions):
            accident_probability = float(probabilities[inference.ACCIDENT_INDEX])
            non_accident_probability = float(probabilities[inference.NON_ACCIDENT_INDEX])
            predicted_class = "Accident" if accident_probability >= threshold else "Non_Accident"
            truth = labels_by_path.get(path, {}).get("ground_truth", "")
            sequence_id = labels_by_path.get(path, {}).get("sequence_id", "")
            row = {
                "filename": path.name,
                "path": str(path),
                "accident_probability": accident_probability,
                "non_accident_probability": non_accident_probability,
                "predicted_class": predicted_class,
                "threshold": threshold,
                "ground_truth": truth,
                "correct": (predicted_class == truth) if truth else "",
                "sequence_id": sequence_id,
                "error": "",
            }
            output_rows.append(row)
            if truth:
                valid_labels.append(CLASS_TO_INDEX[truth])
                valid_probabilities.append(accident_probability)
            if sequence_id:
                window_groups[sequence_id].append((CLASS_TO_INDEX[truth], accident_probability))
        done = min(start + args.batch_size, len(files))
        print(f"Processed {done}/{len(files)} images", flush=True)

    output_rows.sort(key=lambda row: row["path"].casefold())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "filename", "path", "accident_probability", "non_accident_probability",
            "predicted_class", "threshold", "ground_truth", "correct", "sequence_id", "error",
        ]
        with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"CSV written: {args.output.resolve()}")

    if not args.quiet:
        for row in output_rows:
            if row.get("error"):
                print(f"{row['filename']}: ERROR: {row['error']}")
            else:
                print(
                    f"{row['filename']}: accident={row['accident_probability']:.6f}, "
                    f"non_accident={row['non_accident_probability']:.6f}, "
                    f"predicted={row['predicted_class']}, threshold={threshold:.6f}"
                )

    if args.manifest and valid_labels:
        frame_metrics = calculate_metrics(valid_labels, valid_probabilities, threshold)
        grouped_labels, grouped_probabilities = [], []
        for window_id, items in window_groups.items():
            truths = {item[0] for item in items}
            if len(truths) != 1:
                raise ValueError(f"Conflicting test labels in window {window_id}")
            grouped_labels.append(next(iter(truths)))
            grouped_probabilities.append(float(np.mean([item[1] for item in items])))
        window_metrics = calculate_metrics(grouped_labels, grouped_probabilities, threshold)
        metrics = {
            "evaluation_split": args.split,
            "threshold_source": "command line override" if args.threshold is not None else "live inference configuration (metadata, environment override, or code fallback)",
            "frame_metrics": frame_metrics,
            "annotated_window_metrics": window_metrics,
            "source_video_groups": len({labels_by_path[p].get("source_group_id", "") for p in files if p in labels_by_path}),
        }
        print(f"{args.split} metrics (annotated windows):")
        print(json.dumps(window_metrics, indent=2))
        print(f"{args.split} frame count: {frame_metrics['samples']}; annotated window count: {window_metrics['samples']}")
        if args.metrics_output:
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"Metrics written: {args.metrics_output.resolve()}")
    elif args.manifest:
        print("No readable labeled test images were evaluated.")


def tf_gpu_list():
    # Report runtime only; do not change device configuration.
    import tensorflow as tf
    return [device.name for device in tf.config.list_physical_devices("GPU")]


if __name__ == "__main__":
    try:
        run()
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"Verification failed: {error}") from error
