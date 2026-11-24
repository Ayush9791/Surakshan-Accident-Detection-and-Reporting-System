
"""Train and evaluate Surakshan's Keras CNN accident classifier."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_SIZE = 128
ACCIDENT_INDEX = 0
NON_ACCIDENT_INDEX = 1

CLASS_TO_IDX = {
    "Accident": ACCIDENT_INDEX,
    "Non_Accident": NON_ACCIDENT_INDEX,
}

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "keras"
    / "car_accident_model.weights.h5"
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "keras"


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu():
    """Enable TensorFlow GPU memory growth when a compatible GPU is available."""

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        print("[SURAKSHAN] TensorFlow GPU not detected. Using CPU.", flush=True)
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print(f"[SURAKSHAN] GPU configuration warning: {error}", flush=True)

    print(f"[SURAKSHAN] TensorFlow GPUs: {gpus}", flush=True)


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    for section in ("paths", "preprocessing", "training"):
        if section not in config:
            raise ValueError(f"Configuration missing section: {section}")

    return config


def build_model():
    """Build the same architecture used by live_system/predict.py."""

    model = Sequential([
        Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),

        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(2, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def read_frame_manifest(dataset_root: Path):
    """Read the existing manifest and reject source-video split leakage."""

    manifest_path = dataset_root / "frames_manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run preprocess_video_dataset.py first."
        )

    rows = []
    group_splits = defaultdict(set)
    seen_paths = set()

    with manifest_path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as stream:
        for row in csv.DictReader(stream):

            if row.get("status") != "kept":
                continue

            split = row.get("split")
            class_name = row.get("class_name")

            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split: {split}")

            if class_name not in CLASS_TO_IDX:
                raise ValueError(f"Invalid class name: {class_name}")

            source_group_id = (
                row.get("source_group_id")
                or row.get("video_id")
                or row.get("source_video")
            )

            sequence_id = row.get("sequence_id")

            if not source_group_id or not sequence_id:
                raise ValueError(
                    f"Missing source group or sequence ID: {row}"
                )

            relative_path = Path(row.get("frame_path", ""))

            image_path = (dataset_root / relative_path).resolve()

            try:
                image_path.relative_to(dataset_root.resolve())
            except ValueError as error:
                raise ValueError(
                    f"Frame path escapes dataset root: {relative_path}"
                ) from error

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Missing training image: {image_path}"
                )

            path_key = str(image_path).casefold()

            if path_key in seen_paths:
                raise ValueError(f"Duplicate frame path: {image_path}")

            seen_paths.add(path_key)
            group_splits[source_group_id].add(split)

            rows.append({
                "image_path": str(image_path),
                "label": CLASS_TO_IDX[class_name],
                "class_name": class_name,
                "split": split,
                "source_group_id": source_group_id,
                "sequence_id": sequence_id,
            })

    leaking = {
        group: splits
        for group, splits in group_splits.items()
        if len(splits) > 1
    }

    if leaking:
        raise ValueError(
            f"Source videos cross dataset splits: {list(leaking.items())[:5]}"
        )

    if not rows:
        raise ValueError("No usable frames found in the manifest.")

    return rows


def preprocess_image(path):
    """Match live_system/predict.py preprocessing exactly."""

    # tf.py_function passes an EagerTensor; decode it before calling OpenCV.
    if hasattr(path, "numpy"):
        path = path.numpy()
    if isinstance(path, (bytes, bytearray, np.bytes_)):
        path = path.decode("utf-8")
    else:
        path = str(path)

    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Could not read image: {path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    image = resized.astype(np.float32) / 255.0

    return np.expand_dims(image, axis=-1)


def load_image_tf(path, label, sample_weight):
    """Bridge OpenCV preprocessing into TensorFlow's data pipeline."""

    image = tf.py_function(
        func=preprocess_image,
        inp=[path],
        Tout=tf.float32,
    )

    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 1))

    return image, label, sample_weight


def make_dataset(rows, batch_size, training=False, sample_weights=None):
    paths = np.asarray(
        [row["image_path"] for row in rows],
        dtype=str,
    )

    labels = np.asarray(
        [row["label"] for row in rows],
        dtype=np.int32,
    )

    if sample_weights is None:
        sample_weights = np.ones(len(rows), dtype=np.float32)

    weights = np.asarray(sample_weights, dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels, weights)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(rows), 10000),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image_tf,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def make_group_balanced_weights(rows):
    """Balance classes and prevent long source videos dominating training."""

    group_counts = Counter(
        (row["label"], row["source_group_id"])
        for row in rows
    )

    groups_per_class = {
        label: len({
            row["source_group_id"]
            for row in rows
            if row["label"] == label
        })
        for label in (ACCIDENT_INDEX, NON_ACCIDENT_INDEX)
    }

    weights = []

    for row in rows:
        label = row["label"]
        group = row["source_group_id"]

        weight = (
            1.0
            / max(1, groups_per_class[label])
            / max(1, group_counts[(label, group)])
        )

        weights.append(weight)

    weights = np.asarray(weights, dtype=np.float32)

    # Keep average sample weight equal to one.
    weights /= max(float(weights.mean()), 1e-8)

    return weights


def aggregate_windows(rows, probabilities):
    """Average frame probabilities within each annotated video window."""

    grouped = defaultdict(list)

    for row, probability in zip(rows, probabilities):
        grouped[row["sequence_id"]].append(
            (row["label"], float(probability))
        )

    labels = []
    window_probabilities = []
    window_ids = []

    for window_id, values in grouped.items():
        window_labels = {item[0] for item in values}

        if len(window_labels) != 1:
            raise ValueError(
                f"Conflicting labels in annotated window: {window_id}"
            )

        labels.append(next(iter(window_labels)))

        window_probabilities.append(
            float(np.mean([item[1] for item in values]))
        )

        window_ids.append(window_id)

    return (
        np.asarray(labels, dtype=np.int32),
        np.asarray(window_probabilities, dtype=np.float32),
        window_ids,
    )


def predict_probabilities(model, rows, batch_size):
    dataset = make_dataset(rows, batch_size)

    image_dataset = dataset.map(
        lambda image, label, weight: image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    predictions = model.predict(
        image_dataset,
        verbose=1,
    )

    return predictions[:, ACCIDENT_INDEX]


def calculate_metrics(labels, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(np.int32)

    actual_positive = labels == ACCIDENT_INDEX
    predicted_positive = predictions == ACCIDENT_INDEX

    tp = int(np.sum(actual_positive & predicted_positive))
    tn = int(np.sum(~actual_positive & ~predicted_positive))
    fp = int(np.sum(~actual_positive & predicted_positive))
    fn = int(np.sum(actual_positive & ~predicted_positive))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = (
        2 * precision * recall
        / max(1e-8, precision + recall)
    )
    accuracy = (tp + tn) / max(1, len(labels))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def choose_threshold(labels, probabilities, recall_target=0.90):
    """Choose maximum specificity subject to validation recall target."""

    candidates = np.unique(
        np.concatenate([
            probabilities,
            np.asarray([0.0, 1.0]),
        ])
    )

    evaluated = [
        (float(threshold), calculate_metrics(
            labels, probabilities, float(threshold)
        ))
        for threshold in candidates
    ]

    eligible = [
        item for item in evaluated
        if item[1]["recall"] >= recall_target
    ]

    if eligible:
        threshold, metrics = max(
            eligible,
            key=lambda item: (
                item[1]["specificity"],
                item[1]["f1"],
                item[0],
            ),
        )

        reason = (
            "maximum specificity subject to validation recall target"
        )

    else:
        threshold, metrics = max(
            evaluated,
            key=lambda item: (
                item[1]["f1"],
                item[1]["recall"],
            ),
        )

        reason = "fallback: maximum validation F1"

    return threshold, reason, metrics


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )

    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(
        description="Train Surakshan's Keras CNN accident classifier."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/keras_training.json"),
    )

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)

    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)

    settings = config["training"]

    epochs = args.epochs or int(settings.get("epochs", 20))
    batch_size = args.batch_size or int(settings.get("batch_size", 32))
    patience = int(settings.get("patience", 5))
    learning_rate = float(settings.get("learning_rate", 0.0001))
    recall_target = float(settings.get("recall_target", 0.90))

    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("Epochs, batch size and patience must be positive.")

    seed = int(config["preprocessing"].get("seed", 42))

    seed_everything(seed)
    configure_gpu()

    dataset_root = (
        PROJECT_ROOT / config["paths"]["dataset_root"]
    ).resolve()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print("[SURAKSHAN] Reading frame manifest...", flush=True)

    rows = read_frame_manifest(dataset_root)

    splits = {
        name: [row for row in rows if row["split"] == name]
        for name in ("train", "val", "test")
    }

    for name, split_rows in splits.items():
        counts = Counter(row["class_name"] for row in split_rows)

        print(
            f"{name}: frames={len(split_rows)} "
            f"Accident={counts['Accident']} "
            f"Non_Accident={counts['Non_Accident']} "
            f"source_groups={len(set(row['source_group_id'] for row in split_rows))}",
            flush=True,
        )

        if not split_rows:
            raise ValueError(f"{name} split is empty.")

        if {row["label"] for row in split_rows} != {0, 1}:
            raise ValueError(
                f"{name} split must contain both accident classes."
            )

    print("[SURAKSHAN] Building Keras CNN...", flush=True)

    model = build_model()
    model.optimizer.learning_rate.assign(learning_rate)

    print(
        f"[SURAKSHAN] Parameters: {model.count_params():,}",
        flush=True,
    )

    train_weights = make_group_balanced_weights(splits["train"])

    train_dataset = make_dataset(
        splits["train"],
        batch_size,
        training=True,
        sample_weights=train_weights,
    )

    val_dataset = make_dataset(
        splits["val"],
        batch_size,
    )

    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    best_reason = ""
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        print(
            f"\n[SURAKSHAN] Epoch {epoch}/{epochs}",
            flush=True,
        )

        start_time = time.time()

        train_result = model.fit(
            train_dataset,
            epochs=1,
            verbose=1,
        )

        train_loss = float(train_result.history["loss"][-1])
        train_accuracy = float(train_result.history["accuracy"][-1])

        val_probabilities = predict_probabilities(
            model,
            splits["val"],
            batch_size,
        )

        val_labels, val_window_probs, _ = aggregate_windows(
            splits["val"],
            val_probabilities,
        )

        threshold, reason, val_metrics = choose_threshold(
            val_labels,
            val_window_probs,
            recall_target,
        )

        elapsed = round(time.time() - start_time, 2)

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_frame_accuracy": train_accuracy,
            "validation_window_metrics": val_metrics,
            "validation_threshold": float(threshold),
            "threshold_reason": reason,
            "seconds": elapsed,
        }

        history.append(epoch_result)

        print(
            f"Train loss={train_loss:.4f} "
            f"Train accuracy={train_accuracy:.4f} "
            f"Val recall={val_metrics['recall']:.4f} "
            f"Val precision={val_metrics['precision']:.4f} "
            f"Val F1={val_metrics['f1']:.4f} "
            f"Threshold={threshold:.4f}",
            flush=True,
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_threshold = threshold
            best_reason = reason
            stale_epochs = 0

            model.save_weights(str(MODEL_PATH))

            save_json(
                RESULTS_ROOT / "training_history.json",
                history,
            )

            print(
                f"[SURAKSHAN] Saved best weights: {MODEL_PATH}",
                flush=True,
            )

        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            print(
                f"[SURAKSHAN] Early stopping. Best epoch: {best_epoch}",
                flush=True,
            )
            break

    if best_epoch == 0:
        raise RuntimeError("Training did not produce a valid model.")

    # Reload the best validation-selected model before test evaluation.
    model.load_weights(str(MODEL_PATH))

    print("\n[SURAKSHAN] Evaluating held-out test split...", flush=True)

    test_probabilities = predict_probabilities(
        model,
        splits["test"],
        batch_size,
    )

    test_labels, test_window_probs, test_window_ids = aggregate_windows(
        splits["test"],
        test_probabilities,
    )

    test_metrics = calculate_metrics(
        test_labels,
        test_window_probs,
        best_threshold,
    )

    test_metrics.update({
        "evaluation_unit": "annotated video window",
        "best_epoch": best_epoch,
        "threshold": float(best_threshold),
        "threshold_selected_on": "validation split",
        "source_window_count": len(test_window_ids),
    })

    metadata = {
        "model_type": "keras_accident_cnn",
        "architecture": "Surakshan Keras CNN",
        "image_size": IMAGE_SIZE,
        "input_shape": [IMAGE_SIZE, IMAGE_SIZE, 1],
        "class_to_idx": CLASS_TO_IDX,
        "preprocessing": (
            "OpenCV BGR to grayscale; resize 128x128 using INTER_AREA; "
            "float32 normalization to 0-1"
        ),
        "threshold": float(best_threshold),
        "threshold_reason": best_reason,
        "best_epoch": best_epoch,
        "validation_f1": float(best_f1),
        "test_metrics": test_metrics,
        "seed": seed,
    }

    # Keep inference metadata beside the weights so deployment can use the
    # exact validation-selected threshold without relying on ignored results/.
    save_json(MODEL_PATH.parent / "model_metadata.json", metadata)
    save_json(RESULTS_ROOT / "model_metadata.json", metadata)

    save_json(
        RESULTS_ROOT / "training_history.json",
        history,
    )

    save_json(
        RESULTS_ROOT / "test_metrics.json",
        test_metrics,
    )

    print("\n[SURAKSHAN] Training complete.", flush=True)
    print(f"Best epoch: {best_epoch}", flush=True)
    print(f"Threshold: {best_threshold:.4f}", flush=True)
    print(
        "Test metrics:",
        json.dumps(test_metrics, indent=2),
        flush=True,
    )
    print(f"Weights: {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
