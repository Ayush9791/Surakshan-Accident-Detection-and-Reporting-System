
"""Shared temporal video inference for Surakshan cameras."""

import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(
    os.environ.get(
        "SURAKSHAN_MODEL_PATH",
        str(PROJECT_ROOT / "models" / "keras" / "car_accident_model.weights.h5")
    )
)

if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Keras weights not found: {MODEL_PATH.resolve()}"
    )

# Original Keras model settings
IMAGE_SIZE = 128
ACCIDENT_INDEX = 0
NON_ACCIDENT_INDEX = 1

METADATA_PATH = MODEL_PATH.parent / "model_metadata.json"


def _load_default_threshold():
    if METADATA_PATH.is_file():
        import json
        try:
            metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            value = float(metadata["threshold"])
            if 0.0 <= value <= 1.0:
                print(f"[SURAKSHAN] Loaded validation threshold {value:.4f} from {METADATA_PATH}", flush=True)
                return value
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            print(f"[SURAKSHAN] Could not read model threshold metadata: {error}", flush=True)
    return 0.93


ACCIDENT_THRESHOLD = float(
    os.environ.get("SURAKSHAN_ACCIDENT_THRESHOLD", str(_load_default_threshold()))
)

FRAME_INTERVAL_SECONDS = float(
    os.environ.get("SURAKSHAN_FRAME_INTERVAL_SECONDS", "0.5")
)

REQUIRED_CONSECUTIVE_POSITIVES = int(
    os.environ.get("SURAKSHAN_ALERT_CONSECUTIVE", "3")
)

if not 0 <= ACCIDENT_THRESHOLD <= 1:
    raise ValueError("Accident threshold must be between 0 and 1")

if FRAME_INTERVAL_SECONDS <= 0 or REQUIRED_CONSECUTIVE_POSITIVES < 1:
    raise ValueError("Invalid inference timing configuration")


def build_model():
    """Recreate the original trained Keras CNN architecture."""

    model = Sequential([
        Conv2D(
            32, (3, 3), activation="relu",
            padding="same", input_shape=(128, 128, 1)
        ),
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

    return model


# Load the trained model once when this module is imported.
model = build_model()
model.load_weights(str(MODEL_PATH))

print(f"[SURAKSHAN] Keras CNN loaded: {MODEL_PATH}", flush=True)


def _preprocess_frame(frame_bgr):
    """Apply the same preprocessing used by the original Keras model."""

    if frame_bgr.ndim == 2:
        gray = frame_bgr
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    resized = cv2.resize(
        gray,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    image = resized.astype(np.float32) / 255.0

    # Shape: (1, 128, 128, 1)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    return image


stream_states = {}


def reset_predictor(stream_id="default"):
    """Clear temporal state for a camera."""
    stream_states.pop(stream_id, None)


def _predict_frame_probability(frame):
    image = _preprocess_frame(frame)

    probabilities = model(image, training=False).numpy()[0]

    return float(probabilities[ACCIDENT_INDEX])


def _details(
    probability,
    is_new_prediction,
    label=None,
    incident_timestamp=None,
    frame_probability=None
):
    label = label or (
        "ACCIDENT"
        if probability >= ACCIDENT_THRESHOLD
        else "NORMAL"
    )

    confidence = (
        probability if label == "ACCIDENT"
        else 1.0 - probability
    )

    return {
        "label": label,
        "confidence": float(confidence),
        "accident_probability": float(probability),
        "non_accident_probability": float(1.0 - probability),
        "frame_accident_probability": float(
            frame_probability
            if frame_probability is not None
            else probability
        ),
        "threshold": float(ACCIDENT_THRESHOLD),
        "incident_timestamp": incident_timestamp,
        "is_new_prediction": is_new_prediction,
    }


def _warmup_details():
    return {
        "label": "NORMAL",
        "confidence": 0.0,
        "accident_probability": 0.0,
        "non_accident_probability": 1.0,
        "frame_accident_probability": 0.0,
        "threshold": float(ACCIDENT_THRESHOLD),
        "incident_timestamp": None,
        "is_new_prediction": False,
    }


def predict_clip_details(frames):
    """Predict accident probability across an ordered frame collection."""

    if not frames:
        raise ValueError("At least one frame is required")

    probabilities = [
        _predict_frame_probability(frame)
        for frame in frames
    ]

    probability = float(np.mean(probabilities))

    timestamp = (
        datetime.now(timezone.utc).isoformat(timespec="seconds")
        if probability >= ACCIDENT_THRESHOLD
        else None
    )

    return _details(
        probability,
        True,
        incident_timestamp=timestamp,
        frame_probability=probabilities[-1]
    )


def predict_frame_details(frame, stream_id="default"):
    """Predict one camera frame and maintain temporal alert state."""

    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        raise ValueError("Received an empty camera frame")

    now = time.monotonic()

    state = stream_states.setdefault(
        stream_id,
        {
            "last_sample": float("-inf"),
            "probabilities": deque(
                maxlen=REQUIRED_CONSECUTIVE_POSITIVES
            ),
            "positive_count": 0,
            "incident_timestamp": None,
            "result": _warmup_details(),
        },
    )

    # Avoid processing frames too frequently.
    if now - state["last_sample"] < FRAME_INTERVAL_SECONDS:
        result = dict(state["result"])
        result["is_new_prediction"] = False
        return result

    frame_probability = _predict_frame_probability(frame)

    state["last_sample"] = now
    state["probabilities"].append(frame_probability)

    if frame_probability >= ACCIDENT_THRESHOLD:
        state["positive_count"] += 1
    else:
        state["positive_count"] = 0
        state["incident_timestamp"] = None

    probability = float(np.mean(state["probabilities"]))

    label = (
        "ACCIDENT"
        if state["positive_count"] >= REQUIRED_CONSECUTIVE_POSITIVES
        else "NORMAL"
    )

    if label == "ACCIDENT" and state["incident_timestamp"] is None:
        state["incident_timestamp"] = (
            datetime.now(timezone.utc).isoformat(timespec="seconds")
        )

    state["result"] = _details(
        probability,
        True,
        label=label,
        incident_timestamp=state["incident_timestamp"],
        frame_probability=frame_probability,
    )

    return dict(state["result"])


def predict_frame(frame):
    """Backward-compatible API used by live scripts."""
    return predict_frame_details(frame)["label"]
