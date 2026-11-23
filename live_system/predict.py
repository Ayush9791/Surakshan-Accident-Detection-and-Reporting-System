"""Shared temporal video inference for Surakshan cameras."""
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.video import R3D_18_Weights

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import AccidentNet, R3D18AccidentNet, VideoAccidentNet
from video_preprocessing import normalize_rgb_clip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("SURAKSHAN_MODEL_PATH", str(PROJECT_ROOT / "surakshan_model.pth")))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_INTERVAL_SECONDS = 0.34

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model checkpoint not found at {MODEL_PATH.resolve()}. "
        "Place surakshan_model.pth in the project root or set SURAKSHAN_MODEL_PATH."
    )

try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
except TypeError:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    class_to_idx = checkpoint.get("class_to_idx", {"Accident": 0, "Non Accident": 1})
    MODEL_TYPE = checkpoint.get("model_type", "frame_cnn")
    ACCIDENT_THRESHOLD = float(checkpoint.get("threshold", 0.5))
    FRAME_COUNT = int(checkpoint.get("frame_count", 8))
    IMAGE_SIZE = int(checkpoint.get("image_size", 128))
    NORMALIZE_MEAN = checkpoint.get("mean", [0.485, 0.456, 0.406])
    NORMALIZE_STD = checkpoint.get("std", [0.229, 0.224, 0.225])
    PREPROCESSING = checkpoint.get("preprocessing", "")
else:
    state_dict = checkpoint
    class_to_idx = {"Accident": 0, "Non Accident": 1}
    MODEL_TYPE = "frame_cnn"
    ACCIDENT_THRESHOLD = 0.5
    FRAME_COUNT = 8
    IMAGE_SIZE = 128
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    PREPROCESSING = ""

if "Accident" not in class_to_idx:
    raise ValueError(f"Checkpoint does not define the Accident class: {class_to_idx}")

ACCIDENT_INDEX = int(class_to_idx["Accident"])
IDX_TO_CLASS = {int(index): name for name, index in class_to_idx.items()}
IS_R3D = MODEL_TYPE == "r3d18_kinetics400"
IS_MOBILENET_GRU = MODEL_TYPE == "mobilenet_v3_small_gru"
IS_TEMPORAL = IS_R3D or IS_MOBILENET_GRU
if IS_R3D:
    model = R3D18AccidentNet(pretrained_backbone=False).to(DEVICE)
elif IS_MOBILENET_GRU:
    model = VideoAccidentNet(pretrained_backbone=False).to(DEVICE)
else:
    model = AccidentNet().to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

frame_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])
letterbox_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


def _frame_tensor(frame_bgr):
    if PREPROCESSING.startswith("RGB 224x224 aspect-preserving letterbox"):
        height, width = frame_bgr.shape[:2]
        scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
        out_w, out_h = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 114, dtype=np.uint8)
        x, y = (IMAGE_SIZE - out_w) // 2, (IMAGE_SIZE - out_h) // 2
        canvas[y:y + out_h, x:x + out_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        return letterbox_transform(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return frame_transform(image).unsqueeze(0).to(DEVICE)
video_transform = R3D_18_Weights.KINETICS400_V1.transforms()
stream_states = {}


def reset_predictor(stream_id="default"):
    """Clear temporal warmup state for a camera stream."""
    stream_states.pop(stream_id, None)


def _details(probability, is_new_prediction):
    label = "ACCIDENT" if probability >= ACCIDENT_THRESHOLD else "NORMAL"
    confidence = probability if label == "ACCIDENT" else 1.0 - probability
    return {
        "label": label,
        "confidence": float(confidence),
        "accident_probability": float(probability),
        "is_new_prediction": is_new_prediction,
    }


def _warmup_details():
    return {
        "label": "NORMAL",
        "confidence": 0.0,
        "accident_probability": 0.0,
        "is_new_prediction": False,
    }


def _encode_frame(frame):
    tensor = _frame_tensor(frame)
    with torch.inference_mode():
        return model.encode_frames(tensor)[0].detach()


def predict_clip_details(frames):
    """Run a full ordered frame clip through the checkpoint's video model."""
    if len(frames) != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} frames, received {len(frames)}")
    if IS_R3D:
        rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        if PREPROCESSING.startswith("full-frame aspect-preserving letterbox"):
            clip = normalize_rgb_clip(rgb, IMAGE_SIZE, DEVICE).unsqueeze(0)
        else:
            clip = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2)
            clip = video_transform(clip).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            probability = float(model(clip).softmax(1)[0, ACCIDENT_INDEX].item())
        return _details(probability, True)
    if IS_MOBILENET_GRU:
        embeddings = torch.stack([_encode_frame(frame) for frame in frames]).unsqueeze(0)
        with torch.inference_mode():
            probability = float(model.classify_embeddings(embeddings).softmax(1)[0, 1].item())
        return _details(probability, True)

    tensor = _frame_tensor(frames[-1])
    with torch.inference_mode():
        probability = float(model(tensor).softmax(1)[0, ACCIDENT_INDEX].item())
    return _details(probability, True)


def predict_frame_details(frame, stream_id="default"):
    """Update one camera's video window and return its latest clip prediction."""
    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        raise ValueError("predict_frame_details received an empty frame")
    if not IS_TEMPORAL:
        return predict_clip_details([frame] * FRAME_COUNT)

    now = time.monotonic()
    state = stream_states.setdefault(
        stream_id,
        {"last_sample": float("-inf"), "frames": deque(maxlen=FRAME_COUNT),
         "embeddings": deque(maxlen=FRAME_COUNT), "result": _warmup_details()},
    )
    if now - state["last_sample"] < FRAME_INTERVAL_SECONDS:
        result = dict(state["result"])
        result["is_new_prediction"] = False
        return result

    if IS_R3D:
        state["frames"].append(frame.copy())
    else:
        state["embeddings"].append(_encode_frame(frame))
    state["last_sample"] = now
    queue = state["frames"] if IS_R3D else state["embeddings"]
    if len(queue) < FRAME_COUNT:
        state["result"] = _warmup_details()
        return dict(state["result"])

    if IS_R3D:
        state["result"] = predict_clip_details(list(state["frames"]))
    else:
        clip = torch.stack(list(state["embeddings"])).unsqueeze(0)
        with torch.inference_mode():
            probability = float(model.classify_embeddings(clip).softmax(1)[0, 1].item())
        state["result"] = _details(probability, True)
    return dict(state["result"])


def predict_frame(frame):
    """Backward-compatible API used by the live scripts."""
    return predict_frame_details(frame)["label"]
