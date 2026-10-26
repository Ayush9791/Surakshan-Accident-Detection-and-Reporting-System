"""Shared frame inference for Surakshan."""
import os
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import AccidentNet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.environ.get("SURAKSHAN_MODEL_PATH", str(PROJECT_ROOT / "surakshan_model.pth")))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model checkpoint not found at {MODEL_PATH.resolve()}. "
        "Place surakshan_model.pth in the project root or set SURAKSHAN_MODEL_PATH."
    )

# The checkpoint contains tensors and basic metadata, so safe weights-only loading is enough.
# Older PyTorch versions without this argument use their default loader for compatibility.
try:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
except TypeError:
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
    class_to_idx = checkpoint.get("class_to_idx", {"Accident": 0, "Non Accident": 1})
else:
    state_dict = checkpoint
    # ImageFolder alphabetically maps Accident -> 0 and Non Accident -> 1.
    class_to_idx = {"Accident": 0, "Non Accident": 1}

if "Accident" not in class_to_idx:
    raise ValueError(f"Checkpoint does not define the Accident class: {class_to_idx}")

ACCIDENT_INDEX = int(class_to_idx["Accident"])
IDX_TO_CLASS = {int(index): name for name, index in class_to_idx.items()}

model = AccidentNet().to(DEVICE)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


def predict_frame_details(frame):
    """Return label, predicted-class confidence, and accident probability."""
    if frame is None or not hasattr(frame, "shape") or frame.size == 0:
        raise ValueError("predict_frame_details received an empty frame")

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        predicted_index = int(torch.argmax(probabilities).item())

    label = "ACCIDENT" if predicted_index == ACCIDENT_INDEX else "NORMAL"
    accident_probability = float(probabilities[ACCIDENT_INDEX].item())
    confidence = float(probabilities[predicted_index].item())
    return {
        "label": label,
        "confidence": confidence,
        "accident_probability": accident_probability,
    }


def predict_frame(frame):
    """Backward-compatible API used by the live scripts."""
    return predict_frame_details(frame)["label"]
