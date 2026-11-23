"""Aspect-preserving preprocessing for fixed-camera R3D video clips."""
import cv2
import numpy as np
import torch


MEAN = (0.43216, 0.394666, 0.37645)
STD = (0.22803, 0.22145, 0.216989)


def letterbox_rgb_frame(frame_rgb, size=112):
    """Resize a full frame into a square canvas without cropping its edges."""
    height, width = frame_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Expected a non-empty RGB frame")
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, round(width * scale)))
    resized_height = max(1, min(size, round(height * scale)))
    resized = cv2.resize(
        frame_rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def normalize_rgb_clip(frames_rgb, size=112, device=None):
    """Convert ordered RGB frames to normalized R3D input (C, T, H, W)."""
    boxed = np.stack([letterbox_rgb_frame(frame, size) for frame in frames_rgb])
    clip = torch.from_numpy(boxed).permute(3, 0, 1, 2).to(dtype=torch.float32) / 255.0
    mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1, 1)
    clip = (clip - mean) / std
    return clip.to(device) if device is not None else clip
