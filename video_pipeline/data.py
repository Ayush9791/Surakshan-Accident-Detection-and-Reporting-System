"""Manifest-backed datasets for video sequences represented by extracted frames."""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms


CLASS_TO_IDX = {"Non_Accident": 0, "Accident": 1}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def read_sequence_manifest(dataset_root: Path):
    manifest_path = dataset_root / "sequences_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run `python preprocess_video_dataset.py` first."
        )

    sequences = []
    split_groups = defaultdict(set)
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("status") != "ready":
                continue
            split = row["split"]
            label_name = row["class_name"]
            if split not in {"train", "val", "test"} or label_name not in CLASS_TO_IDX:
                raise ValueError(f"Invalid sequence manifest row: {row}")
            try:
                frame_paths = json.loads(row["frame_paths"])
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(f"Invalid frame list for {row.get('sequence_id')}") from error
            frame_paths = [dataset_root / item for item in frame_paths]
            if not frame_paths:
                raise ValueError(f"Empty frame list for {row.get('sequence_id')}")
            missing = [str(path) for path in frame_paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"Sequence {row.get('sequence_id')} references missing frame: {missing[0]}"
                )
            group_id = row["source_group_id"]
            previous_splits = split_groups[group_id]
            previous_splits.add(split)
            sequences.append({
                "sequence_id": row["sequence_id"],
                "source_group_id": group_id,
                "video_id": row["video_id"],
                "split": split,
                "class_name": label_name,
                "label": CLASS_TO_IDX[label_name],
                "frame_paths": frame_paths,
                "source": row["source"],
            })

    leaking = {group: splits for group, splits in split_groups.items() if len(splits) > 1}
    if leaking:
        examples = list(leaking.items())[:5]
        raise ValueError(f"Source video groups cross splits: {examples}")
    if not sequences:
        raise ValueError(f"No ready sequences in {manifest_path}")
    return sequences


class VideoSequenceDataset(Dataset):
    def __init__(self, sequences, clip_frames=8, image_size=224, training=False, seed=42):
        self.sequences = sequences
        self.clip_frames = int(clip_frames)
        self.image_size = int(image_size)
        self.training = bool(training)
        self.seed = int(seed)
        self._to_tensor = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.ToTensor(),
        ])
        self._normalize = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.sequences)

    def _indices(self, count, sequence_id):
        if count <= 0:
            return np.zeros(self.clip_frames, dtype=np.int64)
        edges = np.linspace(0, count - 1, self.clip_frames)
        if self.training and count > self.clip_frames:
            # Pick a jittered point inside each temporal bin, preserving order.
            rng = random.Random(f"{self.seed}:{sequence_id}:{torch.initial_seed()}")
            indices = []
            for i, center in enumerate(edges):
                left = 0 if i == 0 else (edges[i - 1] + center) / 2
                right = count - 1 if i == len(edges) - 1 else (center + edges[i + 1]) / 2
                indices.append(round(rng.uniform(left, right)))
            return np.asarray(indices, dtype=np.int64).clip(0, count - 1)
        return np.rint(edges).astype(np.int64)

    def __getitem__(self, index):
        item = self.sequences[index]
        paths = item["frame_paths"]
        selected = self._indices(len(paths), item["sequence_id"])
        brightness = 0.92 + torch.rand(()).item() * 0.16 if self.training else 1.0
        frames = []
        for frame_index in selected:
            with Image.open(paths[int(frame_index)]) as source:
                tensor = self._to_tensor(source.convert("RGB"))
            if self.training:
                tensor = (tensor * brightness).clamp_(0.0, 1.0)
            frames.append(self._normalize(tensor))
        clip = torch.stack(frames, dim=0)
        return clip, item["label"], item["source_group_id"]


def make_group_balanced_sampler(sequences, seed=42):
    """Balance labels while preventing long videos from dominating an epoch."""
    label_counts = Counter(item["label"] for item in sequences)
    per_group = Counter((item["label"], item["source_group_id"]) for item in sequences)
    weights = [
        1.0 / max(1, label_counts[item["label"]])
        / max(1, per_group[(item["label"], item["source_group_id"])])
        for item in sequences
    ]
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def split_sequence_counts(sequences):
    counts = {split: Counter() for split in ("train", "val", "test")}
    for item in sequences:
        counts[item["split"]][item["class_name"]] += 1
    return counts
