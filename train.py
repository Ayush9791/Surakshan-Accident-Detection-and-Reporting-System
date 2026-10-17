"""Train Surakshan's frame classifier with validation and a held-out test split.

Dataset layout:
    dataset/
      Accident/
      Non Accident/

For a trustworthy evaluation, split by source video/scene where possible. This
script splits individual images because the current repository does not retain
source-video metadata; adjacent frames may therefore leak across splits.
"""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import AccidentNet

DATASET_DIR = Path("dataset")
OUTPUT_PATH = Path("surakshan_model.pth")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 42
NUM_WORKERS = 0


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    true_positive = false_positive = false_negative = 0
    accident_index = None

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            if accident_index is not None:
                true_positive += ((predictions == accident_index) & (labels == accident_index)).sum().item()
                false_positive += ((predictions == accident_index) & (labels != accident_index)).sum().item()
                false_negative += ((predictions != accident_index) & (labels == accident_index)).sum().item()

    accuracy = correct / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": total_loss / total if total else 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR.resolve()}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(str(DATASET_DIR), transform=transform)
    required = {"Accident", "Non Accident"}
    if set(dataset.classes) != required:
        raise ValueError(
            f"Expected class folders {sorted(required)}, found {dataset.classes}. "
            "The class mapping must match the prediction code."
        )
    if len(dataset) < 10:
        raise ValueError(f"Dataset is too small for a split: {len(dataset)} images")

    class_to_idx = dataset.class_to_idx
    generator = torch.Generator().manual_seed(SEED)
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {class_to_idx}")
    print(f"Images: train={len(train_set)}, validation={len(val_set)}, test={len(test_set)}")
    print("Evaluation warning: random image splitting may leak adjacent frames between sets.")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = AccidentNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

        val_metrics = evaluate(model, val_loader, device, criterion)
        train_loss = total_loss / total if total else 0.0
        train_accuracy = correct / total if total else 0.0
        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_precision={val_metrics['precision']:.3f} "
            f"val_recall={val_metrics['recall']:.3f} val_f1={val_metrics['f1']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, criterion)
    print("\nHeld-out test metrics (run once after model selection):")
    for name, value in test_metrics.items():
        print(f"  {name}: {value:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "image_size": [128, 128],
        "metrics": test_metrics,
        "seed": SEED,
    }, OUTPUT_PATH)
    print(f"Saved best validation checkpoint to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
