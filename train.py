"""Train Surakshan's frame classifier with validation and a held-out test split."""
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


def evaluate(model, loader, device, criterion, accident_index):
    model.eval()
    total_loss = total = correct = 0
    true_positive = false_positive = false_negative = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            true_positive += ((predictions == accident_index) & (labels == accident_index)).sum().item()
            false_positive += ((predictions == accident_index) & (labels != accident_index)).sum().item()
            false_negative += ((predictions != accident_index) & (labels == accident_index)).sum().item()

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": total_loss / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR.resolve()}")
    dataset = datasets.ImageFolder(
        str(DATASET_DIR),
        transform=transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]),
    )
    if set(dataset.classes) != {"Accident", "Non Accident"}:
        raise ValueError(f"Expected Accident and Non Accident folders; found {dataset.classes}")
    if len(dataset) < 10:
        raise ValueError(f"Dataset is too small for a split: {len(dataset)} images")

    class_to_idx = dataset.class_to_idx
    accident_index = class_to_idx["Accident"]
    generator = torch.Generator().manual_seed(SEED)
    train_size = int(0.70 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; class mapping: {class_to_idx}")
    print(f"Images: train={len(train_set)}, validation={len(val_set)}, test={len(test_set)}")
    print("WARNING: split is image-level. Group by source video/scene before reporting real-world performance.")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    model = AccidentNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        loss_sum = samples = correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            samples += labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

        val = evaluate(model, val_loader, device, criterion, accident_index)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} train_loss={loss_sum / max(samples, 1):.4f} "
            f"train_acc={correct / max(samples, 1):.3f} val_loss={val['loss']:.4f} "
            f"val_acc={val['accuracy']:.3f} val_precision={val['precision']:.3f} "
            f"val_recall={val['recall']:.3f} val_f1={val['f1']:.3f}"
        )
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_state = deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device, criterion, accident_index)
    print("Held-out test metrics:")
    for key, value in test.items():
        print(f"  {key}: {value:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "image_size": [128, 128],
        "test_metrics": test,
        "seed": SEED,
    }, OUTPUT_PATH)
    print(f"Saved best validation checkpoint to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
