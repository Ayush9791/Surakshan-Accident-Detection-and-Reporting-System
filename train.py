"""Train and evaluate a source-video-grouped ResNet-18 accident classifier."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import ResNet18AccidentNet
from video_pipeline.data import (
    CLASS_TO_IDX,
    FrameImageDataset,
    make_frame_group_balanced_sampler,
    read_frame_manifest,
)
from video_pipeline.metrics import (
    binary_metrics,
    choose_recall_threshold,
    classification_report,
    save_confusion_matrix,
    save_history_plots,
)
from video_pipeline.preprocess import load_config

PROJECT_ROOT = Path(__file__).resolve().parent


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def grouped_window_predictions(labels, probabilities, window_ids):
    grouped = defaultdict(list)
    for label, probability, window_id in zip(labels, probabilities, window_ids):
        grouped[str(window_id)].append((int(label), float(probability)))
    output = []
    for window_id, rows in grouped.items():
        group_labels = {row[0] for row in rows}
        if len(group_labels) != 1:
            raise ValueError(f"An annotated video window has conflicting labels: {window_id}")
        output.append((next(iter(group_labels)), float(np.mean([row[1] for row in rows])), window_id))
    return (
        [row[0] for row in output],
        [row[1] for row in output],
        [row[2] for row in output],
    )


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, seen, frame_correct = 0.0, 0, 0
    labels_all, probabilities_all, window_ids_all = [], [], []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for images, labels, _, window_ids in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if amp_enabled else torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits.float(), labels)
            probabilities = logits.float().softmax(1)[:, CLASS_TO_IDX["Accident"]]
            loss_sum += float(loss.item()) * labels.size(0)
            seen += labels.size(0)
            frame_correct += int((logits.argmax(1) == labels).sum().item())
            labels_all.extend(labels.cpu().tolist())
            probabilities_all.extend(probabilities.cpu().tolist())
            window_ids_all.extend(window_ids)
    window_labels, window_probabilities, window_ids = grouped_window_predictions(
        labels_all, probabilities_all, window_ids_all
    )
    return (
        loss_sum / max(1, seen),
        frame_correct / max(1, seen),
        window_labels,
        window_probabilities,
        window_ids,
    )


def make_checkpoint(model, config, threshold, best_epoch, validation_metrics, seed):
    return {
        "model_type": "resnet18_accident",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "class_to_idx": CLASS_TO_IDX,
        "threshold": float(threshold),
        "threshold_objective": "maximum specificity subject to validation accident recall target; otherwise best F1",
        "image_size": int(config["preprocessing"]["image_size"]),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "preprocessing": "OpenCV BGR to RGB; aspect-preserving 224x224 letterbox; ImageNet normalization",
        "architecture": "TorchVision ResNet-18 ImageNet backbone with a new binary head",
        "best_epoch": int(best_epoch),
        "validation_window_metrics": validation_metrics,
        "consecutive_positive_frames": int(config.get("inference", {}).get("consecutive_positive_frames", 3)),
        "seed": int(seed),
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/resnet18_training.json"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    settings = config["training"]
    prep = config["preprocessing"]
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.workers is not None:
        settings["num_workers"] = args.workers

    seed = int(prep["seed"])
    seed_everything(seed)
    dataset_root = (PROJECT_ROOT / config["paths"]["dataset_root"]).resolve()
    model_root = (PROJECT_ROOT / config["paths"]["models_root"]).resolve()
    results_root = (PROJECT_ROOT / config["paths"]["results_root"]).resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    rows = read_frame_manifest(dataset_root)
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    sequence_counts = {}
    for split in ("train", "val", "test"):
        split_rows = by_split[split]
        labels = [row["class_name"] for row in split_rows]
        sequence_counts[split] = len({row["sequence_id"] for row in split_rows})
        print(
            f"{split}: frames={len(split_rows)} windows={sequence_counts[split]} "
            f"Accident={labels.count('Accident')} Non_Accident={labels.count('Non_Accident')}",
            flush=True,
        )
        if not split_rows or {row["label"] for row in split_rows} != {0, 1}:
            raise ValueError(f"{split} split must contain both Accident and Non_Accident frames")
    if not by_split["train"] or not by_split["val"] or not by_split["test"]:
        raise ValueError("Training, validation, and test splits must all contain frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}; "
            f"VRAM={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB",
            flush=True,
        )

    image_size = int(prep["image_size"])
    datasets = {
        split: FrameImageDataset(
            by_split[split], image_size=image_size, training=(split == "train")
        )
        for split in ("train", "val", "test")
    }
    sampler = make_frame_group_balanced_sampler(by_split["train"], seed=seed)
    batch_size = int(settings["batch_size"])
    workers = int(settings["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, sampler=sampler,
            num_workers=workers, pin_memory=device.type == "cuda",
            persistent_workers=workers > 0, worker_init_fn=seed_worker,
            generator=generator,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=device.type == "cuda",
            persistent_workers=workers > 0, worker_init_fn=seed_worker,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=device.type == "cuda",
            persistent_workers=workers > 0, worker_init_fn=seed_worker,
        ),
    }

    try:
        model = ResNet18AccidentNet(
            pretrained_backbone=bool(settings.get("pretrained", True))
        ).to(device)
    except Exception as error:
        raise RuntimeError(
            "Could not initialize pretrained ResNet-18. Check torchvision and network access "
            "to download.pytorch.org, or place resnet18-f37072fd.pth in "
            "torch.hub.get_dir()/checkpoints. Random initialization is not used."
        ) from error

    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in model.backbone.fc.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.layer4.parameters(), "lr": float(settings["backbone_learning_rate"])},
            {"params": model.backbone.fc.parameters(), "lr": float(settings["head_learning_rate"])},
        ],
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    epochs = int(settings["epochs"])
    freeze_epochs = max(0, int(settings.get("freeze_epochs", 2)))
    patience = int(settings["patience"])
    recall_target = float(settings.get("recall_target", 0.90))
    best_score, best_epoch, best_state = -1.0, 0, None
    best_threshold, best_validation_metrics = 0.5, {}
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        fine_tuning = epoch > freeze_epochs
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad_(fine_tuning)

        model.train()
        # Keep the frozen pretrained feature blocks and their batch-norm stats fixed.
        model.backbone.eval()
        model.backbone.fc.train()
        if fine_tuning:
            model.backbone.layer4.train()

        epoch_start = time.time()
        loss_sum, seen, correct = 0.0, 0, 0
        for images, labels, _, _ in loaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if amp_enabled else torch.bfloat16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * labels.size(0)
            seen += labels.size(0)
            correct += int((logits.argmax(1) == labels).sum().item())

        train_loss = loss_sum / max(1, seen)
        train_accuracy = correct / max(1, seen)
        val_loss, val_frame_accuracy, val_labels, val_probs, _ = evaluate(
            model, loaders["val"], device, criterion
        )
        threshold, threshold_reason = choose_recall_threshold(
            val_labels, val_probs, recall_target
        )
        val_metrics = binary_metrics(val_labels, val_probs, threshold)
        score = float(val_metrics["f1"])
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "phase": "layer4_finetune" if fine_tuning else "frozen_backbone_head_only",
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_frame_accuracy": train_accuracy,
            "val_frame_accuracy": val_frame_accuracy,
            "val_window_metrics": val_metrics,
            "threshold_reason": threshold_reason,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            "seconds": round(time.time() - epoch_start, 2),
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs} phase={row['phase']} "
            f"train_loss={train_loss:.4f} train_frame_acc={train_accuracy:.3f} "
            f"val_window_recall={val_metrics['recall']:.3f} "
            f"val_window_precision={val_metrics['precision']:.3f} "
            f"val_window_f1={val_metrics['f1']:.3f} threshold={threshold:.3f} "
            f"({threshold_reason})",
            flush=True,
        )
        atomic_torch_save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "epoch": epoch, "history": history,
             "config": config},
            model_root / "resnet18_last_checkpoint.pth",
        )
        (model_root / "resnet18_training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        save_history_plots(history, results_root, prefix="resnet18_")

        if score > best_score:
            best_score, best_epoch = score, epoch
            best_threshold, best_validation_metrics = threshold, val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            checkpoint = make_checkpoint(
                model, config, threshold, epoch, val_metrics, seed
            )
            checkpoint["state_dict"] = best_state
            checkpoint["validation_threshold_reason"] = threshold_reason
            atomic_torch_save(checkpoint, model_root / "resnet18_best.pth")
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best validation epoch {best_epoch}.", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Training produced no valid ResNet-18 checkpoint")

    # Test data is first read after the model and threshold are selected on validation.
    model.load_state_dict(best_state)
    test_loss, test_frame_accuracy, test_labels, test_probs, test_windows = evaluate(
        model, loaders["test"], device, criterion
    )
    test_metrics = binary_metrics(test_labels, test_probs, best_threshold)
    tn, fp = test_metrics["confusion_matrix"][0]
    fn, tp = test_metrics["confusion_matrix"][1]
    test_metrics.update({
        "evaluation_unit": "annotated temporal window; mean sampled-frame probability",
        "test_loss_by_frame": test_loss,
        "test_frame_accuracy": test_frame_accuracy,
        "threshold_reason": "selected on validation only",
        "best_epoch": best_epoch,
        "source_window_count": len(test_windows),
        "accident_precision": test_metrics["precision"],
        "accident_recall": test_metrics["recall"],
        "non_accident_precision": tn / max(1, tn + fn),
        "false_negative_rate": fn / max(1, fn + tp),
        "false_negatives": fn,
    })
    report, _ = classification_report(
        test_labels, test_probs, best_threshold, unit_name="annotated-window aggregated"
    )
    (results_root / "resnet18_classification_report.txt").write_text(report, encoding="utf-8")
    (results_root / "resnet18_test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    save_confusion_matrix(
        test_metrics, results_root,
        title="Held-out test annotated-window confusion matrix",
        filename="resnet18_confusion_matrix.png",
    )
    best_checkpoint = {
        "model_type": "resnet18_accident",
        "state_dict": best_state,
        "class_to_idx": CLASS_TO_IDX,
        "threshold": float(best_threshold),
        "threshold_objective": "validation accident recall target; fallback best validation F1",
        "image_size": image_size,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "preprocessing": "OpenCV BGR to RGB; aspect-preserving 224x224 letterbox; ImageNet normalization",
        "architecture": "TorchVision ResNet-18 ImageNet backbone with a new binary head",
        "best_epoch": best_epoch,
        "validation_window_metrics": best_validation_metrics,
        "validation_threshold_reason": "selected on validation only",
        "test_metrics": test_metrics,
        "consecutive_positive_frames": int(config.get("inference", {}).get("consecutive_positive_frames", 3)),
        "seed": seed,
        "config": config,
    }
    atomic_torch_save(best_checkpoint, model_root / "resnet18_best.pth")
    print("Held-out test metrics:", json.dumps(test_metrics, indent=2), flush=True)
    print(f"Fresh best checkpoint: {model_root / 'resnet18_best.pth'}", flush=True)


if __name__ == "__main__":
    main()
