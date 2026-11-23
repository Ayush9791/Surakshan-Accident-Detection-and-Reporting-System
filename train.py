"""Train and evaluate Surakshan's source-video-grouped accident classifier."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model import VideoAccidentNet
from video_pipeline.data import (
    CLASS_TO_IDX, VideoSequenceDataset, make_group_balanced_sampler,
    read_sequence_manifest,
)
from video_pipeline.metrics import (
    binary_metrics, choose_recall_threshold, classification_report,
    save_confusion_matrix, save_history_plots,
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


def atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def grouped_video_predictions(labels, probabilities, group_ids):
    grouped = defaultdict(list)
    for label, probability, group_id in zip(labels, probabilities, group_ids):
        grouped[str(group_id)].append((int(label), float(probability)))
    output = []
    for group_id, rows in grouped.items():
        group_labels = {row[0] for row in rows}
        if len(group_labels) != 1:
            raise ValueError(f"A source video group has conflicting labels: {group_id}")
        output.append((next(iter(group_labels)), float(np.mean([row[1] for row in rows])), group_id))
    return ([row[0] for row in output], [row[1] for row in output], [row[2] for row in output])


def evaluate(model, loader, device, criterion):
    model.eval()
    sum_loss, seen, correct = 0.0, 0, 0
    labels_all, probs_all, groups_all = [], [], []
    with torch.inference_mode():
        for clips, labels, group_ids in loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(clips)
                loss = criterion(logits.float(), labels)
            probs = logits.float().softmax(1)[:, 1]
            sum_loss += float(loss.item()) * labels.size(0)
            seen += labels.size(0)
            correct += int((logits.argmax(1) == labels).sum().item())
            labels_all.extend(labels.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())
            groups_all.extend(group_ids)
    group_labels, group_probs, group_ids = grouped_video_predictions(labels_all, probs_all, groups_all)
    return sum_loss / max(1, seen), correct / max(1, seen), group_labels, group_probs, group_ids


def make_checkpoint(model, cfg, threshold, best_epoch, val_metrics, seed):
    return {
        "model_type": "mobilenet_v3_small_gru",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "class_to_idx": CLASS_TO_IDX,
        "threshold": float(threshold),
        "threshold_objective": "maximum specificity subject to validation source-video recall target; otherwise best F1",
        "frame_count": int(cfg["preprocessing"]["clip_frames"]),
        "image_size": int(cfg["preprocessing"]["image_size"]),
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "preprocessing": "RGB 224x224 aspect-preserving letterbox; ImageNet normalization",
        "architecture": "TorchVision MobileNetV3-Small frame encoder + bidirectional GRU + attention head",
        "best_epoch": int(best_epoch), "validation_video_metrics": val_metrics,
        "seed": int(seed), "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/video_training.json"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--smoke-test", action="store_true", help="Run one short epoch and do not open the test set.")
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    settings = cfg["training"]
    prep = cfg["preprocessing"]
    if args.epochs:
        settings["epochs"] = args.epochs
    if args.batch_size:
        settings["batch_size"] = args.batch_size
    if args.workers is not None:
        settings["num_workers"] = args.workers
    if args.smoke_test:
        settings["epochs"] = 1
        settings["patience"] = 1

    seed = int(prep["seed"])
    seed_everything(seed)
    dataset_root = (PROJECT_ROOT / cfg["paths"]["dataset_root"]).resolve()
    model_root = (PROJECT_ROOT / cfg["paths"]["models_root"]).resolve()
    results_root = (PROJECT_ROOT / cfg["paths"]["results_root"]).resolve()
    model_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    all_sequences = read_sequence_manifest(dataset_root)
    by_split = {split: [row for row in all_sequences if row["split"] == split]
                for split in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        labels = [row["class_name"] for row in by_split[split]]
        print(f"{split}: sequences={len(labels)} Accident={labels.count('Accident')} Non_Accident={labels.count('Non_Accident')}", flush=True)
    if not by_split["train"] or not by_split["val"]:
        raise ValueError("Training and validation splits must both have sequences")
    if len(set(row["label"] for row in by_split["train"])) != 2:
        raise ValueError("Training split must contain both classes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}; VRAM={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB", flush=True)
    clip_frames = int(prep["clip_frames"])
    image_size = int(prep["image_size"])
    datasets = {
        split: VideoSequenceDataset(by_split[split], clip_frames=clip_frames,
                                    image_size=image_size, training=(split == "train"), seed=seed)
        for split in ("train", "val", "test")
    }
    sampler = make_group_balanced_sampler(by_split["train"], seed=seed)
    batch_size = int(settings["batch_size"])
    workers = int(settings["num_workers"])
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, sampler=sampler,
                            num_workers=workers, pin_memory=device.type == "cuda",
                            persistent_workers=workers > 0),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False,
                          num_workers=workers, pin_memory=device.type == "cuda",
                          persistent_workers=workers > 0),
    }
    if not args.smoke_test:
        loaders["test"] = DataLoader(datasets["test"], batch_size=batch_size, shuffle=False,
                                     num_workers=workers, pin_memory=device.type == "cuda",
                                     persistent_workers=workers > 0)

    model = VideoAccidentNet(pretrained_backbone=bool(settings.get("pretrained", True))).to(device)
    for parameter in model.frame_encoder.parameters():
        parameter.requires_grad_(False)
    for block in list(model.frame_encoder.children())[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    encoder_params = [p for p in model.frame_encoder.parameters() if p.requires_grad]
    head_params = list(model.temporal.parameters()) + list(model.attention.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": float(settings["backbone_learning_rate"])},
        {"params": head_params, "lr": float(settings["head_learning_rate"])},
    ], weight_decay=float(settings["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_score, best_epoch, stale = -1.0, 0, 0
    best_threshold, threshold_reason = 0.5, "not_tuned"
    history = []
    epochs = int(settings["epochs"])
    patience = int(settings["patience"])
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        model.frame_encoder.eval()
        running_loss, correct, seen = 0.0, 0, 0
        for batch_idx, (clips, labels, _) in enumerate(loaders["train"], start=1):
            if args.smoke_test and batch_idx > 4:
                break
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(clips)
                loss = criterion(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * labels.size(0)
            correct += int((logits.argmax(1) == labels).sum().item())
            seen += labels.size(0)
        train_loss = running_loss / max(1, seen)
        train_acc = correct / max(1, seen)
        val_loss, val_acc, val_labels, val_probs, _ = evaluate(model, loaders["val"], device, criterion)
        threshold, threshold_reason = choose_recall_threshold(
            val_labels, val_probs, float(settings["recall_target"])
        )
        val_metrics = binary_metrics(val_labels, val_probs, threshold)
        score = val_metrics["recall"] if threshold_reason == "max_specificity_at_recall_target" else val_metrics["f1"]
        scheduler.step(score)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "train_accuracy": train_acc, "val_accuracy": val_acc,
               "val_video_metrics": val_metrics, "threshold": threshold,
               "threshold_reason": threshold_reason,
               "learning_rates": [group["lr"] for group in optimizer.param_groups],
               "seconds": round(time.time() - epoch_start, 2)}
        history.append(row)
        print(f"Epoch {epoch}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_video_recall={val_metrics['recall']:.3f} "
              f"val_video_specificity={val_metrics['specificity']:.3f} "
              f"val_video_f1={val_metrics['f1']:.3f} threshold={threshold:.3f} ({threshold_reason})", flush=True)
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "scaler": scaler.state_dict(), "epoch": epoch,
                           "history": history, "config": cfg}, model_root / "last_checkpoint.pth")
        with (model_root / "training_history.json").open("w", encoding="utf-8") as stream:
            json.dump(history, stream, indent=2)
        save_history_plots(history, results_root)

        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_threshold = threshold
            checkpoint = make_checkpoint(model, cfg, threshold, epoch, val_metrics, seed)
            checkpoint["validation_threshold_reason"] = threshold_reason
            atomic_torch_save(checkpoint, model_root / "best_model.pth")
            (model_root / "model_config.json").write_text(json.dumps({
                "model_type": checkpoint["model_type"], "architecture": checkpoint["architecture"],
                "class_to_idx": CLASS_TO_IDX, "threshold": threshold,
                "frame_count": clip_frames, "image_size": image_size,
                "mean": checkpoint["mean"], "std": checkpoint["std"],
                "preprocessing": checkpoint["preprocessing"], "seed": seed,
                "training_config": cfg["training"], "preprocessing_config": cfg["preprocessing"],
            }, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping: best validation epoch {best_epoch}.", flush=True)
                break

    if best_epoch == 0:
        raise RuntimeError("Training produced no valid checkpoint")
    if args.smoke_test:
        # Smoke checks save and reload without reading the test split.
        checkpoint = torch.load(model_root / "best_model.pth", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        _ = model(torch.zeros(1, clip_frames, 3, image_size, image_size, device=device))
        print("Smoke test passed: training, best checkpoint reload, and forward pass; test set not read.", flush=True)
        return

    # Test is first used after training and validation threshold decisions are finalized.
    best = torch.load(model_root / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    test_loss, test_seq_acc, test_labels, test_probs, test_groups = evaluate(
        model, loaders["test"], device, criterion
    )
    metrics = binary_metrics(test_labels, test_probs, best_threshold)
    metrics.update({"test_loss_by_sequence": test_loss, "test_sequence_accuracy": test_seq_acc,
                    "evaluation_unit": "source_video_group_mean_probability",
                    "threshold_reason": threshold_reason, "best_epoch": best_epoch,
                    "source_video_count": len(test_groups)})
    report, _ = classification_report(test_labels, test_probs, best_threshold)
    (results_root / "classification_report.txt").write_text(report, encoding="utf-8")
    (results_root / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(metrics, results_root)
    best["test_metrics"] = metrics
    atomic_torch_save(best, model_root / "best_model.pth")
    print("Final held-out source-video test metrics:", json.dumps(metrics, indent=2), flush=True)
    print(f"Saved best model: {model_root / 'best_model.pth'}", flush=True)


if __name__ == "__main__":
    main()
