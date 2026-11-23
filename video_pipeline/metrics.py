"""Dependency-light metrics and plots for the binary video classifier."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    pred = probabilities >= float(threshold)
    pos = labels == 1
    tp, fp = int(np.sum(pred & pos)), int(np.sum(pred & ~pos))
    tn, fn = int(np.sum(~pred & ~pos)), int(np.sum(~pred & pos))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    auc = roc_auc(labels, probabilities)
    return {
        "accuracy": (tp + tn) / max(1, len(labels)),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision, "recall": recall, "specificity": specificity,
        "f1": f1, "roc_auc": auc, "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]], "samples": int(len(labels)),
    }


def roc_auc(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    if not pos or not neg:
        return None
    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    sorted_values = np.asarray(probabilities)[order]
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and sorted_values[end] == sorted_values[index]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2
        index = end
    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def choose_recall_threshold(labels, probabilities, recall_target=0.90):
    """Choose the highest-specificity threshold that meets validation recall target."""
    candidates = np.unique(np.concatenate(([0.0], np.asarray(probabilities), [1.0])))
    rows = [binary_metrics(labels, probabilities, value) for value in candidates]
    eligible = [(row["specificity"], row["balanced_accuracy"], -row["threshold"], row)
                for row in rows if row["recall"] >= recall_target]
    if eligible:
        return max(eligible)[-1]["threshold"], "max_specificity_at_recall_target"
    best = max(rows, key=lambda row: (row["f1"], row["balanced_accuracy"], -row["threshold"]))
    return best["threshold"], "best_f1_recall_target_unmet"


def classification_report(labels, probabilities, threshold):
    metrics = binary_metrics(labels, probabilities, threshold)
    matrix = metrics["confusion_matrix"]
    names = ("Non_Accident", "Accident")
    lines = ["Classification report (source-video aggregated)", "", f"Threshold: {threshold:.4f}", "",
             f"{'class':<18}{'precision':>11}{'recall':>11}{'f1-score':>11}{'support':>11}"]
    for index, name in enumerate(names):
        tp = matrix[index][index]
        fp = matrix[1 - index][index]
        fn = matrix[index][1 - index]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        support = int(np.sum(np.asarray(labels) == index))
        lines.append(f"{name:<18}{precision:>11.4f}{recall:>11.4f}{f1:>11.4f}{support:>11d}")
    lines.extend(["", json.dumps(metrics, indent=2)])
    return "\n".join(lines), metrics


def _canvas(width=900, height=560, title=""):
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(image, title, (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    return image


def save_history_plots(history, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    epochs = [item["epoch"] for item in history]
    for key, filename, title in [
        ("train_loss", "training_loss.png", "Training loss"),
        ("val_loss", "validation_loss.png", "Validation loss"),
        ("train_accuracy", "accuracy.png", "Training and validation accuracy"),
    ]:
        image = _canvas(title=title)
        plot = (65, 70, 840, 475)
        x0, y0, x1, y1 = plot
        cv2.rectangle(image, (x0, y0), (x1, y1), (80, 80, 80), 1)
        values = [item.get(key) for item in history]
        if key == "train_accuracy":
            series = [(values, (200, 80, 30), "train")]
            val = [item.get("val_accuracy") for item in history]
            series.append((val, (40, 150, 40), "validation"))
            ymin, ymax = 0.0, 1.0
        else:
            series = [(values, (200, 80, 30), key)]
            ymax = max([float(v) for v in values if v is not None] + [0.1]) * 1.1
            ymin = 0.0
        for series_index, (values_line, color, label) in enumerate(series):
            points = []
            for idx, value in enumerate(values_line):
                if value is None:
                    continue
                x = x0 + (x1 - x0) * (idx / max(1, len(epochs) - 1))
                y = y1 - (y1 - y0) * ((float(value) - ymin) / max(1e-9, ymax - ymin))
                points.append((int(x), int(y)))
            if len(points) > 1:
                cv2.polylines(image, [np.asarray(points, np.int32)], False, color, 3, cv2.LINE_AA)
            for point in points:
                cv2.circle(image, point, 4, color, -1)
            cv2.putText(image, label, (x0 + 10 + series_index * 160, y0 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.imwrite(str(results_dir / filename), image)


def save_confusion_matrix(metrics, results_dir: Path):
    image = _canvas(700, 520, "Held-out test confusion matrix (video-level)")
    matrix = metrics["confusion_matrix"]
    labels = (("TN", "FP"), ("FN", "TP"))
    colors = ((190, 225, 190), (190, 190, 240))
    for row in range(2):
        for col in range(2):
            x, y = 190 + col * 230, 120 + row * 160
            cv2.rectangle(image, (x, y), (x + 210, y + 140), colors[row][col], -1)
            cv2.rectangle(image, (x, y), (x + 210, y + 140), (50, 50, 50), 2)
            cv2.putText(image, f"{labels[row][col]}: {matrix[row][col]}", (x + 25, y + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.imwrite(str(results_dir / "confusion_matrix.png"), image)
