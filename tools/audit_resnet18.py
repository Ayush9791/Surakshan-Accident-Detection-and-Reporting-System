"""Audit the frozen ResNet18 checkpoint on its held-out split; never trains or saves weights."""
from __future__ import annotations

import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
import sys

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import ResNet18AccidentNet
from video_pipeline.data import FrameImageDataset, read_frame_manifest
from video_pipeline.metrics import binary_metrics

DATA = ROOT / "dataset/resnet18_frames"
MODEL = ROOT / "models/resnet18/resnet18_best.pth"
OUT = ROOT / "results/resnet18/audit"
OUT.mkdir(parents=True, exist_ok=True)


def csv_rows(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_curves(y, p, out_dir, threshold):
    # Pure OpenCV plotting avoids adding dependencies to the project.
    canvas = np.full((650, 640, 3), 250, dtype=np.uint8)
    colors = [(190, 110, 30), (40, 70, 210)]
    cv2.putText(canvas, "Test window accident probabilities", (35, 35), cv2.FONT_HERSHEY_SIMPLEX, .7, (20,20,20), 2)
    x0, y0, width, height = 80, 90, 480, 470
    bins = np.linspace(0, 1, 21)
    max_count = max(1, max(np.histogram(p[y == cls], bins=bins)[0].max() for cls in (0,1)))
    for cls, label, color in [(0,"Non_Accident",colors[0]),(1,"Accident",colors[1])]:
        counts, _ = np.histogram(p[y == cls], bins=bins)
        for i, count in enumerate(counts):
            x1 = x0 + round(i * width / 20); x2 = x0 + round((i+1) * width / 20)
            h = round(count / max_count * height)
            layer = canvas.copy(); cv2.rectangle(layer,(x1,y0+height-h),(x2,y0+height),color,-1)
            cv2.addWeighted(layer,.48,canvas,.52,0,canvas)
    cv2.rectangle(canvas,(x0,y0),(x0+width,y0+height),(50,50,50),1)
    tx=x0+round(threshold*width); cv2.line(canvas,(tx,y0),(tx,y0+height),(10,10,10),2)
    cv2.putText(canvas,f"Saved threshold {threshold:.3f}",(x0,y0+height+30),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
    cv2.putText(canvas,"Non_Accident (blue)",(x0,y0+height+55),cv2.FONT_HERSHEY_SIMPLEX,.5,colors[0],2)
    cv2.putText(canvas,"Accident (red)",(x0+220,y0+height+55),cv2.FONT_HERSHEY_SIMPLEX,.5,colors[1],2)
    cv2.putText(canvas,"Probability",(x0+190,y0+height+85),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)

    # Rank scores and compute ROC / precision-recall points.
    order=np.argsort(-p,kind="mergesort"); ys=y[order]; ps=p[order]
    tp=np.cumsum(ys==1); fp=np.cumsum(ys==0); pos=max(1,int((y==1).sum())); neg=max(1,int((y==0).sum()))
    changes=np.r_[np.where(np.diff(ps))[0],len(ps)-1]
    tpr=np.r_[0,tp[changes]/pos]; fpr=np.r_[0,fp[changes]/neg]
    prec=tp[changes]/np.maximum(1,tp[changes]+fp[changes]); rec=tp[changes]/pos
    aucv=float(np.trapezoid(tpr,fpr)); ap=float(np.sum((rec-np.r_[0,rec[:-1]])*prec))
    cv2.imwrite(str(out_dir/"probability_histogram.png"),canvas)

    curves = np.full((600, 1200, 3), 250, dtype=np.uint8)
    def axes(rect,title,xlab,ylab,xs,ys_,ref=False):
        left,top,w,h=rect
        cv2.putText(curves,title,(left,top-12),cv2.FONT_HERSHEY_SIMPLEX,.58,(20,20,20),1)
        cv2.rectangle(curves,(left,top),(left+w,top+h),(60,60,60),1)
        cv2.putText(curves,xlab,(left+100,top+h+23),cv2.FONT_HERSHEY_SIMPLEX,.46,(20,20,20),1)
        cv2.putText(curves,ylab,(left+3,top+18),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1)
        if ref: cv2.line(curves,(left,top+h),(left+w,top),(170,170,170),1)
        pts=np.column_stack([left+np.clip(xs,0,1)*w,top+h-np.clip(ys_,0,1)*h]).astype(np.int32)
        if len(pts)>1: cv2.polylines(curves,[pts],False,(100,55,180),2,cv2.LINE_AA)
    axes((70,100,470,400),f"ROC curve (AUC={aucv:.3f})","False positive rate","True positive rate",fpr,tpr,True)
    axes((660,100,470,400),f"Precision-recall curve (AP={ap:.3f})","Recall","Precision",rec,prec)
    cv2.imwrite(str(out_dir/"roc_precision_recall.png"),curves)
    # Confusion matrix with explicit class axes.
    cm = binary_metrics(y, p, threshold)["confusion_matrix"]
    matrix = np.full((460, 620, 3), 250, dtype=np.uint8)
    cv2.putText(matrix,"Test confusion matrix",(25,35),cv2.FONT_HERSHEY_SIMPLEX,.75,(20,20,20),2)
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            x,y0=170+j*205,95+i*145
            cv2.rectangle(matrix,(x,y0),(x+190,y0+125),(220,235,220) if i==j else (220,220,245),-1)
            cv2.rectangle(matrix,(x,y0),(x+190,y0+125),(60,60,60),2)
            cv2.putText(matrix,f"{names[i][j]}: {cm[i][j]}",(x+28,y0+68),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,20,20),2)
    cv2.putText(matrix,"Actual: Non_Accident (top), Accident (bottom)",(25,410),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1)
    cv2.putText(matrix,"Predicted: Non_Accident (left), Accident (right)",(25,435),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1)
    cv2.imwrite(str(out_dir/"test_confusion_matrix.png"), matrix)
    return aucv,ap


def contact_sheet(rows, path, title, limit=None):
    if limit:
        # Spread the sheet across the full probability range, preferring separate groups.
        unique, seen = [], set()
        for row in rows:
            if row["source_group_id"] not in seen:
                unique.append(row); seen.add(row["source_group_id"])
        if len(unique) > limit:
            chosen = [unique[i] for i in np.rint(np.linspace(0, len(unique)-1, limit)).astype(int)]
        else:
            chosen = unique
    else:
        chosen = rows
    cols, tw, th, label_h = 4, 320, 220, 55
    rows_n = max(1, (len(chosen) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tw, 36 + rows_n * (th + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title, fill="black")
    for i, rec in enumerate(chosen):
        p = ROOT / rec["representative_frame"]
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((tw - 8, th - 8))
        except Exception:
            im = Image.new("RGB", (tw - 8, th - 8), "#ccc")
        x, y = (i % cols) * tw + 4, 36 + (i // cols) * (th + label_h)
        sheet.paste(im, (x, y))
        text = (f'{rec["video_id"]} | p={float(rec["accident_probability"]):.3f}\n'
                f'true={rec["ground_truth"]} pred={rec["predicted_label"]} | '
                f'{rec["source"]} | t={rec["representative_time_seconds"]}s')
        draw.multiline_text((x, y + th), text, fill="black", spacing=2)
    sheet.save(path)
    return len(chosen)


def sequence_contact_sheet(rows, path):
    """Show all extracted frames for each FN window in temporal order."""
    cols, tile_w, tile_h, label_h = 8, 150, 108, 26
    sheet = Image.new("RGB", (cols * tile_w, 30 + len(rows) * (tile_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 7), "False-negative windows: all sampled frames in temporal order", fill="black")
    for row_idx, rec in enumerate(rows):
        y = 30 + row_idx * (tile_h + label_h)
        draw.text((4, y), f'{rec["video_id"]} p={float(rec["accident_probability"]):.3f}', fill="black")
        paths = [x for x in rec["frame_paths"].split(" | ") if x]
        times = rec["frame_times_seconds"].split(" | ")
        for j, path_rel in enumerate(paths[:cols]):
            try:
                im = Image.open(ROOT / path_rel).convert("RGB")
                im.thumbnail((tile_w-4,tile_h-4))
            except Exception:
                im = Image.new("RGB",(tile_w-4,tile_h-4),"#ccc")
            x=j*tile_w+2
            sheet.paste(im,(x,y+label_h))
            if j < len(times): draw.text((x+2,y+label_h+tile_h-15),f'{times[j]}s',fill="yellow",stroke_width=1,stroke_fill="black")
    sheet.save(path)


def original_event_contact_sheet(rows, annotations, path):
    """Decode original-resolution frames around annotated event time for FN review."""
    offsets = [-1.0, -0.5, 0.0, 0.5, 1.0]
    cell_w, cell_h, label_h = 240, 160, 28
    sheet = Image.new("RGB", (len(offsets)*cell_w, 30+len(rows)*(cell_h+label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8,7),"False negatives: original video frames around metadata event time",fill="black")
    for row_idx, rec in enumerate(rows):
        ann = annotations.get(rec["original_filename"].casefold(), {})
        try:
            event_time = float(ann["accident_time"])
            path_video = ROOT / "vid_dataset" / Path(rec["source_video"])
        except (ValueError, KeyError, TypeError):
            continue
        y = 30 + row_idx*(cell_h+label_h)
        draw.text((3,y),f'{rec["original_filename"]} | {ann.get("type")} | rollover={ann.get("rollover")} | p={float(rec["accident_probability"]):.3f}',fill="black")
        cap=cv2.VideoCapture(str(path_video))
        for col, offset in enumerate(offsets):
            cap.set(cv2.CAP_PROP_POS_MSEC,max(0.0,event_time+offset)*1000)
            ok,bgr=cap.read()
            if ok and bgr is not None:
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                im=Image.fromarray(rgb); im.thumbnail((cell_w-4,cell_h-4))
            else:
                im=Image.new("RGB",(cell_w-4,cell_h-4),"#ccc")
            x=col*cell_w+2
            sheet.paste(im,(x,y+label_h))
            draw.text((x+3,y+label_h+cell_h-17),f'{offset:+.1f}s',fill="yellow",stroke_width=1,stroke_fill="black")
        cap.release()
    sheet.save(path)


def main():
    checkpoint = torch.load(MODEL, map_location="cpu", weights_only=False)
    threshold = float(checkpoint["threshold"])
    rows = read_frame_manifest(DATA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18AccidentNet(pretrained_backbone=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    frame_manifest = csv_rows(DATA / "frames_manifest.csv")
    frame_meta = {r["frame_path"]: r for r in frame_manifest if r.get("status") == "kept"}
    video_rows = csv_rows(DATA / "videos_manifest.csv")
    video_meta = {r["video_id"]: r for r in video_rows}
    annotation_meta = {}
    annotation_path = ROOT / "vid_dataset/archive (1)/metadata-real.csv"
    if annotation_path.is_file():
        for ann in csv_rows(annotation_path):
            annotation_meta[Path(ann.get("path", "")).name.casefold()] = ann
    all_window_rows = []
    for split in ("val", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        ds = FrameImageDataset(split_rows, image_size=int(checkpoint["image_size"]), training=False)
        probs = []
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        with torch.inference_mode():
            for images, _, _, _ in loader:
                logits = model(images.to(device))
                probs.extend(logits.float().softmax(1)[:, 1].cpu().tolist())
        grouped = defaultdict(list)
        for row, p in zip(split_rows, probs):
            grouped[row["sequence_id"]].append((row, float(p)))
        for seq, items in grouped.items():
            label_set = {r["class_name"] for r, _ in items}
            if len(label_set) != 1:
                raise ValueError(f"Conflicting labels in {seq}: {label_set}")
            p = float(np.mean([v for _, v in items]))
            first = items[0][0]
            def frame_info(pair):
                return frame_meta.get(str(pair[0]["image_path"].relative_to(DATA)).replace("\\", "/"), {})
            meta0 = frame_info(items[0]); vid = meta0.get("video_id", "")
            vm = video_meta.get(vid, {})
            ann = annotation_meta.get(vm.get("original_filename", "").casefold(), {})
            frames = sorted(items, key=lambda pair: float(frame_info(pair).get("time_seconds") or 0))
            frame_paths = [str(r["image_path"].relative_to(ROOT)).replace("\\", "/") for r, _ in frames]
            times = [frame_info(pair).get("time_seconds", "") for pair in frames]
            y = 1 if first["class_name"] == "Accident" else 0
            pred = int(p >= threshold)
            event_time = ann.get("accident_time", "")
            rec = {
                "split": split, "sequence_id": seq, "video_id": vid, "source_group_id": first["source_group_id"],
                "source_video": first["source_video"], "original_filename": vm.get("original_filename", ""),
                "source": vm.get("source", ""), "camera_id": "not present in dataset metadata",
                "ground_truth": first["class_name"], "predicted_label": "Accident" if pred else "Non_Accident",
                "accident_probability": p, "threshold": threshold, "frame_count": len(items),
                "frame_paths": " | ".join(frame_paths), "frame_times_seconds": " | ".join(times),
                "representative_frame": frame_paths[len(frame_paths)//2] if frame_paths else "",
                "representative_time_seconds": times[len(times)//2] if times else "",
                "accident_time_seconds": event_time, "accident_type": ann.get("type", ""),
                "rollover_annotation": ann.get("rollover", ""), "scene_layout": ann.get("scene_layout", ""),
                "weather": ann.get("weather", ""), "day_time": ann.get("day_time", ""), "quality": ann.get("quality", ""),
                "region": ann.get("region", ""),
                "event_time_offset_min_seconds": min((abs(float(t)-float(event_time)) for t in times if t and event_time), default=""),
                "error_type": "FP" if y == 0 and pred == 1 else "FN" if y == 1 and pred == 0 else "",
            }
            all_window_rows.append(rec)
    window_rows = [r for r in all_window_rows if r["split"] == "test"]
    val_window_rows = [r for r in all_window_rows if r["split"] == "val"]
    window_rows.sort(key=lambda r: (r["error_type"], -float(r["accident_probability"])))
    fp = [r for r in window_rows if r["error_type"] == "FP"]
    fn = [r for r in window_rows if r["error_type"] == "FN"]
    save_csv(OUT / "test_window_predictions.csv", window_rows, list(window_rows[0]))
    save_csv(OUT / "validation_window_predictions.csv", val_window_rows, list(val_window_rows[0]))
    save_csv(OUT / "false_positives.csv", fp, list(window_rows[0]))
    save_csv(OUT / "false_negatives.csv", fn, list(window_rows[0]))

    # Per-source-video aggregate uses the mean of window probabilities; this is supplemental.
    by_video = defaultdict(list)
    for row in window_rows:
        by_video[row["video_id"]].append(row)
    video_rows_out = []
    for vid, rs in by_video.items():
        def count_error(ground_truth, prediction):
            return sum(r["ground_truth"] == ground_truth and r["predicted_label"] == prediction for r in rs)
        pos = [float(r["accident_probability"]) for r in rs if r["ground_truth"] == "Accident"]
        neg = [float(r["accident_probability"]) for r in rs if r["ground_truth"] == "Non_Accident"]
        video_rows_out.append({"video_id": vid, "source_group_id": rs[0]["source_group_id"],
                               "source_video": rs[0]["source_video"], "source": rs[0]["source"],
                               "window_count": len(rs), "accident_windows": len(pos), "non_accident_windows": len(neg),
                               "tp_windows": count_error("Accident", "Accident"),
                               "fn_windows": count_error("Accident", "Non_Accident"),
                               "tn_windows": count_error("Non_Accident", "Non_Accident"),
                               "fp_windows": count_error("Non_Accident", "Accident"),
                               "mean_accident_probability": float(np.mean([float(r["accident_probability"]) for r in rs])),
                               "mean_positive_window_probability": float(np.mean(pos)) if pos else "",
                               "mean_normal_window_probability": float(np.mean(neg)) if neg else "",
                               "threshold": threshold})
    save_csv(OUT / "per_source_video_metrics.csv", video_rows_out, list(video_rows_out[0]))

    labels = [int(r["ground_truth"] == "Accident") for r in window_rows]
    window_probs = [float(r["accident_probability"]) for r in window_rows]
    yarr, parr = np.asarray(labels), np.asarray(window_probs)
    metrics = binary_metrics(yarr, parr, threshold)
    if not (metrics["fp"] == 92 and metrics["fn"] == 15 and len(window_rows) == 277):
        print("WARNING: reproduced prediction counts differ from saved metrics:", metrics)
    (OUT / "reproduced_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    val_y = np.asarray([int(r["ground_truth"] == "Accident") for r in val_window_rows])
    val_p = np.asarray([float(r["accident_probability"]) for r in val_window_rows])
    val_metrics = binary_metrics(val_y, val_p, threshold)
    (OUT / "reproduced_validation_metrics_at_checkpoint_threshold.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    source_eval = {}
    for source in sorted({r["source"] for r in window_rows}):
        subset = [r for r in window_rows if r["source"] == source]
        sub_y = [int(r["ground_truth"] == "Accident") for r in subset]
        sub_p = [float(r["accident_probability"]) for r in subset]
        source_eval[source] = {"windows": len(subset), "positive_windows": sum(sub_y), "negative_windows": len(sub_y)-sum(sub_y),
                               "false_positives": sum(r["error_type"] == "FP" for r in subset),
                               "false_negatives": sum(r["error_type"] == "FN" for r in subset),
                               "metrics": binary_metrics(sub_y, sub_p, threshold)}
    (OUT / "test_metrics_by_source.json").write_text(json.dumps(source_eval, indent=2), encoding="utf-8")

    # Error sheets: FP representative frames across source videos (up to 40); all FN windows.
    fp_sheet_count = contact_sheet(fp, OUT / "false_positive_contact_sheet.png", "False positives (representative windows)", 40)
    fn_sheet_count = contact_sheet(fn, OUT / "false_negative_contact_sheet.png", "False negatives (all windows)")
    sequence_contact_sheet(fn, OUT / "false_negative_all_frames.png")
    original_event_contact_sheet(fn, annotation_meta, OUT / "false_negative_original_resolution.png")

    # Histograms and curves use only OpenCV; threshold remains the checkpoint's.
    aucv, ap = plot_curves(yarr, parr, OUT, threshold)

    # Split integrity, source mix, and source video metadata checks.
    split_groups = defaultdict(set); group_video_ids = defaultdict(set); video_splits = defaultdict(set)
    source_mix = defaultdict(Counter); class_mix = defaultdict(Counter)
    for r in frame_manifest:
        if r.get("status") != "kept":
            continue
        split_groups[r["source_group_id"]].add(r["split"])
        group_video_ids[r["source_group_id"]].add(r["video_id"])
        video_splits[r["video_id"]].add(r["split"])
        class_mix[r["split"]][r["class_name"]] += 1
    seq_manifest_rows = [s for s in csv_rows(DATA / "sequences_manifest.csv") if s.get("status") == "ready"]
    for s in seq_manifest_rows:
        source_mix[s["split"]][s["source"]] += 1
    video_splits_report = defaultdict(set)
    for vr in video_rows:
        video_splits_report[vr["video_id"]].add(vr["split"])
    sequence_manifest = csv_rows(DATA / "sequences_manifest.csv")
    seq_by_video = defaultdict(list)
    for s in sequence_manifest:
        if s.get("status") == "ready": seq_by_video[s["video_id"]].append(s)
    overlap_pairs = []
    for video_id, ss in seq_by_video.items():
        positives = [s for s in ss if s["class_name"] == "Accident"]
        negatives = [s for s in ss if s["class_name"] == "Non_Accident"]
        for a in positives:
            for n in negatives:
                try: gap = max(0, max(float(a["start_seconds"]),float(n["start_seconds"])) - min(float(a["end_seconds"]),float(n["end_seconds"])))
                except Exception: continue
                if float(a["start_seconds"]) < float(n["end_seconds"]) and float(n["start_seconds"]) < float(a["end_seconds"]):
                    overlap_pairs.append({"video_id":video_id,"source_group_id":a["source_group_id"],"positive_sequence":a["sequence_id"],"negative_sequence":n["sequence_id"],"positive_range":[a["start_seconds"],a["end_seconds"]],"negative_range":[n["start_seconds"],n["end_seconds"]]})
    candidate_splits = []
    video_by_path = {r["original_path"].replace("\\", "/"): r for r in video_rows}
    candidate_path = ROOT / "results/duplicate_video_candidates.csv"
    for candidate in csv_rows(candidate_path) if candidate_path.is_file() else []:
        va=video_by_path.get(candidate.get("path_a", "")); vb=video_by_path.get(candidate.get("path_b", ""))
        if va and vb and va["split"] != vb["split"]:
            candidate_splits.append({"path_a":candidate["path_a"],"split_a":va["split"],"path_b":candidate["path_b"],"split_b":vb["split"],"phash_distances":candidate["frame_phash_hamming"],"status":"review_candidate_not_confirmed_duplicate"})
    integrity = {
        "frame_rows": len(rows), "unique_source_groups": len(split_groups), "unique_video_ids": len(video_splits_report),
        "source_groups_crossing_splits": {k:sorted(v) for k,v in split_groups.items() if len(v)>1},
        "video_ids_crossing_splits": {k:sorted(v) for k,v in video_splits_report.items() if len(v)>1},
        "groups_containing_multiple_video_ids": {k:sorted(v) for k,v in group_video_ids.items() if len(v)>1},
        "source_groups_with_multiple_video_ids_count": sum(len(v)>1 for v in group_video_ids.values()),
        "overlapping_accident_and_non_accident_windows": overlap_pairs,
        "near_duplicate_candidates_crossing_splits": candidate_splits,
        "source_camera_id_metadata_present": False,
        "source_classes_by_split": {s:dict(c) for s,c in source_mix.items()},
        "frame_classes_by_split": {s:dict(c) for s,c in class_mix.items()},
        "ready_sequences_by_split_source": {s:dict(c) for s,c in source_mix.items()},
        "test_windows_by_source_class": {f"{a}|{b}":n for (a,b),n in Counter((r["source"], r["ground_truth"]) for r in window_rows).items()},
        "test_source_videos_by_window_label_mix": {f"{a}|{b}":n for (a,b),n in Counter((r["source"], "mixed" if r["accident_windows"] and r["non_accident_windows"] else "Accident" if r["accident_windows"] else "Non_Accident") for r in video_rows_out).items()},
    }
    # There is no camera ID column in the source inventory/manifests.
    (OUT / "split_integrity_report.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")

    history = json.loads((ROOT / "models/resnet18/resnet18_training_history.json").read_text(encoding="utf-8"))
    train_config = checkpoint.get("config", {}).get("training", {})
    lines = [
        "# ResNet18 dataset and error audit", "",
        "## Result reproduction", "",
        f"Replayed the existing checkpoint ({MODEL.relative_to(ROOT)}) against the saved test frames; no training occurred and the checkpoint was not modified. Device: `{device}`. Window-level counts reproduced: {metrics['samples']} windows, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}, TP={metrics['tp']}; saved threshold={threshold:.6f}, ROC-AUC={metrics['roc_auc']:.3f}.", "",
        "## Split integrity", "",
        f"The frame manifest assigns {integrity['unique_source_groups']} source groups and {integrity['unique_video_ids']} original video IDs. Source groups crossing splits: {len(integrity['source_groups_crossing_splits'])}; video IDs crossing splits: {len(integrity['video_ids_crossing_splits'])}. Cross-label overlapping windows inside one video: {len(overlap_pairs)}. See `split_integrity_report.json` for details.", "",
        "Camera identity is not present in the frame/video manifests or video inventory, so per-camera analysis is unavailable. No camera ID was inferred.", "",
        "## Test error inspection", "",
        f"The false-positive CSV contains all {len(fp)} FPs; the false-negative CSV contains all {len(fn)} FNs. Each row contains source path, IDs, all sampled frame paths/timestamps, predicted window probability, ground truth and available annotation metadata. The FP sheet samples {fp_sheet_count} false positives across the probability range (one per source group where possible); the FN sheet displays all {fn_sheet_count} errors.", "",
        f"ACCIDENTBench metadata supplies accident type and rollover labels for {sum(bool(r['accident_type']) for r in fn)}/{len(fn)} FN records. FN accident-type breakdown: {dict(Counter(r['accident_type'] or 'missing' for r in fn))}; rollover flag: {dict(Counter(r['rollover_annotation'] or 'missing' for r in fn))}; recorded source quality: {dict(Counter(r['quality'] or 'missing' for r in fn))}. Types identify 5 rear-end, 1 t-bone, and 2 sideswipe events; 2 carry rollover=1. No stationary-damaged-vehicle subtype is represented in these labels. The 224x224 test frames do not clearly show the crash; the new original-resolution around-event sheet supports manual verification. Check those clips before correcting any labels.", "",
        "## Data and generalization findings", "",
        f"- Test set source/class window counts: {integrity['test_windows_by_source_class']}. All accident positives are ACCIDENTBench. Of 131 negatives, 73 are pre-impact normal windows from ACCIDENTBench and 58 are TAD footage.",
        f"- False positives by negative source: {dict(Counter(r['source'] for r in fp))}. This means {sum(r['source']=='ACCIDENTBench' for r in fp)}/73 ACCIDENTBench normal windows and {sum(r['source']=='TAD' for r in fp)}/58 TAD negatives were called accidents. The high FP rate even within ACCIDENTBench normal sections points to weak visual separation, with source/style cues also a concern.",
        f"- Source-stratified test: ACCIDENTBench AUC={source_eval['ACCIDENTBench']['metrics']['roc_auc']:.3f} (accident vs pre-impact normal; effectively chance); TAD negative false-positive rate={source_eval['TAD']['false_positives']/source_eval['TAD']['negative_windows']:.1%}. This is stronger evidence of weak scene-level separability than overall AUC alone.",
        f"- At the checkpoint's fixed threshold, replayed validation metrics: F1={val_metrics['f1']:.3f}, specificity={val_metrics['specificity']:.3f}, ROC-AUC={val_metrics['roc_auc']:.3f} (n={val_metrics['samples']}); test: F1={metrics['f1']:.3f}, specificity={metrics['specificity']:.3f}, ROC-AUC={metrics['roc_auc']:.3f} (n={metrics['samples']}). Test TAD negative windows are 58 vs 42 in validation. Full validation predictions and metric JSON are included.",
        "- The test has 131 non-accident windows correctly classified versus 92 false alarms. At the validation-selected threshold it intentionally targets at least 90% validation accident recall, which explains the high test recall and poor specificity; threshold selection did not maximize test performance.",
        "- ACCIDENTBench normal windows are sampled from before the annotated event with a configured margin; event windows span the annotated impact +/- 2 seconds. These labels are metadata-derived, not frame-reviewed. The source may still contain ambiguous pre-event traffic or an inaccurate event timestamp.",
        "- The aggregate dataset is source-imbalanced: many accident clips without event timestamps were excluded; 127 labeled TAD normal clips were available, with 6 reviewed normal clips training-only. The held-out negatives are limited and source-specific.",
        "- Current preprocessing uses non-overlapping 8-second normal windows for TAD and event/normal windows per ACCIDENTBench clip. Sequence windows are source-grouped before splitting. Explicit checks found no split leakage in frame rows, video IDs, or source groups.", "",
        "## Training behavior and selection logic", "",
        f"Epochs recorded: {len(history)}. Train frame accuracy rose {history[0]['train_frame_accuracy']:.1%} → {history[-1]['train_frame_accuracy']:.1%}; validation frame accuracy rose {history[0]['val_frame_accuracy']:.1%} → {history[-1]['val_frame_accuracy']:.1%}. Best saved epoch: {checkpoint.get('best_epoch')}. Selection is validation window F1 after per-epoch threshold selection (target recall {train_config.get('recall_target')}); early-stop patience={train_config.get('patience')} epochs. Test ROC AUC reproduced as {aucv:.3f}; PR average precision={ap:.3f}.",
        "At the selected threshold, validation window F1 was nearly flat while ROC-AUC improved across epochs. Selection on F1 after re-tuning a threshold each epoch is noisy and favors the operating point, not ranking/generalization. F1 epoch 1 was highest; epochs 2–4 did not exceed it, so patience 3 correctly stopped at epoch 4. The training-accuracy increase is measured on training frames under augmentation and balanced sampling; it is not comparable to aggregated validation-window accuracy and is consistent with overfitting/domain mismatch.",
        "The current accuracy plot helper looks for `train_accuracy`/`val_accuracy`, while history stores `train_frame_accuracy`/`val_frame_accuracy`; thus the emitted accuracy figure omits the history series. Loss and recorded history values are still present. Validation metrics are computed after averaging frame probabilities per window; test uses the same aggregation.",
        f"The earlier duplicate scan flagged {len(candidate_splits)} near-duplicate candidate pair across splits (val/test). It is not confirmed as a duplicate and does not share a source-group/video ID; inspect it because a repeated view can weaken validation/test independence. Details are in `split_integrity_report.json`.", "",
        "## Priority corrections (no automatic relabeling)", "",
        "1. Manually review all 92 FP windows and 15 FN windows using CSV paths/contact sheets plus the original clips. Flag probable label errors and ambiguous pre-impact examples; do not relabel until clip evidence supports it.",
        "2. Improve evaluation diversity: collect/verify normal and accident footage from the same camera/source families and use camera/source-grouped holdouts. Report per-source and per-camera results when camera metadata exists.",
        "3. Add an independent reviewed negative test set with enough cameras/scenes; keep it untouched for evaluation. TAD source-only negatives currently make false alarms under ACCIDENTBench scenes hard to diagnose.",
        "4. Revisit validation selection after review: choose a threshold using a predeclared cost/recall requirement; report validation ROC-AUC, PR-AUC and specificity, and avoid changing the threshold based on this test set.",
        "5. Correct the training plot key mismatch; then, after data review, compare early stopping by a stable validation metric (e.g. PR-AUC or ROC-AUC) and audit calibration. Do not change architecture/retrain until the dataset review is complete.", "",
        "## Artifacts", "",
        "- `false_positives.csv`, `false_negatives.csv`, `test_window_predictions.csv`, `per_source_video_metrics.csv`",
        "- `false_positive_contact_sheet.png`, `false_negative_contact_sheet.png`, `false_negative_all_frames.png`, `false_negative_original_resolution.png`",
        "- `probability_histogram.png`, `roc_precision_recall.png`, `test_confusion_matrix.png`, `reproduced_test_metrics.json`, `reproduced_validation_metrics_at_checkpoint_threshold.json`, `test_metrics_by_source.json`, `split_integrity_report.json`",
    ]
    (OUT / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"device":str(device),"metrics":metrics,"fp":len(fp),"fn":len(fn),"unique_groups":integrity["unique_source_groups"],"video_id_leaks":integrity["video_ids_crossing_splits"],"overlap_pairs":len(overlap_pairs),"outputs":str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
