"""Run the existing Keras model and incident pipeline on one CCTV video."""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_system import predict as inference
from live_system.api import send_alert
from live_system.incident import IncidentGate
from live_system.logger import AccidentLogger
from live_system.recorder import Recorder


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CCTV video file")
    parser.add_argument("--output", type=Path, help="Annotated video output (default: results/keras/demo_<name>.mp4)")
    parser.add_argument("--camera-id", default="Video Demo")
    parser.add_argument("--sample-interval", type=float, default=0.5, help="Video-time seconds between inference frames")
    parser.add_argument("--threshold", type=float, help="Override the configured accident probability threshold")
    parser.add_argument("--consecutive-positives", type=int, default=3)
    parser.add_argument("--gate-window", type=int, default=10)
    parser.add_argument("--gate-min-positive", type=int, default=6)
    parser.add_argument("--send-alert", action="store_true", help="Send detected incidents to SURAKSHAN_ALERT_URL")
    parser.add_argument("--record-dir", type=Path, default=PROJECT_ROOT / "results" / "keras" / "demo_recordings")
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.input.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")
    if args.sample_interval <= 0:
        raise ValueError("--sample-interval must be positive")
    if args.consecutive_positives < 1:
        raise ValueError("--consecutive-positives must be at least 1")
    if args.threshold is not None:
        if not 0 <= args.threshold <= 1:
            raise ValueError("--threshold must be between 0 and 1")
        inference.ACCIDENT_THRESHOLD = args.threshold
    inference.REQUIRED_CONSECUTIVE_POSITIVES = args.consecutive_positives
    # This runner selects frames according to source-video timestamps; inference
    # must not skip those samples based on the faster offline wall-clock runtime.
    inference.FRAME_INTERVAL_SECONDS = 1e-4
    inference.reset_predictor("video-demo")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, round(fps * args.sample_interval))
    actual_interval = stride / fps

    output = args.output or (PROJECT_ROOT / "results" / "keras" / f"demo_{source.stem}.mp4")
    output = output.resolve()
    if output == source:
        raise ValueError("Output video must not overwrite the source video")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / actual_interval, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create annotated output: {output}")

    gate = IncidentGate(window_size=args.gate_window, min_positive=args.gate_min_positive)
    recorder = Recorder(fps=max(1, round(1.0 / actual_interval)), before=10, after=10, output_dir=args.record_dir)
    logger = AccidentLogger()
    frame_index = 0
    sample_count = 0
    incident_count = 0
    first_positive_time = None
    last_alert_delay = None
    last_probability = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                source_seconds = frame_index / fps
                details = inference.predict_frame_details(frame, stream_id="video-demo")
                sample_count += 1
                probability = float(details["accident_probability"])
                last_probability = probability
                if probability >= inference.ACCIDENT_THRESHOLD:
                    if first_positive_time is None:
                        first_positive_time = source_seconds
                else:
                    first_positive_time = None

                label = details["label"]
                color = (0, 0, 255) if label == "ACCIDENT" else (0, 190, 0)
                cv2.putText(
                    frame,
                    f"{args.camera_id}: {label}  Accident {probability:.1%} / threshold {inference.ACCIDENT_THRESHOLD:.1%}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )
                writer.write(frame)
                recorder.add_frame(frame)

                if details["is_new_prediction"] and gate.update(label):
                    incident_count += 1
                    incident_id = str(uuid.uuid4())
                    last_alert_delay = (
                        source_seconds - first_positive_time if first_positive_time is not None else None
                    )
                    logger.log(
                        args.camera_id,
                        incident_id=incident_id,
                        timestamp=details.get("incident_timestamp"),
                    )
                    evidence_path = recorder.save_snapshot(frame, args.camera_id, incident_id)
                    recorder.start_incident(args.camera_id)
                    print(
                        f"[INCIDENT] source_time={source_seconds:.2f}s confidence={probability:.6f} "
                        f"evidence={evidence_path}", flush=True,
                    )
                    if args.send_alert:
                        accepted = send_alert(
                            args.camera_id,
                            incident_id=incident_id,
                            confidence=probability,
                            image_path=evidence_path,
                            incident_timestamp=details.get("incident_timestamp"),
                        )
                        print(f"[API] accepted={accepted}", flush=True)
            frame_index += 1
    finally:
        capture.release()
        writer.release()
        recorder.close()
        inference.reset_predictor("video-demo")

    duration = frame_index / fps
    print(f"Source: {source}")
    print(f"Samples: {sample_count} at approximately {actual_interval:.3f}s; source duration {duration:.2f}s")
    print(f"Threshold: {inference.ACCIDENT_THRESHOLD:.6f}; consecutive frames: {args.consecutive_positives}; gate: {args.gate_min_positive}/{args.gate_window}")
    print(f"Alerts: {incident_count}; final accident probability: {last_probability:.6f}")
    print(f"Annotated video: {output}")
    if last_alert_delay is not None:
        print(f"Alert delay from first above-threshold sample: {last_alert_delay:.2f}s of source video")
    print("API delivery: enabled" if args.send_alert else "API delivery: disabled")


if __name__ == "__main__":
    main()
