"""Run Surakshan inference against a local camera or the bundled accident test clip.

The bundled clip is stored at the repository root as Accident_Detection.mp4.
This entry point does not open GUI windows, so it also works on headless hosts.
"""
import os
import time
import uuid
from pathlib import Path

import cv2

from live_system.api import send_alert
from live_system.camera import CameraStream, normalize_camera_source
from live_system.incident import IncidentGate
from live_system.logger import AccidentLogger
from live_system.predict import predict_frame_details, reset_predictor
from live_system.recorder import Recorder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_VIDEO_PATH = PROJECT_ROOT / "Accident_Detection.mp4"
SOURCE_MODE = os.environ.get("SURAKSHAN_CAMERA_SOURCE", "bundled-video").strip().lower()

if SOURCE_MODE == "bundled-video":
    if not BUNDLED_VIDEO_PATH.is_file():
        raise FileNotFoundError(f"Bundled test video not found: {BUNDLED_VIDEO_PATH}")
    CAMERA_SOURCES = {"Bundled Accident Test Video": str(BUNDLED_VIDEO_PATH)}
elif SOURCE_MODE == "pc-camera":
    CAMERA_SOURCES = {"PC Webcam": 0}
elif SOURCE_MODE == "video-file":
    VIDEO_PATH = os.environ.get("SURAKSHAN_VIDEO_PATH", "").strip()
    if not VIDEO_PATH:
        raise RuntimeError("SURAKSHAN_VIDEO_PATH is required for video-file input.")
    CAMERA_SOURCES = {"Video File": normalize_camera_source(VIDEO_PATH)}
elif SOURCE_MODE == "ip-webcam":
    CAMERA_SOURCES = {"IP Webcam": normalize_camera_source(os.environ.get("SURAKSHAN_CAMERA_URL", ""))}
else:
    raise RuntimeError("Unsupported camera source. Choose bundled-video, pc-camera, video-file, or ip-webcam.")


def main():
    streams = {}
    recorders = {}
    gates = {}
    logger = AccidentLogger()
    try:
        for cam_id, source in CAMERA_SOURCES.items():
            print(f"[CAMERA] Opening {cam_id}: {source}", flush=True)
            stream = CameraStream(source)
            streams[cam_id] = stream
            if not stream.wait_until_ready(timeout=8.0):
                raise RuntimeError(f"Input {source!r} opened but no video frames arrived.")
            recorders[cam_id] = Recorder()
            gates[cam_id] = IncidentGate()
            print(f"[SURAKSHAN] Detector input ready: {cam_id}", flush=True)

        print("[SURAKSHAN] Inference loop started.", flush=True)
        frame_count = 0
        while True:
            for cam_id, stream in streams.items():
                ret, frame = stream.read()
                if not ret or frame is None:
                    if SOURCE_MODE in {"bundled-video", "video-file"}:
                        print(f"[VIDEO] End of file reached for {cam_id}.", flush=True)
                        return
                    time.sleep(0.02)
                    continue

                frame_count += 1
                details = predict_frame_details(frame, stream_id=cam_id)
                label = details["label"]
                if details["is_new_prediction"]:
                    print(
                        f"[INFERENCE] frame={frame_count} label={label} "
                        f"accident_probability={details['accident_probability']:.4f} "
                        f"threshold={details['threshold']:.4f}",
                        flush=True,
                    )

                color = (0, 0, 255) if label == "ACCIDENT" else (0, 255, 0)
                cv2.putText(frame, f"{cam_id}: {label} ({details['confidence']:.0%})",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                recorder = recorders[cam_id]
                recorder.add_frame(frame)

                if details["is_new_prediction"] and gates[cam_id].update(label):
                    incident_id = str(uuid.uuid4())
                    print(f"[INCIDENT] Potential accident {incident_id} on {cam_id}", flush=True)
                    logger.log(cam_id, incident_id=incident_id,
                               timestamp=details.get("incident_timestamp"))
                    evidence_path = recorder.save_snapshot(frame, cam_id, incident_id)
                    recorder.start_incident(cam_id)
                    send_alert(cam_id, incident_id=incident_id,
                               confidence=details["accident_probability"],
                               image_path=evidence_path,
                               incident_timestamp=details.get("incident_timestamp"))
    finally:
        for stream in streams.values():
            stream.stop()
        for cam_id in CAMERA_SOURCES:
            reset_predictor(cam_id)
        for recorder in recorders.values():
            recorder.close()


if __name__ == "__main__":
    main()
