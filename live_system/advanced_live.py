"""Advanced live monitoring entry point.

Set SURAKSHAN_CAMERA_URL to a camera index or stream URL before starting the API.
For the Android IP Webcam app, use the HTTP MJPEG endpoint, e.g.
http://192.168.1.105:8080/video (replace the IP with the phone's actual address).
"""
import os
import uuid

import cv2

from live_system.api import send_alert
from live_system.camera import CameraStream, normalize_camera_source
from live_system.incident import IncidentGate
from live_system.logger import AccidentLogger
from live_system.predict import predict_frame_details, reset_predictor
from live_system.recorder import Recorder


# The API passes the selected source through environment variables.
# All inputs use the same existing frame predictor and model weights.
SOURCE_MODE = os.environ.get("SURAKSHAN_CAMERA_SOURCE", "ip-webcam").strip().lower()

if SOURCE_MODE == "pc-camera":
    CAMERA_SOURCES = {"PC Webcam": 0}
elif SOURCE_MODE == "video-file":
    VIDEO_PATH = os.environ.get("SURAKSHAN_VIDEO_PATH", "").strip()
    if not VIDEO_PATH:
        raise RuntimeError("SURAKSHAN_VIDEO_PATH is required for video-file input.")
    CAMERA_SOURCES = {"Video File": normalize_camera_source(VIDEO_PATH)}
elif SOURCE_MODE == "ip-webcam":
    CAMERA_SOURCES = {
        "IP Webcam": normalize_camera_source(
            os.environ.get("SURAKSHAN_CAMERA_URL", "http://10.63.101.159:8080/video")
        ),
    }
else:
    raise RuntimeError(
        "Unsupported camera source. Choose pc-camera, ip-webcam, or video-file."
    )


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
                raise RuntimeError(
                    f"Camera source {source!r} opened but no video frames arrived. "
                    "Check that the selected webcam, video file, or IP Webcam stream is accessible."
                )
            recorders[cam_id] = Recorder()
            gates[cam_id] = IncidentGate()

        print("[SURAKSHAN] Live detection started. Press ESC to stop.", flush=True)
        while True:
            for cam_id, stream in streams.items():
                ret, frame = stream.read()
                if not ret or frame is None:
                    continue

                details = predict_frame_details(frame, stream_id=cam_id)
                label = details["label"]
                color = (0, 0, 255) if label == "ACCIDENT" else (0, 255, 0)
                cv2.putText(
                    frame, f"{cam_id}: {label} ({details['confidence']:.0%})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                )

                recorder = recorders[cam_id]
                recorder.add_frame(frame)

                if details["is_new_prediction"] and gates[cam_id].update(label):
                    incident_id = str(uuid.uuid4())
                    print(f"[INCIDENT] Potential accident {incident_id} on {cam_id}", flush=True)
                    logger.log(
                        cam_id,
                        incident_id=incident_id,
                        timestamp=details.get("incident_timestamp"),
                    )
                    # Save the annotated trigger frame and attach it to the remote alert.
                    evidence_path = recorder.save_snapshot(frame, cam_id, incident_id)
                    recorder.start_incident(cam_id)
                    send_alert(
                        cam_id,
                        incident_id=incident_id,
                        confidence=details["accident_probability"],
                        image_path=evidence_path,
                        incident_timestamp=details.get("incident_timestamp"),
                    )

                cv2.imshow(cam_id, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        for stream in streams.values():
            stream.stop()
        for cam_id in CAMERA_SOURCES:
            reset_predictor(cam_id)
        for recorder in recorders.values():
            recorder.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
