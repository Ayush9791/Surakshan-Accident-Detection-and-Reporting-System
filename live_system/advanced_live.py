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
from live_system.predict import predict_frame_details
from live_system.recorder import Recorder


# Default to the local webcam. Configure IP Webcam explicitly through the
# environment so a stale private IP never breaks a fresh checkout.
CAMERA_SOURCES = {
    "Camera 1": normalize_camera_source(os.environ.get("SURAKSHAN_CAMERA_URL", "0")),
}


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
                    "For IP Webcam, use http://PHONE_IP:8080/video, keep the app's "
                    "server running, and confirm the URL works from this computer."
                )
            recorders[cam_id] = Recorder()
            gates[cam_id] = IncidentGate()

        print("[SURAKSHAN] Live detection started. Press ESC to stop.", flush=True)
        while True:
            for cam_id, stream in streams.items():
                ret, frame = stream.read()
                if not ret or frame is None:
                    continue

                details = predict_frame_details(frame)
                label = details["label"]
                color = (0, 0, 255) if label == "ACCIDENT" else (0, 255, 0)
                cv2.putText(
                    frame, f"{cam_id}: {label} ({details['confidence']:.0%})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                )

                recorder = recorders[cam_id]
                recorder.add_frame(frame)

                if gates[cam_id].update(label):
                    incident_id = str(uuid.uuid4())
                    print(f"[INCIDENT] Potential accident {incident_id} on {cam_id}", flush=True)
                    logger.log(cam_id, incident_id=incident_id)
                    recorder.start_incident(cam_id)
                    send_alert(
                        cam_id,
                        incident_id=incident_id,
                        confidence=details["accident_probability"],
                    )

                cv2.imshow(cam_id, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        for stream in streams.values():
            stream.stop()
        for recorder in recorders.values():
            recorder.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
