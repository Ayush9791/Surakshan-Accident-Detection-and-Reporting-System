"""Advanced live monitoring entry point."""
import os
import cv2

from live_system.api import send_alert
from live_system.camera import CameraStream
from live_system.incident import IncidentGate
from live_system.logger import AccidentLogger
from live_system.predict import predict_frame_details
from live_system.recorder import Recorder

CAMERA_SOURCES = {
    "Camera 1": 0,
    # "Camera 2": "sample_video.mp4",
    # "Camera 3": os.environ["SURAKSHAN_RTSP_URL"],
}


def main():
    streams = {cam: CameraStream(src) for cam, src in CAMERA_SOURCES.items()}
    recorders = {cam: Recorder() for cam in CAMERA_SOURCES}
    gates = {cam: IncidentGate() for cam in CAMERA_SOURCES}
    logger = AccidentLogger()

    try:
        while True:
            for cam_id, stream in streams.items():
                ret, frame = stream.read()
                if not ret or frame is None:
                    continue

                details = predict_frame_details(frame)
                label = details["label"]
                color = (0, 0, 255) if label == "ACCIDENT" else (0, 255, 0)
                cv2.putText(
                    frame,
                    f"{cam_id}: {label} ({details['confidence']:.0%})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

                recorder = recorders[cam_id]
                # Append each actual frame. When an incident is active, the
                # recorder collects real subsequent frames for the post-window.
                recorder.add_frame(frame)

                if gates[cam_id].update(label):
                    print(f"[INCIDENT] Confirmed potential accident on {cam_id}")
                    logger.log(cam_id)
                    recorder.start_incident(cam_id)
                    send_alert(cam_id, incident_id=None)

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
