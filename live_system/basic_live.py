"""Basic single-camera live detection."""
import os

import cv2
from live_system.camera import normalize_camera_source
from live_system.predict import predict_frame_details


def main():
    source = normalize_camera_source(os.environ.get("SURAKSHAN_CAMERA_URL", "0"))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source {source!r}. For IP Webcam, use http://PHONE_IP:8080/video.")

    print("Starting Surakshan live detection. Press ESC to exit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab a camera frame.")
                break
            details = predict_frame_details(frame)
            label = details["label"]
            color = (0, 0, 255) if label == "ACCIDENT" else (0, 255, 0)
            cv2.putText(
                frame,
                f"{label} ({details['confidence']:.0%})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )
            cv2.imshow("Surakshan - Basic Live Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
