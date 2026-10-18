"""Basic single-camera live detection."""
import cv2
from live_system.predict import predict_frame_details


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera 0")

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
