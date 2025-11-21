import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from live_system.camera import CameraStream
from live_system.predict import predict_frame
from live_system.recorder import Recorder
from live_system.logger import AccidentLogger
from live_system.api import send_alert  # optional

print("🔥 Surakshan Advanced Live System Starting...")

# ----------------------------------------------------
# MULTI CAMERA SOURCES
# YOU CAN ADD WEBCAM, RTSP, VIDEOS, ANYTHING
# ----------------------------------------------------
CAMERA_SOURCES = {
    "Camera 1": 0,                      # webcam
    # "Camera 2": "test_video.mp4",     # video test
    # "Camera 3": "rtsp://user:pass@IP:Port/stream",
}

# ----------------------------------------------------
# INIT CAMERAS, RECORDERS, LOGGER
# ----------------------------------------------------
streams = {cam: CameraStream(src) for cam, src in CAMERA_SOURCES.items()}
recorders = {cam: Recorder() for cam in CAMERA_SOURCES}
logger = AccidentLogger()

# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
while True:

    for cam_id, stream in streams.items():

        ret, frame = stream.read()
        if not ret:
            print(f"[WARN] No frame from {cam_id}")
            continue

        # Predict accident
        result = predict_frame(frame)

        # Color for overlay
        color = (0, 255, 0) if result == "NORMAL" else (0, 0, 255)

        # Draw label
        cv2.putText(frame, f"{cam_id}: {result}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 3)

        # ------------------------------
        # ACCIDENT DETECTED
        # ------------------------------
        if result == "ACCIDENT":
            print(f"🚨 Accident detected on {cam_id}!")

            # Log the event
            logger.log(cam_id)

            # Save accident clip (before & after)
            recorders[cam_id].save_clip(frame, cam_id)

            # Send optional alert
            send_alert(cam_id)

        # Store current frame for pre-accident buffer
        recorders[cam_id].add_frame(frame)

        # Show live feed
        cv2.imshow(cam_id, frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ----------------------------------------------------
# CLEANUP
# ----------------------------------------------------
for stream in streams.values():
    stream.stop()

cv2.destroyAllWindows()
print("System stopped.")
