import cv2
import os
from collections import deque

class Recorder:
    """
    Saves video clips containing:
    - 10 seconds BEFORE accident
    - 10 seconds AFTER accident

    Uses a deque buffer to store previous frames.
    """

    def __init__(self, fps=30, before=10, after=10):
        self.fps = fps
        self.before = before
        self.after = after
        self.buffer = deque(maxlen=fps * before)
        self.recording = False

        # Ensure 'clips' folder exists
        if not os.path.exists("clips"):
            os.makedirs("clips")

    def add_frame(self, frame):
        """Stores the recent frames BEFORE an accident."""
        self.buffer.append(frame)

    def save_clip(self, frame, cam_id):
        """
        Saves a full accident video clip:
        BEFORE + AFTER frames.
        """
        if self.recording:
            return None  # Prevent double saving

        self.recording = True

        output_path = os.path.join("clips", f"accident_{cam_id}.mp4")
        height, width, _ = frame.shape

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height)
        )

        # Save BEFORE accident frames
        for f in list(self.buffer):
            writer.write(f)

        # Save AFTER accident frames
        for _ in range(self.fps * self.after):
            writer.write(frame)

        writer.release()
        self.recording = False

        print(f"[RECORDER] Accident clip saved → {output_path}")
        return output_path
