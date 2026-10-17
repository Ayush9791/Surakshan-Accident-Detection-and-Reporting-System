"""Rolling-buffer incident recorder with real post-event frames."""
from collections import deque
from datetime import datetime
from pathlib import Path
import re
import threading

import cv2


class Recorder:
    def __init__(self, fps=30, before=10, after=10, output_dir="clips"):
        if fps <= 0 or before < 0 or after < 0:
            raise ValueError("fps must be positive and before/after cannot be negative")
        self.fps = fps
        self.before = before
        self.after = after
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = deque(maxlen=max(1, fps * before))
        self.recording = False
        self._frames = []
        self._cam_id = None
        self._lock = threading.Lock()
        self._writer_threads = []

    def add_frame(self, frame):
        """Store a frame and append it to an active incident's post-event clip."""
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return
        frame_copy = frame.copy()
        frames_to_write = None
        cam_id = None

        with self._lock:
            self.buffer.append(frame_copy)
            if self.recording:
                self._frames.append(frame_copy)
                # The trigger frame is already present in the rolling buffer.
                # This count therefore represents actual frames after the trigger.
                post_count = max(1, self.fps * self.after)
                if len(self._frames) >= self._pre_event_count + post_count:
                    frames_to_write = self._frames
                    cam_id = self._cam_id
                    self._frames = []
                    self._cam_id = None
                    self.recording = False

        if frames_to_write is not None:
            self._write_async(frames_to_write, cam_id)

    @property
    def _pre_event_count(self):
        return self.fps * self.before

    def start_incident(self, cam_id):
        """Freeze the pre-event buffer and collect subsequent frames."""
        with self._lock:
            if self.recording:
                return None
            self.recording = True
            self._cam_id = str(cam_id)
            self._frames = list(self.buffer)
        return True

    def _write_async(self, frames, cam_id):
        thread = threading.Thread(
            target=self._write_clip,
            args=(frames, cam_id),
            daemon=True,
            name=f"clip-writer-{cam_id}",
        )
        self._writer_threads.append(thread)
        thread.start()

    def _write_clip(self, frames, cam_id):
        if not frames:
            return None
        height, width = frames[0].shape[:2]
        safe_cam = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cam_id)).strip("._") or "camera"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = self.output_dir / f"accident_{safe_cam}_{timestamp}.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        try:
            if not writer.isOpened():
                print(f"[RECORDER] Could not open video writer for {output_path}")
                return None
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()
        print(f"[RECORDER] Saved incident clip: {output_path}")
        return str(output_path)

    def close(self):
        """Write any partial active clip and wait briefly for writers to finish."""
        with self._lock:
            pending = self._frames if self.recording else None
            cam_id = self._cam_id
            self._frames = []
            self._cam_id = None
            self.recording = False
        if pending:
            self._write_async(pending, cam_id)
        for thread in list(self._writer_threads):
            thread.join(timeout=2)
