"""Rolling-buffer incident recorder with actual post-event frames."""
from collections import deque
from datetime import datetime
from pathlib import Path
import re
import threading

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Recorder:
    def __init__(self, fps=30, before=10, after=10, output_dir="clips"):
        if fps <= 0 or before < 0 or after < 0:
            raise ValueError("fps must be positive and before/after cannot be negative")
        self.fps = fps
        self.before = before
        self.after = after
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = PROJECT_ROOT / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer = deque(maxlen=max(1, fps * before))
        self.recording = False
        self._frames = []
        self._target_frame_count = 0
        self._cam_id = None
        self._lock = threading.Lock()
        self._writer_threads = []

    def add_frame(self, frame):
        """Store a frame and append real subsequent frames during an incident."""
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return
        frame_copy = frame.copy()
        frames_to_write = None
        cam_id = None
        with self._lock:
            if self.recording:
                self._frames.append(frame_copy)
                if len(self._frames) >= self._target_frame_count:
                    frames_to_write = self._frames
                    cam_id = self._cam_id
                    self._frames = []
                    self._target_frame_count = 0
                    self._cam_id = None
                    self.recording = False
            self.buffer.append(frame_copy)
        if frames_to_write is not None:
            self._write_async(frames_to_write, cam_id)

    def save_snapshot(self, frame, cam_id, incident_id):
        """Save an annotated JPEG for manual incident review."""
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return None
        evidence_dir = PROJECT_ROOT / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(incident_id)).strip("._") or "incident"
        safe_cam = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cam_id)).strip("._") or "camera"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = evidence_dir / f"accident_{safe_cam}_{safe_id}_{timestamp}.jpg"
        try:
            if cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]):
                print(f"[RECORDER] Saved incident evidence photo: {output_path}", flush=True)
                return str(output_path)
        except Exception as exc:
            print(f"[RECORDER] Could not save evidence photo: {exc}", flush=True)
        return None

    def start_incident(self, cam_id):
        """Freeze pre-event frames and start collecting subsequent camera frames."""
        with self._lock:
            if self.recording:
                return False
            self._cam_id = str(cam_id)
            self._frames = list(self.buffer)
            self._target_frame_count = len(self._frames) + self.fps * self.after
            if self.after == 0:
                frames, camera = self._frames, self._cam_id
                self._frames = []
                self._cam_id = None
                self._target_frame_count = 0
                self._write_async(frames, camera)
                return True
            self.recording = True
        return True

    def _write_async(self, frames, cam_id):
        thread = threading.Thread(
            target=self._write_clip, args=(frames, cam_id), daemon=True,
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
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
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
        """Write a partial clip if monitoring stops during an incident."""
        with self._lock:
            pending = self._frames if self.recording else None
            cam_id = self._cam_id
            self._frames = []
            self._cam_id = None
            self._target_frame_count = 0
            self.recording = False
        if pending:
            self._write_async(pending, cam_id)
        for thread in list(self._writer_threads):
            thread.join(timeout=2)
