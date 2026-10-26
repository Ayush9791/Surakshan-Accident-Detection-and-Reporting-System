"""Threaded OpenCV camera reader with safe frame snapshots."""
import threading
import time

import cv2


def normalize_camera_source(value):
    """Convert numeric camera IDs to integers; preserve file and stream URLs."""
    value = str(value).strip()
    if not value:
        raise ValueError("Camera source cannot be empty")
    return int(value) if value.isdecimal() else value


class CameraStream:
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                f"Could not open camera source {source!r}. For Android IP Webcam, "
                "use the reachable HTTP stream URL ending in /video and ensure "
                "the phone and API host can reach each other on the network."
            )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.ret, self.frame = self.cap.read()
        self._thread = threading.Thread(
            target=self._update, daemon=True, name="surakshan-camera"
        )
        self._thread.start()

    def _update(self):
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            with self._lock:
                if ret and frame is not None:
                    self.ret = True
                    self.frame = frame
                else:
                    self.ret = False
            if not ret or frame is None:
                self._stop_event.wait(0.05)

    def wait_until_ready(self, timeout=8.0):
        """Wait briefly for the first decoded frame after the stream opens."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self._stop_event.is_set():
            ready, _ = self.read()
            if ready:
                return True
            self._stop_event.wait(0.05)
        return False

    def read(self):
        with self._lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self._stop_event.set()
        self.cap.release()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
