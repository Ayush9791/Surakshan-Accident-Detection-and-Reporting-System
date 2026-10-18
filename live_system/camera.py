"""Threaded OpenCV camera reader with safe frame snapshots."""
import threading
import time

import cv2


class CameraStream:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self.ret, self.frame = self.cap.read()
        self._thread = threading.Thread(target=self._update, daemon=True, name="surakshan-camera")
        self._thread.start()

    def _update(self):
        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.ret = True
                    self.frame = frame
            else:
                with self._lock:
                    self.ret = False
                # Avoid a hot spin when a camera disconnects or a file ends.
                self._stop_event.wait(0.02)

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
