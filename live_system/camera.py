import cv2
import threading


class CameraStream:
    """
    Threaded camera reader for smooth, high-FPS video capture.
    Works with:
    - Webcam
    - CCTV (RTSP)
    - IP cameras
    - Video files
    """

    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.ret, self.frame = self.cap.read()
        self.running = True

        # Start background thread
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()

    def update(self):
        """Continuously read frames in a separate thread."""
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        """Return the latest frame."""
        return self.ret, self.frame

    def stop(self):
        """Stop the camera stream safely."""
        self.running = False
        self.cap.release()
