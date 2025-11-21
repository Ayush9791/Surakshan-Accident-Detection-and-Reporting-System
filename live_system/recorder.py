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

