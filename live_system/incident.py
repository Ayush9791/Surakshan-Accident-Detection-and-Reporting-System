"""Small temporal gate that turns frame predictions into incident events."""
from collections import deque
import time


class IncidentGate:
    """Confirm incidents over multiple frames and avoid duplicate notifications.

    The model is a frame classifier, not a temporal action model. This gate is
    a practical debounce layer, not proof that a real collision occurred.
    """

    def __init__(
        self,
        window_size=10,
        min_positive=6,
        clear_frames=30,
        cooldown_seconds=30.0,
        clock=time.monotonic,
    ):
        if window_size < 1 or not 1 <= min_positive <= window_size:
            raise ValueError("Require 1 <= min_positive <= window_size")
        if clear_frames < 1 or cooldown_seconds < 0:
            raise ValueError("clear_frames must be positive and cooldown cannot be negative")
        self.window = deque(maxlen=window_size)
        self.min_positive = min_positive
        self.clear_frames = clear_frames
        self.cooldown_seconds = cooldown_seconds
        self.clock = clock
        self.active = False
        self.normal_streak = 0
        self.last_triggered = float("-inf")

    def update(self, label):
        """Return True exactly when a new incident should be opened."""
        is_accident = str(label).upper() == "ACCIDENT"
        self.window.append(is_accident)

        if self.active:
            if is_accident:
                self.normal_streak = 0
            else:
                self.normal_streak += 1
                if self.normal_streak >= self.clear_frames:
                    self.active = False
                    self.normal_streak = 0
            return False

        if is_accident:
            self.normal_streak = 0
        if len(self.window) < self.window.maxlen:
            return False
        if sum(self.window) < self.min_positive:
            return False
        if self.clock() - self.last_triggered < self.cooldown_seconds:
            return False

        self.active = True
        self.last_triggered = self.clock()
        self.normal_streak = 0
        return True
