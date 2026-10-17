import unittest

from live_system.incident import IncidentGate


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class IncidentGateTests(unittest.TestCase):
    def test_requires_multiple_positive_frames(self):
        gate = IncidentGate(window_size=3, min_positive=2, clear_frames=2,
                            cooldown_seconds=0, clock=FakeClock())
        self.assertFalse(gate.update("ACCIDENT"))
        self.assertFalse(gate.update("NORMAL"))
        self.assertTrue(gate.update("ACCIDENT"))

    def test_does_not_trigger_again_while_active(self):
        gate = IncidentGate(window_size=2, min_positive=2, clear_frames=2,
                            cooldown_seconds=0, clock=FakeClock())
        self.assertFalse(gate.update("ACCIDENT"))
        self.assertTrue(gate.update("ACCIDENT"))
        for _ in range(5):
            self.assertFalse(gate.update("ACCIDENT"))
        self.assertTrue(gate.active)

    def test_rearms_after_clear_and_cooldown(self):
        clock = FakeClock()
        gate = IncidentGate(window_size=2, min_positive=2, clear_frames=2,
                            cooldown_seconds=5, clock=clock)
        gate.update("ACCIDENT")
        self.assertTrue(gate.update("ACCIDENT"))
        gate.update("NORMAL")
        gate.update("NORMAL")
        self.assertFalse(gate.active)
        clock.advance(6)
        self.assertFalse(gate.update("ACCIDENT"))
        self.assertTrue(gate.update("ACCIDENT"))

    def test_unknown_label_is_not_accident(self):
        gate = IncidentGate(window_size=2, min_positive=2, clear_frames=1,
                            cooldown_seconds=0)
        self.assertFalse(gate.update("UNKNOWN"))
        self.assertFalse(gate.update("NORMAL"))


if __name__ == "__main__":
    unittest.main()
