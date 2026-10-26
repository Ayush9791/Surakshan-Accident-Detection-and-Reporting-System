import unittest
from unittest.mock import MagicMock, patch

from live_system.camera import CameraStream, normalize_camera_source


class CameraStreamTests(unittest.TestCase):
    def test_camera_source_normalization(self):
        self.assertEqual(normalize_camera_source("0"), 0)
        self.assertEqual(normalize_camera_source("2"), 2)
        url = "http://192.168.1.2:8080/video"
        self.assertEqual(normalize_camera_source(url), url)
        with self.assertRaises(ValueError):
            normalize_camera_source("  ")

    def test_unopenable_source_raises_actionable_error_and_releases_capture(self):
        capture = MagicMock()
        capture.isOpened.return_value = False
        with patch("live_system.camera.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "IP Webcam"):
                CameraStream("http://192.168.1.2:8080/video")
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
