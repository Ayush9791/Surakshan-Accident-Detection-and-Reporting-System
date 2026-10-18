import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from live_system.recorder import Recorder


class RecorderTests(unittest.TestCase):
    def test_clip_contains_buffered_frames_and_real_post_event_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MagicMock()
            writer.isOpened.return_value = True
            with patch("live_system.recorder.cv2.VideoWriter", return_value=writer):
                recorder = Recorder(fps=1, before=2, after=2, output_dir=directory)
                recorder.add_frame(np.full((8, 8, 3), 1, dtype=np.uint8))
                recorder.add_frame(np.full((8, 8, 3), 2, dtype=np.uint8))
                self.assertTrue(recorder.start_incident("Camera 1"))
                recorder.add_frame(np.full((8, 8, 3), 3, dtype=np.uint8))
                self.assertTrue(recorder.recording)
                recorder.add_frame(np.full((8, 8, 3), 4, dtype=np.uint8))
                recorder.close()

            self.assertEqual(writer.write.call_count, 4)
            values = [int(call.args[0][0, 0, 0]) for call in writer.write.call_args_list]
            self.assertEqual(values, [1, 2, 3, 4])
            writer.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
