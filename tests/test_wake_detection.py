import unittest

import numpy as np

from gesture_manager.wake_detection import WakeMotionWorker


class WakeDetectionTests(unittest.TestCase):
    def test_motion_signature_is_small_and_changes_for_large_motion(self) -> None:
        still = np.zeros((240, 320, 3), dtype=np.uint8)
        moved = np.full((240, 320, 3), 255, dtype=np.uint8)
        still_sig = WakeMotionWorker.frame_signature(still)
        moved_sig = WakeMotionWorker.frame_signature(moved)
        self.assertEqual(still_sig.shape, (48, 64))
        self.assertGreater(float(np.mean(np.abs(moved_sig - still_sig))), 12)

    def test_invalid_frame_and_threshold_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            WakeMotionWorker.frame_signature(np.zeros((20, 20), dtype=np.uint8))
        with self.assertRaises(ValueError):
            WakeMotionWorker(None, 0, lambda: None, threshold=float("nan"))


if __name__ == "__main__":
    unittest.main()
