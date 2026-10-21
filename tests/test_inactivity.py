import unittest
from unittest.mock import Mock

import numpy as np

from gesture_manager.inactivity import FrameInactivityTracker
from gesture_manager.runtime import RuntimeState
from gesture_manager.ui import GestureManagerUI


class InactivityTests(unittest.TestCase):
    def test_still_frames_timeout_and_motion_restarts_idle_timer(self):
        tracker = FrameInactivityTracker(timeout_seconds=300, motion_threshold=12)
        still = np.zeros((240, 320, 3), dtype=np.uint8)
        moved = np.full_like(still, 255)
        self.assertFalse(tracker.update(still, now=10))
        self.assertFalse(tracker.update(still, now=309.9))
        self.assertTrue(tracker.update(still, now=310))
        self.assertFalse(tracker.update(moved, now=311))
        self.assertFalse(tracker.update(moved, now=610.9))
        self.assertTrue(tracker.update(moved, now=611))

    def test_tracker_rejects_bad_frames_and_configuration(self):
        with self.assertRaises(ValueError):
            FrameInactivityTracker(timeout_seconds=0)
        with self.assertRaises(ValueError):
            FrameInactivityTracker(motion_threshold=float("nan"))
        tracker = FrameInactivityTracker()
        with self.assertRaises(ValueError):
            tracker.update(np.zeros((12, 12), dtype=np.uint8), now=1)

    def test_idle_timeout_strictly_pauses_the_active_runtime(self):
        ui = GestureManagerUI.__new__(GestureManagerUI)
        ui.controller = Mock(state=RuntimeState.ACTIVE)
        ui.controller.pause.return_value = True
        ui._live_test_active = True
        ui._recognition_paused = False
        ui.camera_manager = Mock()
        ui.camera_manager.get_latest_frame.return_value = (
            10, np.zeros((240, 320, 3), dtype=np.uint8))
        ui._inactivity_last_frame_sequence = None
        ui._inactivity_tracker = Mock()
        ui._inactivity_tracker.update.return_value = True
        ui._last_runtime_action = ""
        self.assertTrue(ui._check_runtime_inactivity())
        ui.controller.pause.assert_called_once_with()
        self.assertIn("5 minutes", ui._last_runtime_action)

    def test_idle_monitor_does_not_pause_an_already_paused_runtime(self):
        ui = GestureManagerUI.__new__(GestureManagerUI)
        ui.controller = Mock(state=RuntimeState.PAUSED)
        ui._live_test_active = True
        ui._recognition_paused = True
        ui.camera_manager = Mock()
        self.assertFalse(ui._check_runtime_inactivity())
        ui.camera_manager.get_latest_frame.assert_not_called()


if __name__ == "__main__":
    unittest.main()
