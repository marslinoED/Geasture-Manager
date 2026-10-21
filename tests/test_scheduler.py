import unittest

from gesture_manager.runtime import RuntimeState
from gesture_manager.scheduler import AdaptiveScheduler, WindowsFocusDetector


class FakeWindowsAPI:
    def __init__(self, foreground: int, roots: dict[int, int] | None = None) -> None:
        self.foreground = foreground
        self.roots = roots or {}
        self.calls: list[tuple[str, int]] = []

    def GetForegroundWindow(self) -> int:
        self.calls.append(("foreground", 0))
        return self.foreground

    def GetAncestor(self, hwnd: int, flags: int) -> int:
        self.calls.append(("ancestor", hwnd))
        return self.roots.get(hwnd, hwnd)


class SchedulerTests(unittest.TestCase):
    def test_runtime_modes_target_fps_and_camera_policy(self) -> None:
        self.assertEqual(AdaptiveScheduler.schedule(RuntimeState.ACTIVE, focused=True).target_fps, 15)
        unfocused = AdaptiveScheduler.schedule(RuntimeState.ACTIVE, focused=False)
        self.assertEqual(unfocused.target_fps, 10)
        self.assertTrue(unfocused.run_recognition)
        shortcut_paused = AdaptiveScheduler.schedule(
            RuntimeState.ACTIVE, focused=True, shortcut_paused=True)
        self.assertEqual(shortcut_paused.mode, "shortcuts_paused")
        self.assertEqual(shortcut_paused.target_fps, 2.5)
        self.assertTrue(shortcut_paused.run_recognition)
        strict = AdaptiveScheduler.schedule(RuntimeState.PAUSED, focused=True)
        self.assertEqual((strict.target_fps, strict.release_camera), (0, True))
        wake = AdaptiveScheduler.schedule(RuntimeState.PAUSED, focused=False,
                                          wake_detection=True, strict_pause=False)
        self.assertEqual(wake.target_fps, 2.5)
        self.assertFalse(wake.run_recognition)
        stopped = AdaptiveScheduler.schedule(RuntimeState.STOPPED, focused=True)
        self.assertEqual((stopped.target_fps, stopped.release_camera), (0, True))

    def test_focus_detector_uses_foreground_window_and_root_handle(self) -> None:
        api = FakeWindowsAPI(100, {100: 1, 200: 1, 300: 2})
        detector = WindowsFocusDetector(api)
        self.assertTrue(detector.is_focused(200))
        self.assertFalse(detector.is_focused(300))
        self.assertEqual(api.calls[0], ("foreground", 0))

    def test_focus_detector_reports_unavailable_when_not_on_windows(self) -> None:
        detector = WindowsFocusDetector(api=None)
        if __import__("os").name != "nt":
            self.assertIsNone(detector.is_focused(123))

