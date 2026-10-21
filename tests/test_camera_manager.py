from __future__ import annotations

import time
import unittest

import numpy as np

from gesture_manager.camera_manager import CameraConfig, CameraManager, CameraState
from gesture_manager.camera_preview import draw_hand_landmarks, frame_to_pil
from gesture_manager.hand_landmarks import DetectedHand, NormalizedLandmark


class FakeCapture:
    def __init__(self, opened: bool = True, fail_reads: bool = False) -> None:
        self.opened = opened
        self.fail_reads = fail_reads
        self.released = 0
        self.properties: list[tuple[int, float]] = []
        self.reads = 0

    def isOpened(self) -> bool:
        return self.opened

    def set(self, prop_id: int, value: float) -> bool:
        self.properties.append((prop_id, value))
        return True

    def read(self) -> tuple[bool, object]:
        self.reads += 1
        if self.fail_reads:
            return False, None
        return True, np.full((3, 4, 3), self.reads % 255, dtype=np.uint8)

    def release(self) -> None:
        self.released += 1


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class CameraManagerTests(unittest.TestCase):
    def test_capture_opens_once_keeps_latest_copy_and_releases(self) -> None:
        capture = FakeCapture()
        calls: list[int] = []

        def factory(index: int) -> FakeCapture:
            calls.append(index)
            return capture

        manager = CameraManager(factory)
        self.assertTrue(manager.start(CameraConfig(2, width=320, height=240, fps=30)))
        self.assertTrue(wait_until(lambda: manager.status.frames_captured >= 2))
        self.assertIsNotNone(manager.status.startup_ms)
        self.assertGreaterEqual(manager.status.startup_ms, 0)
        sequence, frame = manager.get_latest_frame()
        self.assertEqual(sequence, manager.status.frames_captured)
        frame[:] = 0
        self.assertNotEqual(int(manager.get_latest_frame()[1][0, 0, 0]), 0)
        self.assertEqual(calls, [2])
        self.assertTrue(any(value == 320 for _, value in capture.properties))
        self.assertTrue(manager.stop())
        self.assertTrue(manager.wait_for_stop(2))
        self.assertEqual(capture.released, 1)
        self.assertEqual(manager.status.state, CameraState.STOPPED)
        self.assertIsNone(manager.get_latest_frame())

    def test_unavailable_camera_reports_error_and_releases(self) -> None:
        capture = FakeCapture(opened=False)
        manager = CameraManager(lambda _index: capture)
        self.assertTrue(manager.start(CameraConfig(0)))
        self.assertTrue(manager.wait_for_stop(2))
        self.assertEqual(manager.status.state, CameraState.ERROR)
        self.assertIn("already in use", manager.status.error)
        self.assertEqual(capture.released, 1)

    def test_repeated_failed_reads_report_disconnection(self) -> None:
        capture = FakeCapture(fail_reads=True)
        manager = CameraManager(lambda _index: capture, read_failure_limit=2)
        manager.start(CameraConfig(0))
        self.assertTrue(manager.wait_for_stop(2))
        self.assertEqual(manager.status.state, CameraState.ERROR)
        self.assertIn("another app", manager.status.error)
        self.assertEqual(capture.released, 1)

    def test_scan_releases_every_probe(self) -> None:
        captures: list[FakeCapture] = []

        def factory(index: int) -> FakeCapture:
            capture = FakeCapture(opened=index == 1)
            captures.append(capture)
            return capture

        devices = CameraManager.enumerate_cameras(2, factory)
        self.assertEqual([device.index for device in devices], [1])
        self.assertEqual([capture.released for capture in captures], [1, 1, 1])

    def test_scan_excludes_camera_that_opens_but_cannot_read(self) -> None:
        capture = FakeCapture(fail_reads=True)
        devices = CameraManager.enumerate_cameras(
            max_index=0, capture_factory=lambda _index: capture
        )
        self.assertEqual(devices, ())
        self.assertEqual(capture.released, 1)

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            CameraConfig(-1)
        with self.assertRaises(ValueError):
            CameraConfig(0, width=0)


class CameraPreviewConversionTests(unittest.TestCase):
    def test_bgr_conversion_and_bounded_size(self) -> None:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[0, 0] = [0, 0, 255]  # OpenCV blue/green/red order => red pixel.
        image = frame_to_pil(frame, (10, 10))
        self.assertLessEqual(image.width, 10)
        self.assertLessEqual(image.height, 10)
        self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))

    def test_rejects_invalid_frame(self) -> None:
        with self.assertRaises(ValueError):
            frame_to_pil(object())

    def test_landmark_overlay_draws_skeleton_without_mutating_camera_frame(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        points = tuple(NormalizedLandmark(0.5, 0.5, 0.0) for _ in range(21))
        overlaid = draw_hand_landmarks(frame, (DetectedHand(points, 0.9),))
        self.assertFalse(np.array_equal(frame, overlaid))
        self.assertTrue(np.all(frame == 0))


if __name__ == "__main__":
    unittest.main()
