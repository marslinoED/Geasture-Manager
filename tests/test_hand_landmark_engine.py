import threading
import time
import unittest
from pathlib import Path

import numpy as np

from gesture_manager.camera_manager import CameraConfig, CameraState, CameraStatus
from gesture_manager.hand_landmark_engine import EngineConfig, EngineState, HandLandmarkEngine


class FakeCamera:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status = CameraStatus()
        self._frame = None
        self._seq = 0
        self.running = threading.Event()
        self.stopped = threading.Event()

    @property
    def status(self):
        with self._lock:
            return self._status

    @property
    def is_worker_alive(self):
        return self.status.state in (CameraState.STARTING, CameraState.RUNNING, CameraState.STOPPING)

    def start(self, config: CameraConfig) -> bool:
        with self._lock:
            self._status = CameraStatus(state=CameraState.RUNNING, index=config.index,
                                        requested_resolution=(config.width, config.height))
            self._frame = np.zeros((16, 16, 3), dtype=np.uint8)
            self._seq += 1
            self._status = CameraStatus(state=CameraState.RUNNING, index=config.index,
                                        requested_resolution=(config.width, config.height), frames_captured=self._seq)
        self.running.set()
        self.stopped.clear()
        return True

    def get_latest_frame(self):
        with self._lock:
            return (self._seq, self._frame.copy()) if self._frame is not None else None

    def stop(self) -> bool:
        with self._lock:
            self._frame = None
            self._status = CameraStatus()
        self.stopped.set()
        return True

    def wait_for_stop(self, timeout=None) -> bool:
        return True


class DelayedReleaseCamera(FakeCamera):
    """Model a camera backend that takes time to finish releasing the device."""

    def __init__(self) -> None:
        super().__init__()
        self.stop_requested = threading.Event()
        self._release_thread = None

    @property
    def is_worker_alive(self):
        return self.status.state in (CameraState.STARTING, CameraState.RUNNING, CameraState.STOPPING)

    def start(self, config: CameraConfig) -> bool:
        self.stop_requested.clear()
        return super().start(config)

    def stop(self) -> bool:
        with self._lock:
            if self._status.state is CameraState.STOPPING:
                return True
            if self._status.state is CameraState.STOPPED:
                return False
            self._status = CameraStatus(state=CameraState.STOPPING, index=self._status.index)
        self.stop_requested.set()

        def release_later() -> None:
            time.sleep(0.12)
            with self._lock:
                self._frame = None
                self._status = CameraStatus()
            self.stopped.set()

        self._release_thread = threading.Thread(target=release_later, daemon=True)
        self._release_thread.start()
        return True

    def wait_for_stop(self, timeout=None) -> bool:
        return self.stopped.wait(timeout)


class FakeModel:
    def __init__(self, fail_load: bool = False) -> None:
        self.fail_load = fail_load
        self.loads = 0
        self.detects = 0
        self.closed = False

    def load(self) -> None:
        self.loads += 1
        if self.fail_load:
            raise FileNotFoundError("model missing")

    def detect(self, frame):
        self.detects += 1
        return ()

    def close(self) -> None:
        self.closed = True


def wait_until(predicate, timeout=2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class HandLandmarkEngineTests(unittest.TestCase):
    def test_start_pause_resume_stop_loads_model_once(self) -> None:
        camera = FakeCamera()
        model = FakeModel()
        engine = HandLandmarkEngine(camera, model_factory=lambda _p, _t: model)
        config = EngineConfig(CameraConfig(0), processing_rate_hz=30, model_dir=Path("unused"))
        self.assertTrue(engine.start(config))
        self.assertTrue(wait_until(lambda: engine.status.processed_frames >= 1))
        self.assertEqual(engine.status.hands_detected, False)
        self.assertIsNotNone(engine.status.inference_latency_ms)
        self.assertTrue(engine.pause())
        self.assertEqual(engine.status.state, EngineState.PAUSED)
        self.assertTrue(wait_until(lambda: camera.status.state is CameraState.STOPPED))
        paused_count = model.detects
        time.sleep(0.06)
        self.assertEqual(model.detects, paused_count)
        self.assertTrue(engine.resume())
        self.assertTrue(wait_until(lambda: engine.status.processed_frames > paused_count))
        self.assertTrue(engine.stop())
        self.assertTrue(engine.wait_for_stop(2))
        self.assertEqual(engine.status.state, EngineState.STOPPED)
        self.assertEqual(model.loads, 1)
        self.assertTrue(model.closed)

    def test_resume_during_camera_release_reopens_camera_without_engine_error(self) -> None:
        camera = DelayedReleaseCamera()
        model = FakeModel()
        engine = HandLandmarkEngine(camera, model_factory=lambda _p, _t: model)
        config = EngineConfig(CameraConfig(0), processing_rate_hz=30, model_dir=Path("unused"))
        self.assertTrue(engine.start(config))
        self.assertTrue(wait_until(lambda: engine.status.processed_frames >= 1))
        previous = engine.status.processed_frames
        self.assertTrue(engine.pause())
        self.assertTrue(camera.stop_requested.wait(1))
        self.assertTrue(engine.resume())
        self.assertTrue(wait_until(lambda: engine.status.processed_frames > previous))
        self.assertNotEqual(engine.status.state, EngineState.ERROR)
        self.assertTrue(engine.stop())
        self.assertTrue(engine.wait_for_stop(2))

    def test_load_error_is_reported_and_worker_closes_cleanly(self) -> None:
        camera = FakeCamera()
        model = FakeModel(fail_load=True)
        engine = HandLandmarkEngine(camera, model_factory=lambda _p, _t: model)
        self.assertTrue(engine.start(EngineConfig(CameraConfig(0), model_dir=Path("unused"))))
        self.assertTrue(wait_until(lambda: engine.status.state is EngineState.ERROR))
        self.assertIn("model missing", engine.status.error)
        self.assertTrue(engine.wait_for_stop(2))
        self.assertTrue(model.closed)

    def test_processing_rate_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            EngineConfig(CameraConfig(0), processing_rate_hz=0.5)


if __name__ == "__main__":
    unittest.main()
