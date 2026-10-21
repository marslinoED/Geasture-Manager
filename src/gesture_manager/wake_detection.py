"""Opt-in, motion-only wake detector using a low-resolution camera stream."""

from __future__ import annotations

from threading import Event, Thread, current_thread
from time import monotonic
from typing import Callable

import cv2
import numpy as np

from gesture_manager.camera_manager import CameraConfig, CameraManager, CameraState


class WakeMotionWorker:
    """Watch downsampled frame differences; never runs the hand model/classifier."""

    def __init__(self, camera: CameraManager, camera_index: int,
                 on_motion: Callable[[], None], *, threshold: float = 12.0) -> None:
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Wake motion threshold must be finite and positive")
        self.camera = camera
        self.camera_index = camera_index
        self.on_motion = on_motion
        self.threshold = threshold
        self._stop = Event()
        self._thread: Thread | None = None
        self.error: str | None = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.is_alive:
            return False
        self._stop.clear()
        self._thread = Thread(target=self._run, name="GestureManagerWakeMotion", daemon=False)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        self.camera.stop()

    def wait(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None or thread is current_thread():
            return thread is None
        thread.join(timeout)
        return not thread.is_alive()

    @staticmethod
    def frame_signature(frame: object) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3 or not array.size:
            raise ValueError("Wake detector requires a non-empty BGR frame")
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _run(self) -> None:
        if not self.camera.start(CameraConfig(self.camera_index, width=320, height=240, fps=2.5)):
            self.error = "Camera is busy; wake detection could not start."
            return
        previous: np.ndarray | None = None
        sequence = -1
        deadline = monotonic() + 5.0
        try:
            while not self._stop.is_set():
                status = self.camera.status
                if status.state is CameraState.ERROR:
                    self.error = status.error
                    return
                if monotonic() > deadline and status.state is not CameraState.RUNNING:
                    self.error = "Camera did not start for wake detection."
                    return
                newest = self.camera.get_latest_frame()
                if newest is None or newest[0] == sequence:
                    self._stop.wait(0.05)
                    continue
                sequence, frame = newest
                signature = self.frame_signature(frame)
                if previous is not None and float(np.mean(np.abs(signature - previous))) >= self.threshold:
                    self._stop.set()
                    self.on_motion()
                    return
                previous = signature
        except Exception as exc:
            self.error = f"Wake detection failed: {type(exc).__name__}: {exc}"
        finally:
            self.camera.stop()
            self.camera.wait_for_stop(5.0)
