"""Low-cost camera-frame movement tracking for automatic idle pause."""

from __future__ import annotations

import math

import cv2
import numpy as np


class FrameInactivityTracker:
    """Report when sampled camera frames show no meaningful movement."""

    def __init__(self, *, timeout_seconds: float = 300.0,
                 motion_threshold: float = 12.0) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Inactivity timeout must be finite and positive")
        if not math.isfinite(motion_threshold) or motion_threshold <= 0:
            raise ValueError("Motion threshold must be finite and positive")
        self.timeout_seconds = float(timeout_seconds)
        self.motion_threshold = float(motion_threshold)
        self._previous: np.ndarray | None = None
        self._last_motion_at: float | None = None

    @staticmethod
    def signature(frame: object) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3 or not array.size:
            raise ValueError("Inactivity tracking requires a non-empty BGR frame")
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)

    def reset(self) -> None:
        self._previous = None
        self._last_motion_at = None

    def update(self, frame: object, *, now: float) -> bool:
        """Return true once the idle timeout has elapsed since detected motion."""
        if not math.isfinite(now) or now < 0:
            raise ValueError("Inactivity timestamp must be finite and non-negative")
        current = self.signature(frame)
        if self._previous is None:
            self._previous = current
            self._last_motion_at = now
            return False
        difference = float(np.mean(np.abs(current - self._previous)))
        self._previous = current
        if difference >= self.motion_threshold:
            self._last_motion_at = now
        assert self._last_motion_at is not None
        return now - self._last_motion_at >= self.timeout_seconds
