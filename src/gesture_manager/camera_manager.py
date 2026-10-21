"""Thread-owned OpenCV camera capture with a latest-frame buffer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import logging
from contextlib import contextmanager
from threading import Event, Lock, RLock, Thread, current_thread
from time import monotonic
from typing import Callable, Protocol

import cv2

logger = logging.getLogger(__name__)
_opencv_log_lock = RLock()


@contextmanager
def _quiet_opencv_logs():
    """Temporarily silence native OpenCV diagnostics from index probes."""
    logging_api = getattr(getattr(cv2, "utils", None), "logging", None)
    if logging_api is None or not all(
        hasattr(logging_api, name) for name in ("getLogLevel", "setLogLevel", "LOG_LEVEL_SILENT")
    ):
        yield
        return
    with _opencv_log_lock:
        previous = logging_api.getLogLevel()
        logging_api.setLogLevel(logging_api.LOG_LEVEL_SILENT)
        try:
            yield
        finally:
            logging_api.setLogLevel(previous)


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...
    def set(self, prop_id: int, value: float) -> bool: ...
    def read(self) -> tuple[bool, object]: ...
    def release(self) -> None: ...


CaptureFactory = Callable[[int], VideoCaptureLike]


class CameraState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True)
class CameraConfig:
    index: int
    width: int = 640
    height: int = 480
    fps: float = 15.0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Camera index must be zero or greater")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Capture width and height must be positive")
        if not 1 <= self.fps <= 60:
            raise ValueError("Capture rate must be between 1 and 60 FPS")


@dataclass(frozen=True)
class CameraDevice:
    index: int
    name: str


@dataclass(frozen=True)
class CameraStatus:
    state: CameraState = CameraState.STOPPED
    index: int | None = None
    requested_resolution: tuple[int, int] | None = None
    frame_resolution: tuple[int, int] | None = None
    frames_captured: int = 0
    startup_ms: float | None = None
    error: str | None = None


class CameraManager:
    """Manage one camera and publish thread-safe status and newest-frame data.

    One worker owns the VideoCapture instance from open through release. The
    capture loop overwrites a single frame slot, so slow consumers never build
    a backlog. No frame is written to disk.
    """

    def __init__(
        self,
        capture_factory: CaptureFactory | None = None,
        *,
        read_failure_limit: int = 5,
    ) -> None:
        if read_failure_limit < 1:
            raise ValueError("read_failure_limit must be positive")
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._read_failure_limit = read_failure_limit
        self._lock = Lock()
        self._status = CameraStatus()
        self._latest_frame: object | None = None
        self._capture: VideoCaptureLike | None = None
        self._worker: Thread | None = None
        self._stop_event: Event | None = None

    @property
    def status(self) -> CameraStatus:
        with self._lock:
            return self._status

    @property
    def is_worker_alive(self) -> bool:
        with self._lock:
            worker = self._worker
        return worker is not None and worker.is_alive()

    def get_latest_frame(self) -> tuple[int, object] | None:
        """Return (sequence, copy) of the newest frame, or None before first read."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._status.frames_captured, self._latest_frame.copy()

    def start(self, config: CameraConfig) -> bool:
        """Start a worker and return immediately; opening occurs in that worker."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return False
            stop_event = Event()
            self._stop_event = stop_event
            self._latest_frame = None
            self._status = CameraStatus(
                state=CameraState.STARTING,
                index=config.index,
                requested_resolution=(config.width, config.height),
            )
            worker = Thread(
                target=self._capture_loop,
                args=(config, stop_event),
                name=f"GestureManagerCamera-{config.index}",
                daemon=False,
            )
            self._worker = worker
            try:
                worker.start()
            except Exception as exc:
                self._status = replace(
                    self._status,
                    state=CameraState.ERROR,
                    error=f"Could not start camera worker: {exc}",
                )
                logger.exception("Could not start camera worker")
                return False
        return True

    def stop(self) -> bool:
        """Request stop without blocking the caller; the worker releases capture."""
        with self._lock:
            worker = self._worker
            stop_event = self._stop_event
            if worker is None or not worker.is_alive():
                self._latest_frame = None
                self._status = replace(
                    self._status,
                    state=CameraState.STOPPED,
                    frame_resolution=None,
                    frames_captured=0,
                    error=None,
                )
                return False
            self._status = replace(self._status, state=CameraState.STOPPING)
            if stop_event is not None:
                stop_event.set()
            return True

    def wait_for_stop(self, timeout: float | None = None) -> bool:
        """Join the worker for up to timeout seconds; None waits until released."""
        with self._lock:
            worker = self._worker
        if worker is None:
            return True
        if worker is current_thread():
            return False
        worker.join(timeout)
        return not worker.is_alive()

    @classmethod
    def enumerate_cameras(
        cls,
        max_index: int = 5,
        capture_factory: CaptureFactory | None = None,
    ) -> tuple[CameraDevice, ...]:
        """Best-effort index scan; every opened probe is released immediately.

        OpenCV does not provide portable device names on all backends. This
        probes indices 0 through max_index and labels openable devices by index.
        Call it from a background worker because opening a probe may take time.
        """
        if max_index < 0:
            raise ValueError("max_index must be zero or greater")
        factory = capture_factory or cv2.VideoCapture
        devices: list[CameraDevice] = []
        with _quiet_opencv_logs():
            for index in range(max_index + 1):
                capture: VideoCaptureLike | None = None
                try:
                    capture = factory(index)
                    if capture.isOpened():
                        ok, frame = capture.read()
                        if ok and frame is not None:
                            devices.append(CameraDevice(index=index, name=f"Camera {index}"))
                except Exception:
                    logger.debug("Camera index %s could not be probed", index, exc_info=True)
                finally:
                    if capture is not None:
                        try:
                            capture.release()
                        except Exception:
                            logger.debug("Camera probe release failed", exc_info=True)
        return tuple(devices)

    def _capture_loop(self, config: CameraConfig, stop_event: Event) -> None:
        # Backends such as MSMF print native WARN/ERROR lines for expected
        # contention and disconnect cases. The manager reports those in status.
        with _quiet_opencv_logs():
            self._capture_loop_body(config, stop_event)

    def _capture_loop_body(self, config: CameraConfig, stop_event: Event) -> None:
        capture: VideoCaptureLike | None = None
        failure: str | None = None
        startup_started = monotonic()
        try:
            if stop_event.is_set():
                return
            capture = self._capture_factory(config.index)
            with self._lock:
                self._capture = capture

            if not capture.isOpened():
                failure = (
                    f"Could not open camera {config.index}; it may be missing, "
                    "unavailable, or already in use."
                )
                return

            self._set_capture_property(capture, cv2.CAP_PROP_FRAME_WIDTH, config.width)
            self._set_capture_property(capture, cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            self._set_capture_property(capture, cv2.CAP_PROP_FPS, config.fps)
            self._set_capture_property(capture, cv2.CAP_PROP_BUFFERSIZE, 1)
            self._set_status(state=CameraState.RUNNING, error=None,
                             startup_ms=max(0.0, (monotonic() - startup_started) * 1000.0))

            read_failures = 0
            frame_period = 1.0 / config.fps
            while not stop_event.is_set():
                read_started = monotonic()
                ok, frame = capture.read()
                if not ok or frame is None:
                    read_failures += 1
                    if read_failures >= self._read_failure_limit:
                        with self._lock:
                            had_frames = self._status.frames_captured > 0
                        failure = (
                            "Camera stopped returning frames after capture started; it may have been disconnected."
                            if had_frames
                            else (
                                f"Camera {config.index} opened but could not provide frames; "
                                "it may be in use by another app or blocked by camera privacy settings."
                            )
                        )
                        break
                    if stop_event.wait(min(0.05, frame_period)):
                        break
                    continue

                read_failures = 0
                height, width = frame.shape[:2]
                with self._lock:
                    if stop_event.is_set():
                        break
                    self._latest_frame = frame
                    self._status = replace(
                        self._status,
                        frame_resolution=(width, height),
                        frames_captured=self._status.frames_captured + 1,
                    )

                delay = frame_period - (monotonic() - read_started)
                if delay > 0 and stop_event.wait(delay):
                    break
        except Exception as exc:
            failure = f"Camera error: {type(exc).__name__}: {exc}"
            logger.exception("Camera capture failed")
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    logger.exception("Could not release camera %s", config.index)
            with self._lock:
                self._capture = None
                self._latest_frame = None if stop_event.is_set() or failure else self._latest_frame
                if stop_event.is_set():
                    self._status = replace(
                        self._status,
                        state=CameraState.STOPPED,
                        frame_resolution=None,
                        frames_captured=0,
                        error=None,
                    )
                elif failure:
                    self._status = replace(
                        self._status,
                        state=CameraState.ERROR,
                        error=failure,
                    )
                elif self._status.state is not CameraState.ERROR:
                    self._status = replace(self._status, state=CameraState.STOPPED)

    @staticmethod
    def _set_capture_property(
        capture: VideoCaptureLike, property_id: int, value: float
    ) -> None:
        try:
            capture.set(property_id, float(value))
        except Exception:
            # Some backends do not implement every optional capture property.
            logger.debug("Camera backend rejected capture property", exc_info=True)

    def _set_status(self, **changes: object) -> None:
        with self._lock:
            self._status = replace(self._status, **changes)

