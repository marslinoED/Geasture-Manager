"""Background scheduler and lifecycle for local hand landmark inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from time import monotonic, perf_counter
from typing import Callable, Protocol

from gesture_manager.camera_manager import CameraConfig, CameraManager, CameraState
from gesture_manager.assets import model_assets_directory
from gesture_manager.hand_landmarks import DetectedHand, LiteHandLandmarkModel


class EngineState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass(frozen=True)
class EngineConfig:
    camera: CameraConfig
    processing_rate_hz: float = 8.0
    model_dir: Path = field(default_factory=model_assets_directory)
    num_threads: int = 2

    def __post_init__(self) -> None:
        if not 1 <= self.processing_rate_hz <= 30:
            raise ValueError("Processing rate must be between 1 and 30 Hz")
        if self.num_threads < 1:
            raise ValueError("num_threads must be positive")


@dataclass(frozen=True)
class EngineStatus:
    state: EngineState = EngineState.STOPPED
    model_loaded: bool = False
    hands_detected: bool | None = None
    hand_count: int = 0
    hands: tuple[DetectedHand, ...] = ()
    inference_latency_ms: float | None = None
    processed_frames: int = 0
    error: str | None = None


class ModelLike(Protocol):
    def load(self) -> None: ...
    def detect(self, frame: object) -> tuple[DetectedHand, ...]: ...
    def close(self) -> None: ...


ModelFactory = Callable[[Path, int], ModelLike]


class HandLandmarkEngine:
    """Inference runs on one worker, separate from CameraManager's capture worker.

    Camera frames are obtained only when the scheduler is ready to infer. The
    manager retains one newest frame, and inference never builds a frame queue.
    The model is loaded once per engine start and retained during pause.
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        *,
        model_factory: ModelFactory | None = None,
        clock: Callable[[], float] = monotonic,
        timer: Callable[[], float] = perf_counter,
    ) -> None:
        self.camera_manager = camera_manager
        self._factory = model_factory or (lambda path, threads: LiteHandLandmarkModel(path, num_threads=threads))
        self._clock = clock
        self._timer = timer
        self._lock = Lock()
        self._status = EngineStatus()
        self._worker: Thread | None = None
        self._config: EngineConfig | None = None
        self._processing_rate_hz = 8.0
        self._stop = Event()
        self._pause = Event()

    @property
    def status(self) -> EngineStatus:
        with self._lock:
            return self._status

    @property
    def is_worker_alive(self) -> bool:
        with self._lock:
            worker = self._worker
        return worker is not None and worker.is_alive()

    def start(self, config: EngineConfig) -> bool:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return False
            self._stop = Event()
            self._pause = Event()
            self._config = config
            self._processing_rate_hz = config.processing_rate_hz
            self._status = EngineStatus(state=EngineState.STARTING,
                                        processed_frames=self._status.processed_frames)
            worker = Thread(target=self._run, name="GestureManagerHandInference", daemon=False)
            self._worker = worker
            try:
                worker.start()
            except Exception as exc:
                self._status = EngineStatus(state=EngineState.ERROR, error=f"Could not start inference worker: {exc}")
                return False
        return True

    def pause(self) -> bool:
        with self._lock:
            if self._status.state not in (EngineState.STARTING, EngineState.RUNNING):
                return False
            self._pause.set()
            self._status = EngineStatus(
                state=EngineState.PAUSED,
                model_loaded=self._status.model_loaded,
                hands_detected=self._status.hands_detected,
                hand_count=self._status.hand_count,
                hands=self._status.hands,
                inference_latency_ms=self._status.inference_latency_ms,
                processed_frames=self._status.processed_frames,
            )
            return True

    def set_processing_rate(self, rate_hz: float) -> None:
        """Change the target inference cadence without restarting the model."""
        if not 1 <= rate_hz <= 30:
            raise ValueError("Processing rate must be between 1 and 30 Hz")
        with self._lock:
            self._processing_rate_hz = float(rate_hz)

    def resume(self) -> bool:
        with self._lock:
            if self._status.state is not EngineState.PAUSED:
                return False
            self._pause.clear()
            self._status = EngineStatus(state=EngineState.STARTING, model_loaded=self._status.model_loaded,
                                        processed_frames=self._status.processed_frames)
            return True

    def stop(self) -> bool:
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._status = EngineStatus(state=EngineState.STOPPED,
                                            processed_frames=self._status.processed_frames)
                return False
            if self._status.state is not EngineState.ERROR:
                self._status = EngineStatus(state=EngineState.STOPPING,
                                            model_loaded=self._status.model_loaded,
                                            processed_frames=self._status.processed_frames)
            self._stop.set()
            self._pause.clear()
            self.camera_manager.stop()
            return True

    def wait_for_stop(self, timeout: float | None = None) -> bool:
        with self._lock:
            worker = self._worker
        if worker is None:
            return True
        if worker is current_thread():
            return False
        worker.join(timeout)
        return not worker.is_alive()

    def _set(self, **changes: object) -> None:
        with self._lock:
            self._status = EngineStatus(**(self._status.__dict__ | changes))

    def _run(self) -> None:
        model: ModelLike | None = None
        config = self._config
        error: str | None = None
        try:
            assert config is not None
            model = self._factory(Path(config.model_dir), config.num_threads)
            model.load()
            if self._stop.is_set():
                return
            self._set(model_loaded=True)
            camera_started = False
            camera_release_pending = False
            last_sequence = -1
            next_inference = self._clock()
            while not self._stop.is_set():
                if self._pause.is_set():
                    if camera_started:
                        camera_release_pending = True
                        self.camera_manager.stop()
                        self.camera_manager.wait_for_stop(0.1)
                        if not self.camera_manager.is_worker_alive:
                            camera_started = False
                            camera_release_pending = False
                    # A strict pause releases the camera; poll slowly so the
                    # retained worker/model does not wake frequently while idle.
                    self._stop.wait(0.25)
                    continue
                if not camera_started:
                    status = self.camera_manager.status
                    if status.state is CameraState.RUNNING:
                        raise RuntimeError("Stop standalone camera capture before starting hand detection")
                    if status.state in (CameraState.STARTING, CameraState.STOPPING):
                        self._stop.wait(0.05)
                        continue
                    if status.state is CameraState.ERROR:
                        raise RuntimeError(status.error or "Camera failed to start")
                    if not self.camera_manager.start(config.camera):
                        raise RuntimeError("Camera is already running or stopping")
                    camera_started = True
                    last_sequence = -1
                    self._stop.wait(0.02)
                    continue
                camera_status = self.camera_manager.status
                if camera_started and camera_status.state in (
                        CameraState.STARTING, CameraState.STOPPING):
                    # Resume can arrive before the camera worker finishes the
                    # strict-pause release. Wait for it, then reopen cleanly.
                    self._stop.wait(0.02)
                    continue
                if camera_started and camera_status.state is CameraState.STOPPED:
                    if camera_release_pending:
                        camera_started = False
                        camera_release_pending = False
                        continue
                    if not self._stop.is_set():
                        raise RuntimeError("Camera stopped unexpectedly")
                if camera_status.state is CameraState.ERROR:
                    raise RuntimeError(camera_status.error or "Camera capture failed")
                now = self._clock()
                if now < next_inference:
                    self._stop.wait(min(0.02, next_inference - now))
                    continue
                newest = self.camera_manager.get_latest_frame()
                if newest is None or newest[0] == last_sequence:
                    self._stop.wait(0.01)
                    continue
                sequence, frame = newest
                last_sequence = sequence
                started = self._timer()
                hands = model.detect(frame)
                latency = max(0.0, (self._timer() - started) * 1000.0)
                with self._lock:
                    current = self._status
                    current_state = (
                        EngineState.STOPPING if self._stop.is_set() else
                        EngineState.PAUSED if self._pause.is_set() else EngineState.RUNNING
                    )
                    self._status = EngineStatus(
                        state=current_state, model_loaded=True, hands_detected=bool(hands),
                        hand_count=len(hands), hands=hands, inference_latency_ms=latency,
                        processed_frames=current.processed_frames + 1,
                    )
                # Schedule from completion to prevent catch-up bursts.
                with self._lock:
                    rate_hz = self._processing_rate_hz
                next_inference = self._clock() + 1.0 / rate_hz
        except Exception as exc:
            error = f"Hand landmark engine error: {type(exc).__name__}: {exc}"
            with self._lock:
                self._status = EngineStatus(state=EngineState.ERROR, model_loaded=self._status.model_loaded,
                                            processed_frames=self._status.processed_frames, error=error)
        finally:
            self.camera_manager.stop()
            self.camera_manager.wait_for_stop(5.0)
            if model is not None:
                try:
                    model.close()
                except Exception:
                    pass
            with self._lock:
                if self._status.state is not EngineState.ERROR:
                    self._status = EngineStatus(state=EngineState.STOPPED,
                                                processed_frames=self._status.processed_frames)
