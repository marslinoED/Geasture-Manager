"""Camera-independent runtime state and command handling."""

from __future__ import annotations

from enum import Enum
import logging
from queue import Empty, SimpleQueue
from threading import RLock
from typing import Callable, Generic, TypeVar


class RuntimeState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class RuntimeCommand(str, Enum):
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"


class QueueResult(Enum):
    CONTINUE = "continue"
    STOP = "stop"


LifecycleAction = Callable[[], None]
StateListener = Callable[[RuntimeState, str | None], None]
CommandT = TypeVar("CommandT")
logger = logging.getLogger(__name__)


class RuntimeController:
    """Own runtime state and call injected lifecycle actions.

    The controller remains camera-independent; the application injects its
    camera and recognition lifecycle from the Tk main-thread command handler.
    """

    def __init__(
        self,
        *,
        start_action: LifecycleAction | None = None,
        pause_action: LifecycleAction | None = None,
        resume_action: LifecycleAction | None = None,
        stop_action: LifecycleAction | None = None,
        on_change: StateListener | None = None,
    ) -> None:
        self._start_action = start_action or (lambda: None)
        self._pause_action = pause_action or (lambda: None)
        self._resume_action = resume_action or (lambda: None)
        self._stop_action = stop_action or (lambda: None)
        self._on_change = on_change
        self._state = RuntimeState.STOPPED
        self._error: str | None = None
        self._lock = RLock()

    @property
    def state(self) -> RuntimeState:
        with self._lock:
            return self._state

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def handle_command(self, command: RuntimeCommand) -> bool:
        """Apply a command; invalid-for-current-state commands are harmless."""
        handlers = {
            RuntimeCommand.START: self.start,
            RuntimeCommand.PAUSE: self.pause,
            RuntimeCommand.RESUME: self.resume,
            RuntimeCommand.STOP: self.stop,
        }
        handler = handlers.get(command)
        if handler is None:
            return False
        return handler()

    def report_error(self, message: str) -> None:
        """Publish an application error through the same shared state."""
        with self._lock:
            self._error = message
            self._set_state(RuntimeState.ERROR)

    def start(self) -> bool:
        with self._lock:
            if self._state not in (RuntimeState.STOPPED, RuntimeState.ERROR):
                return False
            self._error = None
            self._set_state(RuntimeState.STARTING)
            try:
                self._start_action()
            except Exception as exc:  # Runtime failures are shown in Settings.
                self._fail(exc)
                return True
            self._error = None
            self._set_state(RuntimeState.ACTIVE)
            return True

    def pause(self) -> bool:
        with self._lock:
            if self._state is not RuntimeState.ACTIVE:
                return False
            try:
                self._pause_action()
            except Exception as exc:
                self._fail(exc)
                return True
            self._set_state(RuntimeState.PAUSED)
            return True

    def resume(self) -> bool:
        with self._lock:
            if self._state is not RuntimeState.PAUSED:
                return False
            try:
                self._resume_action()
            except Exception as exc:
                self._fail(exc)
                return True
            self._set_state(RuntimeState.ACTIVE)
            return True

    def stop(self) -> bool:
        with self._lock:
            if self._state is RuntimeState.STOPPED:
                return False
            self._set_state(RuntimeState.STOPPING)
            try:
                self._stop_action()
            except Exception as exc:
                self._fail(exc)
                return True
            self._error = None
            self._set_state(RuntimeState.STOPPED)
            return True

    def _fail(self, exc: Exception) -> None:
        logger.exception("Runtime lifecycle action failed")
        self._error = f"{type(exc).__name__}: {exc}"
        self._set_state(RuntimeState.ERROR)

    def _set_state(self, state: RuntimeState) -> None:
        self._state = state
        if self._on_change is not None:
            try:
                self._on_change(state, self._error)
            except Exception:
                logger.exception("Runtime state listener failed")


class CommandQueue(Generic[CommandT]):
    """Thread-safe queue shared by tray callbacks and the Tk event loop.

    Producers only enqueue enum commands. The Tk main thread drains the queue
    and invokes the controller, so tray worker threads never touch Tk objects.
    """

    def __init__(self) -> None:
        self._queue: SimpleQueue[CommandT] = SimpleQueue()

    def submit(self, command: CommandT) -> None:
        self._queue.put(command)

    def process(self, handler: Callable[[CommandT], object]) -> int:
        """Drain pending commands and return the number that were handled."""
        processed = 0
        while True:
            try:
                command = self._queue.get_nowait()
            except Empty:
                return processed
            result = handler(command)
            processed += 1
            if result is QueueResult.STOP:
                return processed
