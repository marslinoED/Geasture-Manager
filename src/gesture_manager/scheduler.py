"""Pure runtime FPS policy and Windows foreground-window detection."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from typing import Protocol

from gesture_manager.runtime import RuntimeState


class FocusAPI(Protocol):
    def GetForegroundWindow(self) -> int: ...
    def GetAncestor(self, hwnd: int, flags: int) -> int: ...


class WindowsFocusDetector:
    """Detect whether the app's top-level HWND is the foreground window."""

    GA_ROOT = 2

    def __init__(self, api: FocusAPI | None = None) -> None:
        self._api = api
        if self._api is None and os.name == "nt":
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            user32.GetForegroundWindow.restype = ctypes.c_void_p
            user32.GetAncestor.argtypes = (ctypes.c_void_p, ctypes.c_uint)
            user32.GetAncestor.restype = ctypes.c_void_p
            self._api = user32

    def is_focused(self, hwnd: int) -> bool | None:
        if self._api is None or hwnd <= 0:
            return None
        foreground = int(self._api.GetForegroundWindow())
        if foreground <= 0:
            return False
        foreground_root = int(self._api.GetAncestor(foreground, self.GA_ROOT)) or foreground
        app_root = int(self._api.GetAncestor(hwnd, self.GA_ROOT)) or hwnd
        return foreground_root == app_root


@dataclass(frozen=True)
class Schedule:
    mode: str
    target_fps: float
    release_camera: bool
    run_recognition: bool


class AdaptiveScheduler:
    """Select requested work rates; rates are targets, not throughput promises."""

    # Keep recognition responsive while another application has focus: that
    # is the normal case when gestures control shortcuts in that application.
    # Camera capture defaults to 15 FPS, so the focused target matches it.
    ACTIVE_FOCUSED_FPS = 15.0
    ACTIVE_UNFOCUSED_FPS = 10.0
    SHORTCUT_PAUSED_FPS = 2.5
    PAUSED_WAKE_FPS = 2.5

    @classmethod
    def schedule(cls, state: RuntimeState, *, focused: bool,
                 wake_detection: bool = False, strict_pause: bool = True,
                 shortcut_paused: bool = False) -> Schedule:
        if state in (RuntimeState.STOPPED, RuntimeState.STOPPING, RuntimeState.ERROR):
            return Schedule("stopped", 0.0, True, False)
        if state is RuntimeState.PAUSED:
            if wake_detection and not strict_pause:
                return Schedule("paused_wake", cls.PAUSED_WAKE_FPS, False, False)
            return Schedule("strict_paused", 0.0, True, False)
        if shortcut_paused:
            # Keep control gestures available while ordinary actions are
            # blocked, with much less inference work than the active mode.
            return Schedule("shortcuts_paused", cls.SHORTCUT_PAUSED_FPS, False, True)
        if focused:
            return Schedule("active_focused", cls.ACTIVE_FOCUSED_FPS, False, True)
        return Schedule("active_unfocused", cls.ACTIVE_UNFOCUSED_FPS, False, True)
