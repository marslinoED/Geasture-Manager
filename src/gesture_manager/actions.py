"""Versioned local shortcut mappings and allowlisted keyboard dispatch."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from threading import Lock
from time import monotonic
from typing import Any, Protocol


ACTION_FORMAT = "gesture-manager-actions"
ACTION_VERSION = 1
SUPPORTED_ACTION_TYPES = frozenset({"keyboard_shortcut"})
MODIFIERS = ("Ctrl", "Alt", "Shift", "Win")
_KEY_ALIASES = {
    "control": "Ctrl", "ctrl": "Ctrl", "alt": "Alt", "shift": "Shift",
    "win": "Win", "windows": "Win", "cmd": "Win", "command": "Win",
    "esc": "Esc", "escape": "Esc", "return": "Enter", "enter": "Enter",
    "spacebar": "Space", "space": "Space", "pgup": "PageUp", "pageup": "PageUp",
    "pgdn": "PageDown", "pagedown": "PageDown", "left": "Left", "right": "Right",
    "up": "Up", "down": "Down", "del": "Delete", "delete": "Delete",
    "ins": "Insert", "insert": "Insert", "backspace": "Backspace",
    "tab": "Tab", "home": "Home", "end": "End", "plus": "Plus",
    "minus": "Minus", "comma": "Comma", "period": "Period",
    "capslock": "CapsLock", "caps lock": "CapsLock",
    "printscreen": "PrintScreen", "print screen": "PrintScreen",
    "scrolllock": "ScrollLock", "scroll lock": "ScrollLock",
    "pause": "Pause", "menu": "Menu", "apps": "Menu",
    "numlock": "NumLock", "num lock": "NumLock",
    "volumeup": "VolumeUp", "volume up": "VolumeUp",
    "volumedown": "VolumeDown", "volume down": "VolumeDown",
    "volumemute": "VolumeMute", "volume mute": "VolumeMute", "mute": "VolumeMute",
    "medianext": "MediaNext", "media next": "MediaNext",
    "mediaprevious": "MediaPrevious", "media previous": "MediaPrevious",
    "mediastop": "MediaStop", "media stop": "MediaStop",
    "mediaplaypause": "MediaPlayPause", "media play pause": "MediaPlayPause",
    "semicolon": "Semicolon", "equal": "Equal", "equals": "Equal",
    "openbracket": "OpenBracket", "leftbracket": "OpenBracket",
    "backslash": "Backslash", "closebracket": "CloseBracket",
    "rightbracket": "CloseBracket", "quote": "Quote", "apostrophe": "Quote",
    "slash": "Slash", "grave": "Grave", "backtick": "Grave",
    ";": "Semicolon", "=": "Equal", "[": "OpenBracket", "\\": "Backslash",
    "]": "CloseBracket", "'": "Quote", "/": "Slash", "`": "Grave",
    "-": "Minus", ",": "Comma", ".": "Period",
}
_SUPPORTED_MAIN_KEYS = frozenset(
    set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    | {f"F{number}" for number in range(1, 13)}
    | {"Tab", "Enter", "Esc", "Space", "Backspace", "Delete", "Insert",
       "Home", "End", "PageUp", "PageDown", "Left", "Right", "Up", "Down",
       "Plus", "Minus", "Comma", "Period", "CapsLock", "PrintScreen",
       "ScrollLock", "Pause", "Menu", "NumLock", "Semicolon", "Equal",
       "OpenBracket", "Backslash", "CloseBracket", "Quote", "Slash", "Grave"}
    | {"VolumeUp", "VolumeDown", "VolumeMute", "MediaNext", "MediaPrevious",
       "MediaStop", "MediaPlayPause"}
)


@dataclass(frozen=True)
class ActionMapping:
    gesture_id: str
    gesture_name: str
    action_type: str
    shortcut: str
    enabled: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.gesture_id, str) or not self.gesture_id.strip()
                or not isinstance(self.gesture_name, str) or not self.gesture_name.strip()):
            raise ValueError("Gesture ID and name are required")
        if not isinstance(self.action_type, str) or self.action_type not in SUPPORTED_ACTION_TYPES:
            raise ValueError("Unsupported action type")
        normalized = normalize_shortcut(self.shortcut)
        object.__setattr__(self, "shortcut", normalized)
        if not isinstance(self.enabled, bool):
            raise ValueError("Action enabled flag must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "gesture_id": self.gesture_id,
            "gesture_name": self.gesture_name,
            "action_type": self.action_type,
            "shortcut": self.shortcut,
            "enabled": self.enabled,
        }


def normalize_shortcut(shortcut: str) -> str:
    """Validate one supported keyboard key, optionally with modifiers."""
    if not isinstance(shortcut, str):
        raise ValueError("Shortcut must be text")
    pieces = [piece.strip() for piece in shortcut.split("+")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("Enter a shortcut such as Ctrl+C or Alt+Tab")
    normalized: list[str] = []
    for piece in pieces:
        alias = _KEY_ALIASES.get(piece.casefold())
        if alias is None:
            upper = piece.upper()
            if upper in _SUPPORTED_MAIN_KEYS:
                alias = upper
            else:
                raise ValueError(f"Unsupported key: {piece}")
        normalized.append(alias)
    modifiers = [key for key in normalized if key in MODIFIERS]
    main_keys = [key for key in normalized if key not in MODIFIERS]
    if len(main_keys) != 1 or len(set(normalized)) != len(normalized):
        raise ValueError("Use one main key, optionally preceded by modifiers")
    ordered_modifiers = [modifier for modifier in MODIFIERS if modifier in modifiers]
    normalized_shortcut = "+".join((*ordered_modifiers, main_keys[0]))
    if normalized_shortcut == "Ctrl+Alt+Delete":
        raise ValueError("Ctrl+Alt+Delete is reserved by Windows and cannot be dispatched safely")
    return normalized_shortcut


def pynput_key_token(key: Any, key_enum: Any) -> str | None:
    """Translate a pynput key event into one allowlisted shortcut token."""
    modifier_names = {
        "Ctrl": ("ctrl", "ctrl_l", "ctrl_r"),
        "Alt": ("alt", "alt_l", "alt_r", "alt_gr"),
        "Shift": ("shift", "shift_l", "shift_r"),
        "Win": ("cmd", "cmd_l", "cmd_r"),
    }
    for token, names in modifier_names.items():
        for name in names:
            enum_key = getattr(key_enum, name, None)
            if enum_key is not None and key == enum_key:
                return token
    special_names = {
        "tab": "Tab", "enter": "Enter", "esc": "Esc", "space": "Space",
        "backspace": "Backspace", "delete": "Delete", "insert": "Insert",
        "home": "Home", "end": "End", "page_up": "PageUp", "page_down": "PageDown",
        "left": "Left", "right": "Right", "up": "Up", "down": "Down",
        "caps_lock": "CapsLock", "print_screen": "PrintScreen",
        "scroll_lock": "ScrollLock", "pause": "Pause", "menu": "Menu",
        "num_lock": "NumLock",
        "media_volume_up": "VolumeUp", "media_volume_down": "VolumeDown",
        "media_volume_mute": "VolumeMute", "media_next": "MediaNext",
        "media_previous": "MediaPrevious", "media_stop": "MediaStop",
        "media_play_pause": "MediaPlayPause",
    }
    # pynput exposes function keys as enum members on non-Windows backends too;
    # they do not necessarily carry a Windows virtual-key code or character.
    special_names.update({f"f{number}": f"F{number}" for number in range(1, 13)})
    for name, token in special_names.items():
        enum_key = getattr(key_enum, name, None)
        if enum_key is not None and key == enum_key:
            return token
    char = getattr(key, "char", None)
    if isinstance(char, str) and len(char) == 1:
        if char.isalpha() or char.isdigit():
            return char.upper()
        punctuation = {
            "+": "Plus", "-": "Minus", ",": "Comma", ".": "Period",
            ";": "Semicolon", "=": "Equal", "[": "OpenBracket",
            "\\": "Backslash", "]": "CloseBracket", "'": "Quote",
            "/": "Slash", "`": "Grave",
        }
        if char in punctuation:
            return punctuation[char]
    virtual_key = getattr(key, "vk", None)
    if isinstance(virtual_key, int):
        token = windows_vk_token(virtual_key)
        if token is not None:
            return token
    return None


def windows_vk_token(virtual_key: int) -> str | None:
    """Translate a Windows virtual-key code for the recorder's low-level hook."""
    modifiers = {
        0x10: "Shift", 0xA0: "Shift", 0xA1: "Shift",
        0x11: "Ctrl", 0xA2: "Ctrl", 0xA3: "Ctrl",
        0x12: "Alt", 0xA4: "Alt", 0xA5: "Alt",
        0x5B: "Win", 0x5C: "Win",
    }
    if virtual_key in modifiers:
        return modifiers[virtual_key]
    if 0x41 <= virtual_key <= 0x5A:
        return chr(virtual_key)
    if 0x30 <= virtual_key <= 0x39:
        return chr(virtual_key)
    if 0x70 <= virtual_key <= 0x7B:
        return f"F{virtual_key - 0x6F}"
    special = {
        0x09: "Tab", 0x0D: "Enter", 0x1B: "Esc", 0x20: "Space",
        0x08: "Backspace", 0x2E: "Delete", 0x2D: "Insert",
        0x24: "Home", 0x23: "End", 0x21: "PageUp", 0x22: "PageDown",
        0x25: "Left", 0x27: "Right", 0x26: "Up", 0x28: "Down",
        0xBB: "Plus", 0xBD: "Minus", 0xBC: "Comma", 0xBE: "Period",
        0x14: "CapsLock", 0x2C: "PrintScreen", 0x91: "ScrollLock",
        0x13: "Pause", 0x5D: "Menu", 0x90: "NumLock",
        0xAD: "VolumeMute", 0xAE: "VolumeDown", 0xAF: "VolumeUp",
        0xB0: "MediaNext", 0xB1: "MediaPrevious", 0xB2: "MediaStop",
        0xB3: "MediaPlayPause",
        0xBA: "Semicolon", 0xBB: "Plus", 0xBD: "Minus", 0xBC: "Comma",
        0xBE: "Period", 0xBF: "Slash", 0xC0: "Grave", 0xDB: "OpenBracket",
        0xDC: "Backslash", 0xDD: "CloseBracket", 0xDE: "Quote",
    }
    return special.get(virtual_key)


class ActionMappingStore:
    """Safe JSON persistence separate from datasets and classifier models."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @staticmethod
    def default_path() -> Path:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "GestureManager" / "settings" / "actions.json"

    def load(self) -> dict[str, ActionMapping]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read action mappings: {exc}") from exc
        if (not isinstance(data, dict)
                or set(data) != {"format", "version", "mappings"}
                or data.get("format") != ACTION_FORMAT):
            raise ValueError("Unsupported action mapping file format")
        if type(data.get("version")) is not int or data.get("version") != ACTION_VERSION:
            raise ValueError("Unsupported action mapping version")
        records = data.get("mappings")
        if not isinstance(records, list):
            raise ValueError("Action mapping list is invalid")
        result: dict[str, ActionMapping] = {}
        try:
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("Action mapping record must be an object")
                if set(record) != {"gesture_id", "gesture_name", "action_type", "shortcut", "enabled"}:
                    raise ValueError("Action mapping contains unsupported fields")
                mapping = ActionMapping(
                    gesture_id=record["gesture_id"],
                    gesture_name=record["gesture_name"],
                    action_type=record["action_type"],
                    shortcut=record["shortcut"],
                    enabled=record["enabled"],
                )
                if mapping.gesture_id in result:
                    raise ValueError("Duplicate gesture mapping")
                result[mapping.gesture_id] = mapping
        except (KeyError, TypeError) as exc:
            raise ValueError("Action mapping record is incomplete") from exc
        return result

    def save(self, mappings: dict[str, ActionMapping]) -> None:
        if any(key != mapping.gesture_id for key, mapping in mappings.items()):
            raise ValueError("Mapping keys must match gesture IDs")
        payload = {
            "format": ACTION_FORMAT,
            "version": ACTION_VERSION,
            "mappings": [mappings[key].to_dict() for key in sorted(mappings)],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                             prefix=f".{self.path.name}.", suffix=".tmp",
                                             delete=False) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


class KeyboardBackend(Protocol):
    def send_shortcut(self, shortcut: str) -> None: ...


class PynputKeyboardBackend:
    """Windows desktop keyboard backend; imported only after real-input opt-in."""

    _MODIFIER_KEYS = {"Ctrl": "ctrl", "Alt": "alt", "Shift": "shift", "Win": "cmd"}
    _SPECIAL_KEYS = {
        "Tab": "tab", "Enter": "enter", "Esc": "esc", "Space": "space",
        "Backspace": "backspace", "Delete": "delete", "Insert": "insert",
        "Home": "home", "End": "end", "PageUp": "page_up", "PageDown": "page_down",
        "Left": "left", "Right": "right", "Up": "up", "Down": "down",
        "CapsLock": "caps_lock", "PrintScreen": "print_screen",
        "ScrollLock": "scroll_lock", "Pause": "pause", "Menu": "menu",
        "NumLock": "num_lock",
        "VolumeUp": "media_volume_up", "VolumeDown": "media_volume_down",
        "VolumeMute": "media_volume_mute", "MediaNext": "media_next",
        "MediaPrevious": "media_previous", "MediaStop": "media_stop",
        "MediaPlayPause": "media_play_pause",
        "Plus": "+", "Minus": "-", "Comma": ",", "Period": ".",
        "Semicolon": ";", "Equal": "=", "OpenBracket": "[",
        "Backslash": "\\", "CloseBracket": "]", "Quote": "'",
        "Slash": "/", "Grave": "`",
    }

    def __init__(self) -> None:
        try:
            from pynput.keyboard import Controller, Key
        except ImportError as exc:
            raise RuntimeError(
                "Real keyboard input requires pynput. Install the project dependencies and restart."
            ) from exc
        self._controller = Controller()
        self._key = Key

    def normalize(self, shortcut: str) -> str:
        return normalize_shortcut(shortcut)

    def _resolve(self, token: str):
        if token in self._MODIFIER_KEYS:
            return getattr(self._key, self._MODIFIER_KEYS[token])
        if token.startswith("F") and token[1:].isdigit():
            return getattr(self._key, token.lower())
        if token in self._SPECIAL_KEYS:
            special = self._SPECIAL_KEYS[token]
            if special in {"+", "-", ",", "."}:
                from pynput.keyboard import KeyCode
                return KeyCode.from_char(special)
            return getattr(self._key, special)
        from pynput.keyboard import KeyCode
        return KeyCode.from_char(token.lower())

    def send_shortcut(self, shortcut: str) -> None:
        tokens = normalize_shortcut(shortcut).split("+")
        pressed: list[object] = []
        try:
            for token in tokens:
                key = self._resolve(token)
                self._controller.press(key)
                pressed.append(key)
        finally:
            release_error: Exception | None = None
            for key in reversed(pressed):
                try:
                    self._controller.release(key)
                except Exception as exc:  # Release every modifier even if one release fails.
                    release_error = release_error or exc
            if release_error is not None:
                raise release_error


@dataclass(frozen=True)
class DispatchResult:
    status: str
    message: str
    shortcut: str | None = None


class ActionDispatcher:
    """Allowlisted, cooldown-limited action dispatch with emergency stop."""

    def __init__(self, backend: KeyboardBackend | None = None, *,
                 cooldown_seconds: float = 1.0,
                 global_cooldown_seconds: float = 0.25) -> None:
        if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
            raise ValueError("Action cooldown must be finite and non-negative")
        if not math.isfinite(global_cooldown_seconds) or global_cooldown_seconds < 0:
            raise ValueError("Global action cooldown must be finite and non-negative")
        self._backend = backend
        self._dry_run = True
        self._real_input_confirmed = False
        self.cooldown_seconds = cooldown_seconds
        self.global_cooldown_seconds = global_cooldown_seconds
        self._stopped = False
        self._last_action: dict[str, float] = {}
        self._last_dispatch: float | None = None
        self._lock = Lock()

    @property
    def dry_run(self) -> bool:
        with self._lock:
            return self._dry_run

    @property
    def emergency_stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def set_dry_run(self, enabled: bool) -> None:
        with self._lock:
            if not enabled and (not self._real_input_confirmed or self._backend is None):
                raise RuntimeError("Confirm real input and configure a keyboard backend first")
            self._dry_run = bool(enabled)
            if enabled:
                self._real_input_confirmed = False

    def set_backend(self, backend: KeyboardBackend) -> None:
        with self._lock:
            self._backend = backend

    def confirm_real_input(self, *, confirmed: bool) -> None:
        """Enable real input only after the UI's explicit user confirmation."""
        if not confirmed:
            raise ValueError("Real keyboard input was not confirmed")
        with self._lock:
            if self._stopped:
                raise RuntimeError("Rearm the action dispatcher before enabling real input")
            if self._backend is None:
                raise RuntimeError("No keyboard backend is configured")
            self._real_input_confirmed = True
            self._dry_run = False

    def emergency_stop(self) -> None:
        with self._lock:
            self._stopped = True
            self._dry_run = True
            self._real_input_confirmed = False

    def rearm(self) -> None:
        with self._lock:
            self._stopped = False
            self._dry_run = True
            self._real_input_confirmed = False
            self._last_action.clear()
            self._last_dispatch = None

    def dispatch(self, event: object, mappings: dict[str, ActionMapping], *,
                 now: float | None = None) -> DispatchResult:
        if not getattr(event, "trigger_eligible", False):
            return DispatchResult("ignored", "Recognition event is not trigger-eligible")
        gesture_id = getattr(event, "gesture_id", None)
        mapping = mappings.get(gesture_id)
        if mapping is None or not mapping.enabled:
            return DispatchResult("disabled", "No enabled action is assigned to this gesture")
        if mapping.action_type not in SUPPORTED_ACTION_TYPES:
            return DispatchResult("rejected", "Action type is not allowlisted")
        try:
            shortcut = normalize_shortcut(mapping.shortcut)
        except ValueError as exc:
            return DispatchResult("rejected", str(exc))
        instant = monotonic() if now is None else now
        with self._lock:
            if self._stopped:
                return DispatchResult("emergency_stopped", "Global emergency stop is active")
            previous = self._last_action.get(gesture_id)
            if previous is not None and instant - previous < self.cooldown_seconds:
                return DispatchResult("cooldown", "Action cooldown is active", shortcut)
            if self._last_dispatch is not None and instant - self._last_dispatch < self.global_cooldown_seconds:
                return DispatchResult("cooldown", "Global action rate limit is active", shortcut)
            self._last_action[gesture_id] = instant
            self._last_dispatch = instant
            dry_run = self._dry_run
            backend = self._backend
        if dry_run:
            return DispatchResult("dry_run", f"Dry run: {mapping.gesture_name} would send {shortcut}", shortcut)
        if backend is None:
            return DispatchResult("error", "No keyboard backend is configured", shortcut)
        try:
            backend.send_shortcut(shortcut)
        except Exception as exc:
            return DispatchResult("error", f"Keyboard action failed: {type(exc).__name__}: {exc}", shortcut)
        return DispatchResult("sent", f"Sent {shortcut} for {mapping.gesture_name}", shortcut)
