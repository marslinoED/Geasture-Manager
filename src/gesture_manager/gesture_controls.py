"""Fail-closed routing for static gesture pause, resume, and stop controls."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math


class GestureControlRouter:
    """Consume configured control gestures before ordinary action mappings.

    Recognition stability and per-label thresholds are enforced by the live
    recognition gate. This router independently requires an eligible event,
    finite positive score, a fresh timezone-aware timestamp, and a new frame.
    """

    def __init__(self, *, max_age_seconds: float = 3.0,
                 future_skew_seconds: float = 0.5,
                 minimum_pause_seconds: float = 1.0) -> None:
        if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be finite and positive")
        if not math.isfinite(future_skew_seconds) or future_skew_seconds < 0:
            raise ValueError("future_skew_seconds must be finite and non-negative")
        if not math.isfinite(minimum_pause_seconds) or minimum_pause_seconds < 0:
            raise ValueError("minimum_pause_seconds must be finite and non-negative")
        self.max_age = timedelta(seconds=max_age_seconds)
        self.future_skew = timedelta(seconds=future_skew_seconds)
        self.minimum_pause = timedelta(seconds=minimum_pause_seconds)
        self.pause_gesture_id: str | None = None
        self.resume_gesture_id: str | None = None
        self.stop_gesture_id: str | None = None
        self.actions_paused = False
        self._last_control_sequence: int | None = None
        self._last_ordinary_sequence: int | None = None
        self._shared_control_armed = True
        self._paused_since: datetime | None = None
        self.last_rejection_reason = ""

    def configure(self, *, pause_gesture_id: str | None,
                  resume_gesture_id: str | None,
                  stop_gesture_id: str | None) -> None:
        if stop_gesture_id is not None and stop_gesture_id in {
                pause_gesture_id, resume_gesture_id}:
            raise ValueError("Stop Gesture cannot use the same gesture as Pause or Resume")
        selected = [value for value in
                    (pause_gesture_id, resume_gesture_id, stop_gesture_id)
                    if value is not None]
        if any(not isinstance(value, str) or not value.strip() or len(value) > 128
               for value in selected):
            raise ValueError("Control gesture ID is invalid")
        self.pause_gesture_id = pause_gesture_id
        self.resume_gesture_id = resume_gesture_id
        self.stop_gesture_id = stop_gesture_id
        self.reset()

    def reset(self) -> None:
        self.actions_paused = False
        self._last_control_sequence = None
        self._last_ordinary_sequence = None
        self._shared_control_armed = True
        self._paused_since = None
        self.last_rejection_reason = ""

    def set_actions_paused(self, paused: bool) -> None:
        paused = bool(paused)
        if paused != self.actions_paused:
            self._paused_since = datetime.now(timezone.utc) if paused else None
        self.actions_paused = paused

    @property
    def ordinary_actions_allowed(self) -> bool:
        return not self.actions_paused

    def dispatch_ordinary(self, event: object, dispatcher: object,
                          mappings: object) -> object | None:
        """Apply the pause and control-ID gates before calling the dispatcher."""
        gesture_id = getattr(event, "gesture_id", None)
        controls = {value for value in (self.pause_gesture_id,
                                        self.resume_gesture_id,
                                        self.stop_gesture_id) if value is not None}
        if (self.actions_paused or gesture_id in controls
                or getattr(event, "trigger_eligible", False) is not True):
            return None
        score = getattr(event, "decision_score", None)
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or score <= 0):
            return None
        sequence = getattr(event, "frame_sequence", None)
        if (isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0
                or (self._last_ordinary_sequence is not None
                    and sequence <= self._last_ordinary_sequence)):
            return None
        try:
            stamp = datetime.fromisoformat(getattr(event, "timestamp", ""))
            if stamp.tzinfo is None:
                return None
            age = (datetime.now(timezone.utc) - stamp.astimezone(timezone.utc))
        except (TypeError, ValueError):
            return None
        if age > self.max_age or age < -self.future_skew:
            return None
        self._last_ordinary_sequence = sequence
        dispatch = getattr(dispatcher, "dispatch", None)
        if not callable(dispatch):
            raise TypeError("dispatcher must provide dispatch(event, mappings)")
        return dispatch(event, mappings)

    def route(self, event: object, *, now: datetime | None = None) -> str | None:
        """Return ``pause``, ``resume``, ``stop``, ``rejected``, or ``None``.

        Any event matching a configured control is consumed, even if invalid,
        so a rejected control candidate can never fall through to a shortcut.
        """
        gesture_id = getattr(event, "gesture_id", None)
        if (self.pause_gesture_id is not None
                and self.pause_gesture_id == self.resume_gesture_id == gesture_id):
            if not self._shared_control_armed:
                self.last_rejection_reason = "shared Pause/Resume gesture is waiting for release"
                return "rejected"
            control = "resume" if self.actions_paused else "pause"
        else:
            # A different prediction (including no-hand/uncertain) releases the
            # shared control pose so it can be used for the next state change.
            if self.pause_gesture_id is not None and self.pause_gesture_id == self.resume_gesture_id:
                self._shared_control_armed = True
            control = next((kind for configured, kind in (
                (self.pause_gesture_id, "pause"),
                (self.resume_gesture_id, "resume"),
                (self.stop_gesture_id, "stop"),
            ) if configured is not None and gesture_id == configured), None)
        if control is None:
            return None
        if getattr(event, "trigger_eligible", False) is not True:
            self.last_rejection_reason = "recognition gate marked the event ineligible"
            return "rejected"
        score = getattr(event, "decision_score", None)
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or score <= 0):
            self.last_rejection_reason = "decision score is missing or invalid"
            return "rejected"
        sequence = getattr(event, "frame_sequence", None)
        if (isinstance(sequence, bool) or not isinstance(sequence, int)
                or sequence < 0 or (self._last_control_sequence is not None
                                    and sequence <= self._last_control_sequence)):
            self.last_rejection_reason = "control event frame is invalid or duplicated"
            return "rejected"
        try:
            stamp = datetime.fromisoformat(getattr(event, "timestamp", ""))
            if stamp.tzinfo is None:
                raise ValueError("timezone missing")
            stamp = stamp.astimezone(timezone.utc)
        except (TypeError, ValueError):
            self.last_rejection_reason = "control event timestamp is invalid"
            return "rejected"
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        age = current.astimezone(timezone.utc) - stamp
        if age > self.max_age or age < -self.future_skew:
            self.last_rejection_reason = "control event is stale or from the future"
            return "rejected"
        self._last_control_sequence = sequence

        if control == "pause":
            if self.actions_paused:
                return "rejected"
            self.actions_paused = True
            self._paused_since = stamp
            if self.pause_gesture_id == self.resume_gesture_id:
                self._shared_control_armed = False
            return "pause"
        if control == "resume":
            if not self.actions_paused:
                return "rejected"
            if (self._paused_since is not None
                    and current.astimezone(timezone.utc) - self._paused_since < self.minimum_pause):
                self.last_rejection_reason = "resume is held for one second after Pause"
                return "rejected"
            self.actions_paused = False
            self._paused_since = None
            if self.pause_gesture_id == self.resume_gesture_id:
                self._shared_control_armed = False
            return "resume"
        self.actions_paused = True
        return "stop"
