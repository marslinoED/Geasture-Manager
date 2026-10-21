"""Persistent Stop Gesture selection and a fail-closed recognition gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tempfile


class StopGestureStore:
    """Store a static gesture ID separately from classifier assets and mappings."""

    FORMAT = "gesture-manager-stop-gesture"
    VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> str | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Stop Gesture settings could not be read: {exc}") from exc
        if (not isinstance(data, dict)
                or set(data) != {"format", "version", "gesture_id"}
                or data.get("format") != self.FORMAT
                or data.get("version") != self.VERSION):
            raise ValueError("Stop Gesture settings have an unsupported format")
        gesture_id = data["gesture_id"]
        if gesture_id is not None and (not isinstance(gesture_id, str)
                                       or not gesture_id.strip()
                                       or len(gesture_id) > 128):
            raise ValueError("Stop Gesture ID is invalid")
        return gesture_id

    def save(self, gesture_id: str | None) -> None:
        if gesture_id is not None and (not isinstance(gesture_id, str)
                                       or not gesture_id.strip()
                                       or len(gesture_id) > 128):
            raise ValueError("Stop Gesture ID is invalid")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"format": self.FORMAT, "version": self.VERSION,
                              "gesture_id": gesture_id}, indent=2) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=".stop-gesture-", suffix=".tmp",
                                         dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise


class StopGestureTrigger:
    """Accept only fresh, stable, eligible stop events and reject duplicates."""

    def __init__(self, *, max_age: float = 3.0, future_skew: float = 0.5) -> None:
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("max_age must be positive and finite")
        if not math.isfinite(future_skew) or future_skew < 0:
            raise ValueError("future_skew must be non-negative and finite")
        self.max_age = timedelta(seconds=max_age)
        self.future_skew = timedelta(seconds=future_skew)
        self._last_sequence: int | None = None
        self.last_rejection_reason = ""

    def consume(self, event: object, gesture_id: str | None, *,
                now: datetime | None = None) -> bool:
        self.last_rejection_reason = ""
        if not gesture_id or getattr(event, "gesture_id", None) != gesture_id:
            self.last_rejection_reason = "gesture ID did not match the configured Stop Gesture"
            return False
        if getattr(event, "trigger_eligible", False) is not True:
            self.last_rejection_reason = f"recognition gate rejected event ({getattr(event, 'reason', 'ineligible')})"
            return False
        score = getattr(event, "decision_score", None)
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or score <= 0):
            self.last_rejection_reason = "decision score was missing, invalid, or non-positive"
            return False
        sequence = getattr(event, "frame_sequence", None)
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            self.last_rejection_reason = "frame sequence was missing or invalid"
            return False
        if self._last_sequence is not None and sequence <= self._last_sequence:
            self.last_rejection_reason = "duplicate or out-of-order frame"
            return False
        try:
            stamp = datetime.fromisoformat(getattr(event, "timestamp", ""))
            if stamp.tzinfo is None:
                self.last_rejection_reason = "event timestamp has no timezone"
                return False
            stamp = stamp.astimezone(timezone.utc)
        except (TypeError, ValueError):
            self.last_rejection_reason = "event timestamp is invalid"
            return False
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        age = current.astimezone(timezone.utc) - stamp
        if age > self.max_age:
            self.last_rejection_reason = f"event was stale ({age.total_seconds():.1f}s old)"
            return False
        if age < -self.future_skew:
            self.last_rejection_reason = "event timestamp is too far in the future"
            return False
        self._last_sequence = sequence
        return True


def describe_stop_candidate(event: object, activation_frames: int) -> str:
    """Explain the recognition gate state without calling pending a rejection."""
    reason = getattr(event, "reason", "unknown")
    if reason == "activation_pending":
        return ("Stop Gesture detected · confirming a stable pose across "
                f"{activation_frames} consecutive fresh frames.")
    explanations = {
        "ambiguous_or_below_threshold": "prediction was uncertain or below its configured score threshold",
        "cooldown": "the gesture cooldown is still active",
        "awaiting_neutral_release": "release the pose before trying again",
        "gesture_disabled": "the gesture is disabled in recognition settings",
        "fewer_than_two_gestures_enabled": "enable at least two gestures for classification",
    }
    detail = explanations.get(reason, reason.replace("_", " "))
    return f"Stop Gesture not activated · {detail}; no action dispatched."
