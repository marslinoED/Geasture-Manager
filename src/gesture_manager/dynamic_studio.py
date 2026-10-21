"""Small adapter from fresh hand-engine snapshots to bounded dynamic takes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gesture_manager.dynamic_gestures import DynamicGestureExample, DynamicRecordingSession


@dataclass(frozen=True)
class RecordingUpdate:
    frame_count: int
    example: DynamicGestureExample | None = None
    message: str = ""

    @property
    def added_frames(self) -> int:
        """Compatibility alias; value is the current take's frame count."""
        return self.frame_count


class DynamicStudioRecorder:
    """Consumes each processed engine snapshot at most once; owns no camera."""

    def __init__(self, session: DynamicRecordingSession, last_sequence: int = -1) -> None:
        self.session = session
        self.last_sequence = last_sequence
        self._frame_count = 0

    @property
    def recording(self) -> bool:
        return self.session.recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def begin(self) -> None:
        self.session.begin_example()
        self._frame_count = 0

    def consume(self, status: Any) -> RecordingUpdate:
        sequence = getattr(status, "processed_frames", -1)
        if not self.recording or sequence == self.last_sequence:
            return RecordingUpdate(self.frame_count)
        self.last_sequence = sequence
        if getattr(status, "hands_detected", False) is not True:
            return RecordingUpdate(self.frame_count, message="Waiting for exactly one visible hand.")
        try:
            self.session.add_frame(tuple(status.hands), frame_sequence=sequence)
        except ValueError as exc:
            return RecordingUpdate(self.frame_count, message=str(exc))
        self._frame_count += 1
        if self.frame_count == 60:
            return RecordingUpdate(60, self.finish(), "Sequence limit reached; take saved.")
        return RecordingUpdate(self.frame_count, message="Recording numerical landmarks locally.")

    def finish(self) -> DynamicGestureExample:
        example = self.session.finish_example()
        self._frame_count = 0
        return example

    def cancel(self) -> None:
        self.session.cancel_example()
        self._frame_count = 0
