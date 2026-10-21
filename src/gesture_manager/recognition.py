"""Safety gate and diagnostic event worker for local custom recognition."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from queue import Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any, Callable, Mapping

from gesture_manager.gesture_data import FEATURES_PER_HAND, normalize_hand_features
from gesture_manager.gesture_training import (
    ClassifierPrediction,
    GestureLabel,
    PortableGestureModel,
)


UNCERTAIN_PREDICTION = "Uncertain — no gesture detected"
NO_HAND_PREDICTION = "No hand detected"


@dataclass(frozen=True)
class RecognitionEvent:
    gesture_id: str | None
    prediction: str
    decision_score: float | None
    score_kind: str
    timestamp: str
    trigger_eligible: bool
    reason: str
    frame_sequence: int | None = None


class RecognitionGate:
    """Turn frame predictions into debounced, release-gated diagnostic events.

    Per-label thresholds apply to the selected SVC's raw aggregated one-vs-one
    decision-margin difference. This is an uncalibrated model score, not a
    probability. It must be tuned against held-out user samples.
    """

    def __init__(
        self,
        labels: tuple[GestureLabel, ...],
        *,
        enabled: Mapping[str, bool] | None = None,
        minimum_margins: Mapping[str, float] | None = None,
        activation_frames: int = 3,
        cooldown_seconds: float = 1.0,
        cooldowns: Mapping[str, float] | None = None,
        release_frames: int = 2,
    ) -> None:
        if activation_frames < 1:
            raise ValueError("activation_frames must be at least one")
        if release_frames < 1:
            raise ValueError("release_frames must be at least one")
        if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be finite and non-negative")
        self.labels = {label.dataset_id: label for label in labels}
        self.enabled = {key: True for key in self.labels}
        self.enabled.update(enabled or {})
        self.minimum_margins = {key: 0.0 for key in self.labels}
        self.minimum_margins.update(minimum_margins or {})
        if set(self.enabled) - self.labels.keys() or set(self.minimum_margins) - self.labels.keys():
            raise ValueError("Recognition settings contain an unknown gesture ID")
        if any(not math.isfinite(value) or value < 0 for value in self.minimum_margins.values()):
            raise ValueError("SVC decision thresholds must be finite and non-negative")
        self.activation_frames = activation_frames
        self.cooldowns = {key: cooldown_seconds for key in self.labels}
        self.cooldowns.update(cooldowns or {})
        if (set(self.cooldowns) - self.labels.keys()
                or any(not math.isfinite(value) or value < 0 for value in self.cooldowns.values())):
            raise ValueError("Per-gesture cooldown settings are invalid")
        self.cooldown_seconds = cooldown_seconds
        self.release_frames = release_frames
        self._lock = Lock()
        self._paused = False
        self._emergency_stopped = False
        self._candidate_id: str | None = None
        self._candidate_frames = 0
        self._last_sequence: int | None = None
        self._last_trigger: dict[str, float] = {}
        self._needs_release = False
        self._no_hand_frames = 0

    def update_gesture(self, gesture_id: str, *, enabled: bool | None = None,
                       minimum_margin: float | None = None,
                       cooldown_seconds: float | None = None) -> None:
        with self._lock:
            if gesture_id not in self.labels:
                raise ValueError("Unknown gesture ID")
            if enabled is not None:
                self.enabled[gesture_id] = enabled
            if minimum_margin is not None:
                if not math.isfinite(minimum_margin) or minimum_margin < 0:
                    raise ValueError("SVC decision threshold must be finite and non-negative")
                self.minimum_margins[gesture_id] = minimum_margin
            if cooldown_seconds is not None:
                if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
                    raise ValueError("Cooldown must be finite and non-negative")
                self.cooldowns[gesture_id] = cooldown_seconds
            if not self.enabled[gesture_id] and self._candidate_id == gesture_id:
                self._reset_candidate()

    @property
    def enabled_gesture_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(key for key, is_enabled in self.enabled.items() if is_enabled)

    @property
    def paused(self) -> bool:
        """Whether the gate can safely sleep without polling frame snapshots."""
        with self._lock:
            return self._paused or self._emergency_stopped

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = paused
            self._reset_candidate()
            if paused:
                # Require a neutral release after resuming, preventing a held
                # pose from becoming a fresh activation by itself.
                self._needs_release = True
                self._no_hand_frames = 0

    def emergency_stop(self) -> None:
        with self._lock:
            self._emergency_stopped = True
            self._reset_candidate()

    def process(self, prediction: ClassifierPrediction | None, *,
                hands_present: bool, frame_sequence: int | None = None,
                now: float | None = None,
                uncertain_reason: str = "ambiguous_or_invalid_prediction") -> RecognitionEvent:
        now = monotonic() if now is None else now
        stamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if self._emergency_stopped:
                return self._event(None, "Recognition emergency-stopped", None,
                                   False, "emergency_stop", frame_sequence, stamp)
            if self._paused:
                return self._event(None, "Recognition paused", None, False,
                                   "paused", frame_sequence, stamp)
            if (frame_sequence is not None and self._last_sequence is not None
                    and frame_sequence != self._last_sequence + 1):
                self._reset_candidate()
            if frame_sequence is not None:
                self._last_sequence = frame_sequence

            if not hands_present:
                self._reset_candidate()
                self._no_hand_frames += 1
                if self._needs_release and self._no_hand_frames >= self.release_frames:
                    self._needs_release = False
                    reason = "neutral_release"
                else:
                    reason = "no_hand"
                return self._event(None, NO_HAND_PREDICTION, None, False,
                                   reason, frame_sequence, stamp)

            self._no_hand_frames = 0
            if prediction is None:
                self._reset_candidate()
                return self._event(None, UNCERTAIN_PREDICTION, None, False,
                                   uncertain_reason, frame_sequence, stamp)

            label = prediction.label
            if label.dataset_id not in self.labels:
                self._reset_candidate()
                return self._event(None, UNCERTAIN_PREDICTION, prediction.decision_margin,
                                   False, "unknown_model_label", frame_sequence, stamp)
            if prediction.vote_tie or prediction.decision_margin <= self.minimum_margins[label.dataset_id]:
                self._reset_candidate()
                return self._event(label.dataset_id, UNCERTAIN_PREDICTION,
                                   prediction.decision_margin, False,
                                   "ambiguous_or_below_threshold", frame_sequence, stamp)
            if not self.enabled[label.dataset_id]:
                self._reset_candidate()
                return self._event(None, UNCERTAIN_PREDICTION, prediction.decision_margin,
                                   False, "gesture_disabled", frame_sequence, stamp)
            if self._needs_release:
                self._reset_candidate()
                return self._event(label.dataset_id, label.name, prediction.decision_margin,
                                   False, "awaiting_neutral_release", frame_sequence, stamp)

            if self._candidate_id != label.dataset_id:
                self._candidate_id = label.dataset_id
                self._candidate_frames = 1
            else:
                self._candidate_frames += 1
            if self._candidate_frames < self.activation_frames:
                return self._event(label.dataset_id, label.name, prediction.decision_margin,
                                   False, "activation_pending", frame_sequence, stamp)

            last_trigger = self._last_trigger.get(label.dataset_id)
            if last_trigger is not None and now - last_trigger < self.cooldowns[label.dataset_id]:
                return self._event(label.dataset_id, label.name, prediction.decision_margin,
                                   False, "cooldown", frame_sequence, stamp)

            self._last_trigger[label.dataset_id] = now
            self._needs_release = True
            self._no_hand_frames = 0
            self._reset_candidate()
            return self._event(label.dataset_id, label.name, prediction.decision_margin,
                               True, "stable_prediction", frame_sequence, stamp)

    def _reset_candidate(self) -> None:
        self._candidate_id = None
        self._candidate_frames = 0

    @staticmethod
    def _event(gesture_id: str | None, prediction: str, score: float | None,
               eligible: bool, reason: str, sequence: int | None,
               stamp: str) -> RecognitionEvent:
        return RecognitionEvent(gesture_id, prediction, score,
                                "svc_ovo_decision_margin", stamp,
                                eligible, reason, sequence)


class LiveRecognitionWorker:
    """Classify fresh hand-engine snapshots and enqueue diagnostic events only."""

    def __init__(self, model: PortableGestureModel, status_provider: Callable[[], Any],
                 output_queue: Queue[RecognitionEvent], *,
                 enabled: Mapping[str, bool] | None = None,
                 minimum_margins: Mapping[str, float] | None = None,
                 activation_frames: int = 3, cooldown_seconds: float = 1.0,
                 cooldowns: Mapping[str, float] | None = None,
                 release_frames: int = 2) -> None:
        self.model = model
        self.status_provider = status_provider
        self.output_queue = output_queue
        self.gate = RecognitionGate(model.labels, enabled=enabled,
                                    minimum_margins=minimum_margins,
                                    activation_frames=activation_frames,
                                    cooldown_seconds=cooldown_seconds,
                                    cooldowns=cooldowns,
                                    release_frames=release_frames)
        self._stop = Event()
        self._thread: Thread | None = None
        self._last_sequence = getattr(status_provider(), "processed_frames", -1)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        if self.is_alive:
            return False
        self._stop.clear()
        self._thread = Thread(target=self._run, name="GestureManagerLiveRecognition", daemon=False)
        self._thread.start()
        return True

    def pause(self) -> None:
        self.gate.set_paused(True)

    def resume(self) -> None:
        self.gate.set_paused(False)

    def emergency_stop(self) -> None:
        self.gate.emergency_stop()
        self.stop()

    def stop(self) -> None:
        self._stop.set()

    def wait(self, timeout: float | None = None) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _publish(self, event: RecognitionEvent) -> None:
        try:
            self.output_queue.put_nowait(event)
        except Full:
            try:
                self.output_queue.get_nowait()
            except Exception:
                pass
            try:
                self.output_queue.put_nowait(event)
            except Full:
                pass

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            if self.gate.paused:
                self._stop.wait(0.5)
                continue
            status = self.status_provider()
            sequence = getattr(status, "processed_frames", -1)
            if sequence <= self._last_sequence:
                continue
            self._last_sequence = sequence
            if getattr(status, "hands_detected", None) is not True:
                self._publish(self.gate.process(None, hands_present=False,
                                                frame_sequence=sequence))
                continue
            hands = getattr(status, "hands", ())
            if len(hands) * FEATURES_PER_HAND != self.model.feature_size:
                self._publish(self.gate.process(None, hands_present=True,
                                                frame_sequence=sequence))
                continue
            try:
                features = normalize_hand_features(tuple(hands))
                enabled_ids = self.gate.enabled_gesture_ids
                if len(enabled_ids) < 2:
                    event = self.gate.process(
                        None, hands_present=True, frame_sequence=sequence,
                        uncertain_reason="fewer_than_two_gestures_enabled")
                    self._publish(event)
                    continue
                prediction = self.model.predict_with_scores(
                    features, enabled_label_ids=enabled_ids)
                event = self.gate.process(prediction, hands_present=True,
                                          frame_sequence=sequence)
            except Exception:
                event = self.gate.process(None, hands_present=True,
                                          frame_sequence=sequence)
            self._publish(event)
