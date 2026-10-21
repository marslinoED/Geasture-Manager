from __future__ import annotations

import unittest
from uuid import uuid4

from gesture_manager.gesture_training import ClassifierPrediction, GestureLabel
from gesture_manager.recognition import (
    NO_HAND_PREDICTION,
    UNCERTAIN_PREDICTION,
    RecognitionGate,
)


class RecognitionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = GestureLabel(str(uuid4()), "custom wave")
        self.second = GestureLabel(str(uuid4()), "custom pinch")
        self.gate = RecognitionGate((self.first, self.second), activation_frames=3,
                                    cooldown_seconds=1.0, release_frames=2)

    def stable_prediction(self, label: GestureLabel | None = None,
                          margin: float = 0.7) -> ClassifierPrediction:
        return ClassifierPrediction(label or self.first, margin, 1.0, False)

    def frame(self, seq: int, now: float, prediction: ClassifierPrediction | None = None,
              *, hands: bool = True):
        return self.gate.process(prediction or self.stable_prediction(),
                                 hands_present=hands, frame_sequence=seq, now=now)

    def test_requires_consecutive_fresh_frames_before_activation(self) -> None:
        self.assertEqual(self.frame(1, 0).reason, "activation_pending")
        self.assertEqual(self.frame(2, .1).reason, "activation_pending")
        event = self.frame(3, .2)
        self.assertTrue(event.trigger_eligible)
        self.assertEqual(event.gesture_id, self.first.dataset_id)
        self.assertEqual(event.prediction, self.first.name)
        self.assertIsNotNone(event.timestamp)
        self.assertEqual(event.score_kind, "svc_ovo_decision_margin")

    def test_repeated_held_detection_needs_neutral_release(self) -> None:
        self.frame(1, 0)
        self.frame(2, .1)
        self.assertTrue(self.frame(3, .2).trigger_eligible)
        self.frame(4, .3)
        self.frame(5, .4)
        event = self.frame(6, .5)
        self.assertFalse(event.trigger_eligible)
        self.assertEqual(event.reason, "awaiting_neutral_release")
        self.assertEqual(event.prediction, self.first.name)

    def test_cooldown_applies_after_neutral_release(self) -> None:
        self.gate = RecognitionGate((self.first, self.second), activation_frames=2,
                                    cooldown_seconds=1.0, release_frames=2)
        self.frame(1, 0)
        self.assertTrue(self.frame(2, .1).trigger_eligible)
        self.assertEqual(self.frame(3, .2, hands=False).prediction, NO_HAND_PREDICTION)
        self.assertEqual(self.frame(4, .3, hands=False).reason, "neutral_release")
        self.frame(5, .4)
        event = self.frame(6, .5)
        self.assertFalse(event.trigger_eligible)
        self.assertEqual(event.reason, "cooldown")
        self.assertTrue(self.frame(7, 1.2).trigger_eligible)

    def test_profile_can_configure_cooldown_per_gesture(self) -> None:
        self.gate = RecognitionGate((self.first, self.second), activation_frames=1,
                                    cooldown_seconds=5.0, cooldowns={self.first.dataset_id: 0.0},
                                    release_frames=1)
        self.assertTrue(self.frame(1, 0).trigger_eligible)
        self.frame(2, .1, hands=False)
        self.assertTrue(self.frame(3, .2).trigger_eligible)
        self.frame(4, .3, hands=False)
        self.assertTrue(self.frame(5, .4, self.stable_prediction(self.second)).trigger_eligible)
        self.frame(6, .5, hands=False)
        self.assertEqual(self.frame(7, .6, self.stable_prediction(self.second)).reason, "cooldown")

    def test_ambiguous_or_low_margin_prediction_is_uncertain(self) -> None:
        self.gate.update_gesture(self.first.dataset_id, minimum_margin=.5)
        low = self.frame(1, 0, self.stable_prediction(margin=.4))
        self.assertEqual(low.prediction, UNCERTAIN_PREDICTION)
        self.assertEqual(low.reason, "ambiguous_or_below_threshold")
        tied = ClassifierPrediction(self.first, .9, .5, True)
        event = self.frame(2, .1, tied)
        self.assertEqual(event.prediction, UNCERTAIN_PREDICTION)
        self.assertFalse(event.trigger_eligible)

    def test_disabled_gesture_never_becomes_trigger_eligible(self) -> None:
        self.gate.update_gesture(self.first.dataset_id, enabled=False)
        event = self.frame(1, 0)
        self.assertEqual(event.prediction, UNCERTAIN_PREDICTION)
        self.assertIsNone(event.gesture_id)
        self.assertEqual(event.reason, "gesture_disabled")
        self.assertFalse(event.trigger_eligible)

    def test_enabled_gesture_ids_are_thread_safe_snapshot(self) -> None:
        self.assertEqual(self.gate.enabled_gesture_ids,
                         frozenset((self.first.dataset_id, self.second.dataset_id)))
        self.gate.update_gesture(self.second.dataset_id, enabled=False)
        self.assertEqual(self.gate.enabled_gesture_ids, frozenset((self.first.dataset_id,)))

    def test_paused_and_emergency_stopped_states_block_eligibility(self) -> None:
        self.gate.set_paused(True)
        event = self.frame(1, 0)
        self.assertEqual(event.reason, "paused")
        self.assertFalse(event.trigger_eligible)
        self.gate.set_paused(False)
        self.frame(2, .1)
        self.frame(3, .2)
        self.assertEqual(self.frame(4, .3).reason, "awaiting_neutral_release")
        self.gate.emergency_stop()
        stopped = self.frame(3, .2)
        self.assertEqual(stopped.reason, "emergency_stop")
        self.assertFalse(stopped.trigger_eligible)

    def test_sequence_gap_resets_activation_and_no_hand_releases(self) -> None:
        self.frame(1, 0)
        event = self.frame(3, .2)
        self.assertEqual(event.reason, "activation_pending")
        self.assertFalse(event.trigger_eligible)
        self.gate.emergency_stop()
        self.assertEqual(self.frame(4, .3, hands=False).reason, "emergency_stop")


if __name__ == "__main__":
    unittest.main()
