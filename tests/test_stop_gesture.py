from datetime import datetime, timedelta, timezone
import tempfile
from pathlib import Path
import unittest

from gesture_manager.recognition import RecognitionEvent
from gesture_manager.runtime import RuntimeController, RuntimeState
from gesture_manager.stop_gesture import (
    StopGestureStore, StopGestureTrigger, describe_stop_candidate,
)


def event(*, gesture_id="custom-a", eligible=True, reason="stable_prediction",
          score=0.8, timestamp=None, sequence=10):
    return RecognitionEvent(
        gesture_id=gesture_id,
        prediction="Custom A",
        decision_score=score,
        score_kind="svc_decision_margin",
        timestamp=(timestamp if isinstance(timestamp, str) else
                   (timestamp or datetime.now(timezone.utc)).isoformat()),
        trigger_eligible=eligible,
        reason=reason,
        frame_sequence=sequence,
    )


class StopGestureStoreTests(unittest.TestCase):
    def test_persists_and_loads_profile_local_gesture_id(self):
        with tempfile.TemporaryDirectory() as temp:
            store = StopGestureStore(Path(temp) / "settings" / "stop.json")
            self.assertIsNone(store.load())
            store.save("gesture-123")
            self.assertEqual(store.load(), "gesture-123")
            store.save(None)
            self.assertIsNone(store.load())

    def test_rejects_corrupt_or_unsupported_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stop.json"
            path.write_text('{"format":"wrong","version":1,"gesture_id":null}', encoding="utf-8")
            with self.assertRaises(ValueError):
                StopGestureStore(path).load()
            with self.assertRaises(ValueError):
                StopGestureStore(path).save(" ")


class StopGestureTriggerTests(unittest.TestCase):
    def test_activation_pending_is_reported_as_confirmation_not_rejection(self):
        message = describe_stop_candidate(
            event(eligible=False, reason="activation_pending"), activation_frames=4)
        self.assertIn("confirming a stable pose", message)
        self.assertIn("4 consecutive fresh frames", message)
        self.assertNotIn("rejected", message)

    def test_low_confidence_candidate_explains_gate_rejection(self):
        message = describe_stop_candidate(
            event(eligible=False, reason="ambiguous_or_below_threshold"), activation_frames=3)
        self.assertIn("uncertain or below its configured score threshold", message)
        self.assertIn("no action dispatched", message)

    def test_only_stable_fresh_configured_event_triggers_once(self):
        trigger = StopGestureTrigger()
        now = datetime.now(timezone.utc)
        self.assertTrue(trigger.consume(event(timestamp=now), "custom-a", now=now))
        self.assertFalse(trigger.consume(event(timestamp=now), "custom-a", now=now))
        self.assertFalse(trigger.consume(event(sequence=11, timestamp=now), None, now=now))

    def test_uncertain_low_confidence_stale_future_invalid_and_wrong_gesture_rejected(self):
        now = datetime.now(timezone.utc)
        trigger = StopGestureTrigger()
        rejected = (
            event(eligible=False, reason="ambiguous_or_invalid_prediction", timestamp=now),
            event(eligible=False, score=0.0, timestamp=now),
            event(timestamp=now - timedelta(seconds=4)),
            event(timestamp=now + timedelta(seconds=2)),
            event(timestamp="not-a-date"),
            event(sequence=None, timestamp=now),
            event(gesture_id="other", timestamp=now),
            event(score=float("nan"), timestamp=now),
        )
        for item in rejected:
            with self.subTest(item=item):
                self.assertFalse(trigger.consume(item, "custom-a", now=now))

    def test_stop_callback_stops_runtime_and_same_event_is_consumed_before_mapping(self):
        stopped = []
        controller = RuntimeController(stop_action=lambda: stopped.append("camera released"))
        controller.start()
        trigger = StopGestureTrigger()
        stop_event = event()
        handled = trigger.consume(stop_event, "custom-a")
        dispatched = []
        if handled:
            controller.stop()
        else:
            dispatched.append(stop_event)
        self.assertEqual(controller.state, RuntimeState.STOPPED)
        self.assertEqual(stopped, ["camera released"])
        self.assertEqual(dispatched, [])
        self.assertFalse(trigger.consume(stop_event, "custom-a"))
        self.assertTrue(controller.start())  # explicit/manual restart path


if __name__ == "__main__":
    unittest.main()
