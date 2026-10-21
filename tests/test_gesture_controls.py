from datetime import datetime, timedelta, timezone
import unittest
from uuid import uuid4

from gesture_manager.actions import ActionDispatcher, ActionMapping
from gesture_manager.gesture_controls import GestureControlRouter
from gesture_manager.recognition import RecognitionEvent


class MockKeyboardBackend:
    def __init__(self):
        self.sent = []

    def send_shortcut(self, shortcut):
        self.sent.append(shortcut)


def make_event(gesture_id, *, eligible=True, sequence=1, timestamp=None,
               score=0.8, reason="stable_prediction"):
    stamp = timestamp or datetime.now(timezone.utc)
    if isinstance(stamp, datetime):
        stamp = stamp.isoformat()
    return RecognitionEvent(
        gesture_id=gesture_id, prediction=gesture_id, decision_score=score,
        score_kind="svc_decision_margin",
        timestamp=stamp,
        trigger_eligible=eligible, reason=reason, frame_sequence=sequence,
    )


class GestureControlRouterTests(unittest.TestCase):
    def setUp(self):
        self.pause_id, self.resume_id, self.stop_id, self.action_id = (
            str(uuid4()) for _ in range(4))
        self.router = GestureControlRouter()
        self.router.configure(pause_gesture_id=self.pause_id,
                              resume_gesture_id=self.resume_id,
                              stop_gesture_id=self.stop_id)
        self.backend = MockKeyboardBackend()
        self.dispatcher = ActionDispatcher(self.backend, cooldown_seconds=0,
                                           global_cooldown_seconds=0)
        self.mapping = ActionMapping(self.action_id, "Action", "keyboard_shortcut",
                                     "Ctrl+C", True)
        self.mappings = {self.action_id: self.mapping}

    def activate_real_input(self):
        self.dispatcher.set_backend(self.backend)
        self.dispatcher.confirm_real_input(confirmed=True)

    def test_ordinary_actions_run_active_block_paused_resume_restores(self):
        self.activate_real_input()
        now = datetime.now(timezone.utc)
        ordinary = make_event(self.action_id)
        self.assertEqual(self.router.dispatch_ordinary(
            ordinary, self.dispatcher, self.mappings).status, "sent")
        self.assertEqual(self.router.route(make_event(self.pause_id, sequence=2, timestamp=now), now=now), "pause")
        self.assertIsNone(self.router.dispatch_ordinary(
            make_event(self.action_id, sequence=3), self.dispatcher, self.mappings))
        resumed_at = now + timedelta(seconds=1)
        self.assertEqual(self.router.route(make_event(self.resume_id, sequence=4, timestamp=resumed_at),
                                           now=resumed_at), "resume")
        self.assertEqual(self.router.dispatch_ordinary(
            make_event(self.action_id, sequence=5), self.dispatcher, self.mappings).status,
            "sent")
        self.assertEqual(len(self.backend.sent), 2)

    def test_controls_never_fall_through_to_ordinary_mapping(self):
        self.activate_real_input()
        for control_id, sequence in ((self.pause_id, 1), (self.resume_id, 2),
                                     (self.stop_id, 3)):
            event = make_event(control_id, sequence=sequence)
            self.router.route(event)
            self.assertIsNone(self.router.dispatch_ordinary(
                event, self.dispatcher, self.mappings))
        self.assertEqual(self.backend.sent, [])

    def test_ambiguous_stale_invalid_and_duplicate_control_events_are_rejected(self):
        now = datetime.now(timezone.utc)
        cases = (
            make_event(self.pause_id, eligible=False, sequence=1, timestamp=now),
            make_event(self.pause_id, sequence=1, timestamp=now - timedelta(seconds=5)),
            make_event(self.pause_id, sequence=1, timestamp=now, score=float("nan")),
            make_event(self.pause_id, sequence=None, timestamp=now),
        )
        for event in cases:
            with self.subTest(event=event):
                self.assertEqual(self.router.route(event, now=now), "rejected")
                self.assertFalse(self.router.actions_paused)
        accepted = make_event(self.pause_id, sequence=10, timestamp=now)
        self.assertEqual(self.router.route(accepted, now=now), "pause")
        self.assertEqual(self.router.route(accepted, now=now), "rejected")
        self.assertEqual(self.router.route(make_event(self.pause_id, sequence=11,
                                                      timestamp=now), now=now), "rejected")
        self.assertTrue(self.router.actions_paused)

    def test_stale_or_invalid_ordinary_event_cannot_dispatch(self):
        self.activate_real_input()
        now = datetime.now(timezone.utc)
        for event in (
            make_event(self.action_id, timestamp=now - timedelta(seconds=5)),
            make_event(self.action_id, eligible=False, timestamp=now),
            make_event(self.action_id, score=float("inf"), timestamp=now),
            make_event(self.action_id, timestamp="bad timestamp"),
        ):
            with self.subTest(event=event):
                self.assertIsNone(self.router.dispatch_ordinary(
                    event, self.dispatcher, self.mappings))
        self.assertEqual(self.backend.sent, [])

    def test_holding_control_does_not_toggle_repeatedly(self):
        now = datetime.now(timezone.utc)
        self.assertEqual(self.router.route(make_event(self.pause_id, sequence=1,
                                                      timestamp=now), now=now), "pause")
        self.assertEqual(self.router.route(make_event(self.pause_id, sequence=2,
                                                      timestamp=now), now=now), "rejected")
        self.assertTrue(self.router.actions_paused)
        too_soon = now + timedelta(milliseconds=500)
        self.assertEqual(self.router.route(make_event(self.resume_id, sequence=3,
                                                      timestamp=too_soon), now=too_soon), "rejected")
        resumed_at = now + timedelta(seconds=1)
        self.assertEqual(self.router.route(make_event(self.resume_id, sequence=4,
                                                      timestamp=resumed_at), now=resumed_at), "resume")
        self.assertEqual(self.router.route(make_event(self.resume_id, sequence=5,
                                                      timestamp=resumed_at), now=resumed_at), "rejected")
        self.assertFalse(self.router.actions_paused)

    def test_same_pause_resume_gesture_toggles_and_cannot_be_stop(self):
        now = datetime.now(timezone.utc)
        self.router.configure(pause_gesture_id=self.pause_id,
                              resume_gesture_id=self.pause_id,
                              stop_gesture_id=None)
        self.assertEqual(self.router.route(make_event(self.pause_id, sequence=1, timestamp=now), now=now), "pause")
        self.assertEqual(self.router.route(make_event(self.pause_id, sequence=2,
                                                      timestamp=now), now=now), "rejected")
        self.assertTrue(self.router.actions_paused)
        released_at = now + timedelta(seconds=1)
        self.assertIsNone(self.router.route(make_event(self.action_id, sequence=3,
                                                      timestamp=released_at), now=released_at))
        self.assertEqual(self.router.route(make_event(self.pause_id, sequence=4,
                                                      timestamp=released_at), now=released_at), "resume")
        self.assertFalse(self.router.actions_paused)
        with self.assertRaisesRegex(ValueError, "Stop Gesture cannot use"):
            self.router.configure(pause_gesture_id=self.pause_id,
                                  resume_gesture_id=self.pause_id,
                                  stop_gesture_id=self.pause_id)

    def test_emergency_stop_blocks_actions_in_active_and_paused_modes(self):
        self.activate_real_input()
        self.dispatcher.emergency_stop()
        ordinary = make_event(self.action_id)
        self.assertEqual(self.router.dispatch_ordinary(
            ordinary, self.dispatcher, self.mappings).status, "emergency_stopped")
        self.router.set_actions_paused(True)
        self.assertIsNone(self.router.dispatch_ordinary(
            ordinary, self.dispatcher, self.mappings))
        resumed_at = datetime.now(timezone.utc) + timedelta(seconds=1)
        self.assertEqual(self.router.route(make_event(self.resume_id, sequence=2,
                                                      timestamp=resumed_at), now=resumed_at), "resume")
        self.assertEqual(self.router.dispatch_ordinary(
            make_event(self.action_id, sequence=3), self.dispatcher, self.mappings).status,
            "emergency_stopped")
        self.assertEqual(self.backend.sent, [])

    def test_dry_run_stays_default_and_real_input_requires_confirmation(self):
        ordinary = make_event(self.action_id)
        result = self.router.dispatch_ordinary(ordinary, self.dispatcher, self.mappings)
        self.assertEqual(result.status, "dry_run")
        now = datetime.now(timezone.utc)
        self.assertEqual(self.router.route(make_event(self.pause_id, sequence=2,
                                                      timestamp=now), now=now), "pause")
        resumed_at = now + timedelta(seconds=1)
        self.assertEqual(self.router.route(make_event(self.resume_id, sequence=3,
                                                      timestamp=resumed_at), now=resumed_at), "resume")
        self.assertEqual(self.backend.sent, [])


if __name__ == "__main__":
    unittest.main()
