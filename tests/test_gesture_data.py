from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from uuid import uuid4

from gesture_manager.gesture_data import (
    FEATURE_VERSION,
    GestureRecordingSession,
    GestureStore,
    TrainingExample,
    create_draft,
    dataset_to_dict,
    normalize_hand_features,
)
from gesture_manager.hand_landmarks import DetectedHand, NormalizedLandmark


def sample_hand(*, shift=(0.0, 0.0), scale=1.0, angle=0.0) -> DetectedHand:
    import math

    points = []
    for index in range(21):
        x = (0.4 + (index % 5) * 0.012) * scale
        y = (0.5 + (index // 5) * 0.015) * scale
        x, y = x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle)
        points.append(NormalizedLandmark(x + shift[0], y + shift[1], (index % 3) * 0.001 * scale))
    return DetectedHand(tuple(points), 0.9)


class GestureDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = GestureStore(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_create_record_save_and_load_feature_only_dataset(self) -> None:
        dataset = create_draft("My own pose", 1)
        session = GestureRecordingSession(dataset)
        session.add_hands((sample_hand(),))
        self.store.save(session.dataset)
        loaded = self.store.load(dataset.id)
        self.assertEqual(loaded.name, "My own pose")
        self.assertEqual(loaded.feature_version, FEATURE_VERSION)
        self.assertEqual(len(loaded.examples), 1)
        self.assertEqual(len(loaded.examples[0].features), 63)
        payload = self.store.path_for(dataset.id).read_text(encoding="utf-8")
        self.assertNotIn("frame", payload.lower())
        self.assertNotIn("video", payload.lower())

    def test_features_reduce_position_scale_and_rotation_sensitivity(self) -> None:
        baseline = normalize_hand_features((sample_hand(),))
        transformed = normalize_hand_features((sample_hand(shift=(0.2, -0.1), scale=1.7, angle=0.4),))
        for expected, actual in zip(baseline, transformed, strict=True):
            self.assertAlmostEqual(expected, actual, places=6)

    def test_remove_example_and_safe_delete_remove_local_dataset(self) -> None:
        dataset = create_draft("Delete me", 1)
        session = GestureRecordingSession(dataset)
        first = session.add_hands((sample_hand(),))
        session.add_hands((sample_hand(shift=(0.02, 0)),))
        self.store.save(session.dataset)
        updated = self.store.remove_example(dataset.id, first.id)
        self.assertEqual(len(updated.examples), 1)
        path = self.store.path_for(dataset.id)
        self.assertTrue(path.is_file())
        self.store.delete(dataset.id)
        self.assertFalse(path.exists())
        with self.assertRaises(FileNotFoundError):
            self.store.load(dataset.id)

    def test_rename_and_duplicate_name_validation(self) -> None:
        dataset = create_draft("First", 1)
        self.store.save(dataset)
        renamed = self.store.rename(dataset.id, "Renamed")
        self.assertEqual(renamed.name, "Renamed")
        duplicate = create_draft("renamed", 2)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.save(duplicate)

    def test_invalid_data_and_unsupported_versions_are_rejected(self) -> None:
        dataset = create_draft("Bad", 1)
        session = GestureRecordingSession(dataset)
        valid = session.add_hands((sample_hand(),))
        bad = TrainingExample(valid.id, valid.captured_at, (float("nan"),))
        with self.assertRaises(ValueError):
            self.store.save(dataset.__class__(**{**session.dataset.__dict__, "examples": (bad,)}))
        valid_but_nan = replace(session.dataset, examples=(replace(valid, features=(float("nan"),) * 63),))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.store.save(valid_but_nan)
        payload = dataset_to_dict(dataset)
        payload["format_version"] = 99
        with self.assertRaisesRegex(ValueError, "Unsupported dataset version"):
            from gesture_manager.gesture_data import dataset_from_dict
            dataset_from_dict(payload)
        with self.assertRaises(ValueError):
            create_draft("  ", 1)

    def test_json_dataset_does_not_coerce_boolean_features_into_numbers(self) -> None:
        from gesture_manager.gesture_data import dataset_from_dict

        session = GestureRecordingSession(create_draft("Strict types", 1))
        session.add_hands((sample_hand(),))
        payload = dataset_to_dict(session.dataset)
        payload["examples"][0]["features"][0] = True
        with self.assertRaisesRegex(ValueError, "field types"):
            dataset_from_dict(payload)

    def test_dataset_store_rejects_symlink_that_points_outside_storage(self) -> None:
        dataset_id = str(uuid4())
        outside = Path(self.temp.name).parent / f"{dataset_id}.json"
        link = self.store.path_for(dataset_id)
        outside.write_text("{}", encoding="utf-8")
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            outside.unlink(missing_ok=True)
            self.skipTest(f"Symbolic links are unavailable: {exc}")
        try:
            with self.assertRaisesRegex(ValueError, "inside its storage"):
                self.store.load(dataset_id)
        finally:
            link.unlink(missing_ok=True)
            outside.unlink(missing_ok=True)

    def test_training_readiness_requires_three_valid_examples(self) -> None:
        dataset = create_draft("Readiness", 1)
        session = GestureRecordingSession(dataset)
        session.add_hands((sample_hand(),))
        self.assertFalse(session.dataset.is_trainable)
        session.add_hands((sample_hand(shift=(0.01, 0)),))
        session.add_hands((sample_hand(shift=(0.02, 0)),))
        self.assertTrue(session.dataset.is_trainable)

    def test_interrupted_recording_is_not_persisted(self) -> None:
        dataset = create_draft("Interrupted", 1)
        session = GestureRecordingSession(dataset)
        session.add_hands((sample_hand(),))
        session.cancel()
        self.assertEqual(session.dataset.examples, ())
        self.assertFalse(self.store.path_for(dataset.id).exists())

    def test_interrupted_additional_take_preserves_saved_dataset(self) -> None:
        dataset = create_draft("Existing", 1)
        original = GestureRecordingSession(dataset)
        original.add_hands((sample_hand(),))
        self.store.save(original.dataset)
        take = GestureRecordingSession(original.dataset)
        take.add_hands((sample_hand(shift=(0.02, 0)),))
        take.cancel()
        self.assertEqual(take.dataset, original.dataset)
        self.assertEqual(self.store.load(dataset.id), original.dataset)

    def test_delete_rejects_non_uuid_path(self) -> None:
        with self.assertRaises(ValueError):
            self.store.delete("../../outside")

    def test_recording_rejects_wrong_hand_count_and_degenerate_points(self) -> None:
        session = GestureRecordingSession(create_draft("Validation", 1))
        with self.assertRaisesRegex(ValueError, "requires exactly"):
            session.add_hands(())
        degenerate = DetectedHand(tuple(NormalizedLandmark(0, 0, 0) for _ in range(21)), 1.0)
        with self.assertRaisesRegex(ValueError, "too close"):
            session.add_hands((degenerate,))


if __name__ == "__main__":
    unittest.main()
