from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

import numpy as np

from gesture_manager.dynamic_gestures import (
    DYNAMIC_FEATURE_VERSION,
    DynamicGestureDataset,
    DynamicGestureExample,
    DynamicGestureModelStore,
    DynamicGestureStore,
    DynamicGestureTrainer,
    DynamicRecordingSession,
    create_dynamic_draft,
    dtw_distance,
    dynamic_dataset_from_dict,
    dynamic_dataset_to_dict,
    dynamic_frame_features,
    validate_sequence,
)
from gesture_manager.hand_landmarks import DetectedHand, NormalizedLandmark


def sequence(kind: int, seed: int, frames: int = 16):
    rng = np.random.default_rng(seed)
    output = []
    for index in range(frames):
        t = index / (frames - 1)
        values = rng.normal(0, 0.04, 66)
        if kind == 0:
            values[63] += math_sin(2 * np.pi * t) * 1.2
            values[64] += 0.03 * np.sin(2 * np.pi * t)
        else:
            values[63] += 0.03 * np.sin(2 * np.pi * t)
            values[64] += math_sin(2 * np.pi * t) * 1.2
        output.append(tuple(float(value) for value in values))
    return tuple(output)


def math_sin(value: float) -> float:
    return float(np.sin(value))


def make_datasets(count: int = 10):
    datasets = []
    for label_index, name in enumerate(("Left-right wave", "Up-down wave")):
        draft = create_dynamic_draft(name)
        examples = tuple(DynamicGestureExample(
            str(uuid4()), datetime.now(timezone.utc).isoformat(),
            sequence(label_index, seed=label_index * 100 + index),
        ) for index in range(count))
        datasets.append(DynamicGestureDataset(draft.id, draft.name, examples))
    return tuple(datasets)


def shifted_hand(x_offset: float = 0.0) -> DetectedHand:
    points = tuple(NormalizedLandmark(
        x_offset + 0.4 + (index % 5) * 0.025,
        0.3 + (index // 5) * 0.035 + (index % 3) * 0.004,
        (index % 4) * 0.003,
    ) for index in range(21))
    return DetectedHand(points, 0.99)


class DynamicGestureTests(unittest.TestCase):
    def test_frame_features_preserve_path_while_normalizing_pose(self) -> None:
        original, origin, scale = dynamic_frame_features(shifted_hand())
        moved, _, _ = dynamic_frame_features(shifted_hand(0.1), origin=origin, reference_scale=scale)
        self.assertEqual(len(original), 66)
        self.assertAlmostEqual(original[63], 0.0)
        self.assertGreater(moved[63], 0.5)
        self.assertTrue(np.allclose(original[:63], moved[:63], atol=1e-6))

    def test_recording_bounds_takes_rejects_short_and_discards_interrupted_take(self) -> None:
        session = DynamicRecordingSession(create_dynamic_draft("Wave"))
        session.begin_example()
        for frame in range(7):
            session.add_frame((shifted_hand(frame * 0.005),), frame_sequence=frame)
        with self.assertRaisesRegex(ValueError, "at least 8"):
            session.finish_example()
        session.cancel_example()
        self.assertEqual(session.dataset.examples, ())
        session.begin_example()
        for frame in range(12):
            session.add_frame((shifted_hand(frame * 0.005),), frame_sequence=frame)
        before = len(session._recording)
        session.add_frame((shifted_hand(0.2),), frame_sequence=11)
        self.assertEqual(len(session._recording), before)
        session.finish_example()
        self.assertEqual(len(session.dataset.examples), 1)

    def test_dynamic_dataset_round_trip_and_invalid_sequence_rejection(self) -> None:
        draft = create_dynamic_draft("Wave")
        example = DynamicGestureExample(str(uuid4()), datetime.now(timezone.utc).isoformat(), sequence(0, 2))
        dataset = DynamicGestureDataset(draft.id, draft.name, (example,))
        self.assertEqual(dynamic_dataset_from_dict(dynamic_dataset_to_dict(dataset)), dataset)
        self.assertEqual(dataset.feature_version, DYNAMIC_FEATURE_VERSION)
        raw = dynamic_dataset_to_dict(dataset)
        raw["examples"][0]["frames"][0].pop()
        with self.assertRaisesRegex(ValueError, "malformed"):
            dynamic_dataset_from_dict(raw)
        with self.assertRaisesRegex(ValueError, "8 to 60"):
            validate_sequence(tuple(sequence(0, 4)[:7]))

    def test_dtw_classifier_evaluation_and_model_reload(self) -> None:
        datasets = make_datasets(10)
        trainer = DynamicGestureTrainer()
        model = trainer.train(datasets).with_thresholds(
            max_distance=0.0001, minimum_separation=0.04)
        self.assertTrue(model.evaluation.evaluated)
        self.assertTrue(model.evaluation.reliable)
        self.assertEqual(model.evaluation.accuracy, 1.0)
        for index, dataset in enumerate(datasets):
            prediction = model.predict(dataset.examples[0].frames,
                                       max_distance=0.2, minimum_separation=0.01)
            self.assertFalse(prediction.uncertain)
            self.assertEqual(prediction.label.dataset_id, dataset.id)
        with tempfile.TemporaryDirectory() as directory:
            store = DynamicGestureModelStore(Path(directory) / "dynamic-model.json")
            store.save(model)
            loaded = store.load()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.to_dict(), model.to_dict())
            self.assertEqual(loaded.max_distance_threshold, 0.0001)
            self.assertEqual(loaded.minimum_separation_threshold, 0.04)
            self.assertEqual(loaded.predict(datasets[0].examples[0].frames),
                             model.predict(datasets[0].examples[0].frames))
            self.assertTrue(loaded.predict(sequence(0, 999)).uncertain)

    def test_low_separation_and_distance_return_uncertain(self) -> None:
        datasets = make_datasets(6)
        model = DynamicGestureTrainer().train(datasets)
        sample = datasets[0].examples[0].frames
        novel_sample = sequence(0, 809)
        self.assertTrue(model.predict(novel_sample, max_distance=0.0001).uncertain)
        self.assertTrue(model.predict(sample, max_distance=10.0, minimum_separation=100.0).uncertain)

    def test_dataset_store_is_separate_and_delete_removes_data(self) -> None:
        dataset = make_datasets(3)[0]
        with tempfile.TemporaryDirectory() as directory:
            store = DynamicGestureStore(Path(directory) / "dynamic")
            store.save(dataset)
            self.assertEqual(store.list(), (dataset,))
            store.delete(dataset.id)
            self.assertEqual(store.list(), ())

    def test_dtw_handles_speed_variation(self) -> None:
        fast = sequence(0, 11, 12)
        slow = sequence(0, 12, 20)
        other = sequence(1, 13, 16)
        self.assertLess(dtw_distance(fast, slow), dtw_distance(fast, other))
        longest = sequence(0, 14, 60)
        self.assertTrue(np.isfinite(dtw_distance(fast, longest)))

    def test_dtw_rejects_empty_wrong_sized_and_non_finite_sequences(self) -> None:
        with self.assertRaisesRegex(ValueError, "8 to 60"):
            dtw_distance((), sequence(0, 15))
        wrong_size = tuple(tuple(frame[:-1]) for frame in sequence(0, 16))
        with self.assertRaisesRegex(ValueError, "malformed"):
            dtw_distance(wrong_size, sequence(0, 17))
        with self.assertRaisesRegex(ValueError, "malformed"):
            validate_sequence((None, *sequence(0, 17)[1:]))
        non_finite = list(sequence(0, 18))
        first = list(non_finite[0])
        first[0] = float("nan")
        non_finite[0] = tuple(first)
        with self.assertRaisesRegex(ValueError, "malformed"):
            validate_sequence(tuple(non_finite))

    def test_model_rejects_missing_or_incompatible_data_and_bad_thresholds(self) -> None:
        model = DynamicGestureTrainer().train(make_datasets(6))
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            type(model).from_dict({})
        payload = model.to_dict()
        payload["version"] = 999
        with self.assertRaisesRegex(ValueError, "Unsupported dynamic model version"):
            type(model).from_dict(payload)
        payload = model.to_dict()
        del payload["thresholds"]
        with self.assertRaisesRegex(ValueError, "thresholds"):
            type(model).from_dict(payload)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            model.with_thresholds(max_distance=float("nan"), minimum_separation=0.1)


if __name__ == "__main__":
    unittest.main()
