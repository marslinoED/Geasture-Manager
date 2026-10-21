from datetime import datetime, timezone
from pathlib import Path
import tempfile
import queue
import time
import unittest
from uuid import uuid4

import numpy as np

from gesture_manager.gesture_data import (
    FEATURE_VERSION,
    GestureDataset,
    TrainingExample,
    create_draft,
)
from gesture_manager.gesture_training import (
    CLASSIFIER_VERSION,
    GestureModelStore,
    GestureTrainer,
    LiveClassifierWorker,
    grouped_validation_split,
    make_training_data,
    model_matches_datasets,
)
from gesture_manager.hand_landmarks import DetectedHand, NormalizedLandmark
from gesture_manager.gesture_data import normalize_hand_features
from gesture_manager.recognition import LiveRecognitionWorker, RecognitionEvent


def synthetic_datasets(*, groups_per_class: int = 9, repeats: int = 2):
    rng = np.random.default_rng(701)
    names = ("User's salute", "My curled pose", "Custom point")
    datasets = []
    for label_index, name in enumerate(names):
        draft = create_draft(name, 1)
        examples = []
        for group_index in range(groups_per_class):
            center = rng.normal(0, 0.12, 63)
            center[label_index] += 1.3
            for _ in range(repeats):
                features = center + rng.normal(0, 0.001, 63)
                examples.append(TrainingExample(
                    str(uuid4()), datetime.now(timezone.utc).isoformat(),
                    tuple(float(value) for value in features),
                ))
        datasets.append(GestureDataset(
            id=draft.id, name=draft.name, gesture_type=draft.gesture_type,
            hand_count=1, examples=tuple(examples),
        ))
    return tuple(datasets)


class GestureTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp.name) / "models" / "custom_gestures.json"
        self.model_store = GestureModelStore(self.model_path)
        self.trainer = GestureTrainer(self.model_store)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_custom_label_training_grouped_evaluation_and_serialization(self) -> None:
        datasets = synthetic_datasets()
        x, y, groups, labels, counts = make_training_data(datasets)
        train, validation = grouped_validation_split(x, y, groups)
        self.assertTrue(set(groups[train]).isdisjoint(set(groups[validation])))
        self.assertEqual(tuple(label.name for label in labels),
                         ("Custom point", "My curled pose", "User's salute"))
        outcome = self.trainer.train(datasets)
        self.assertTrue(outcome.report.evaluated)
        self.assertTrue(outcome.report.reliable)
        self.assertIsNotNone(outcome.report.confusion)
        self.assertEqual(len(outcome.report.classes), 3)
        self.assertEqual(outcome.classifier_version, CLASSIFIER_VERSION)
        self.assertEqual(outcome.model.feature_version, FEATURE_VERSION)
        self.assertEqual(outcome.model.training_sample_count, sum(counts))
        self.assertTrue(model_matches_datasets(outcome.model, datasets))
        loaded = self.model_store.load()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.to_dict(), outcome.model.to_dict())
        for dataset in datasets:
            expected = next(label for label in loaded.labels if label.dataset_id == dataset.id)
            self.assertEqual(loaded.predict_one(dataset.examples[0].features), expected)
            scored = loaded.predict_with_scores(dataset.examples[0].features)
            self.assertEqual(scored.label, expected)
            self.assertTrue(np.isfinite(scored.decision_margin))
            self.assertGreaterEqual(scored.vote_share, 0.0)
            self.assertLessEqual(scored.vote_share, 1.0)

    def test_two_class_portable_svc_matches_custom_label_order(self) -> None:
        datasets = synthetic_datasets()[:2]
        model = self.trainer.train(datasets).model
        for dataset in datasets:
            for example in dataset.examples:
                self.assertEqual(model.predict_one(example.features).dataset_id, dataset.id)

    def test_saved_model_load_does_not_retrain_and_missing_file_is_safe(self) -> None:
        self.assertIsNone(self.trainer.load_saved())
        datasets = synthetic_datasets(groups_per_class=6)
        self.trainer.train(datasets)
        loaded = self.trainer.load_saved()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.labels[0].name, "Custom point")

    def test_disabled_labels_are_removed_from_live_model_comparison(self) -> None:
        model = self.trainer.train(synthetic_datasets()).model
        excluded = model.labels[-1].dataset_id
        enabled = {label.dataset_id for label in model.labels[:-1]}
        for dataset in synthetic_datasets(groups_per_class=2, repeats=1):
            prediction = model.predict_with_scores(
                dataset.examples[0].features, enabled_label_ids=enabled
            )
            self.assertIn(prediction.label.dataset_id, enabled)
            self.assertNotEqual(prediction.label.dataset_id, excluded)
        with self.assertRaisesRegex(ValueError, "At least two gestures"):
            model.predict_with_scores(
                synthetic_datasets()[0].examples[0].features,
                enabled_label_ids={model.labels[0].dataset_id},
            )

    def test_invalid_class_counts_and_mixed_hand_configuration_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two custom gestures"):
            make_training_data(synthetic_datasets()[:1])
        datasets = list(synthetic_datasets())
        datasets[0] = GestureDataset(
            id=datasets[0].id, name=datasets[0].name, gesture_type="static_pose",
            hand_count=2, examples=datasets[0].examples,
        )
        with self.assertRaisesRegex(ValueError, "same one-hand or two-hand"):
            make_training_data(tuple(datasets))
        too_small = list(synthetic_datasets())
        too_small[0] = GestureDataset(
            id=too_small[0].id, name=too_small[0].name, gesture_type="static_pose",
            hand_count=1, examples=too_small[0].examples[:2],
        )
        with self.assertRaisesRegex(ValueError, "at least 3"):
            make_training_data(tuple(too_small))

    def test_too_few_examples_produce_unreliable_warning_not_training_accuracy(self) -> None:
        outcome = self.trainer.train(synthetic_datasets(groups_per_class=3, repeats=1))
        self.assertTrue(outcome.report.evaluated)
        self.assertFalse(outcome.report.reliable)
        self.assertTrue(any("fewer than 5" in warning for warning in outcome.report.warnings))
        self.assertFalse(hasattr(outcome.report, "training_accuracy"))

    def test_invalid_and_incompatible_model_files_are_handled(self) -> None:
        self.model_path.parent.mkdir(parents=True)
        self.model_path.write_text("{invalid", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not valid JSON"):
            self.model_store.load()
        model = self.trainer.train(synthetic_datasets(groups_per_class=5)).model
        payload = model.to_dict()
        payload["feature_version"] = "future-version"
        import json
        self.model_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Incompatible feature version"):
            self.model_store.load()

    def test_model_rejects_string_coercion_for_reliability_metadata(self) -> None:
        model = self.trainer.train(synthetic_datasets(groups_per_class=5)).model
        payload = model.to_dict()
        payload["training_metadata"]["evaluation_reliable"] = "false"
        from gesture_manager.gesture_training import PortableGestureModel
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            PortableGestureModel.from_dict(payload)

    def test_changed_examples_mark_saved_model_stale(self) -> None:
        datasets = synthetic_datasets(groups_per_class=5)
        model = self.trainer.train(datasets).model
        changed = list(datasets)
        first = changed[0]
        changed[0] = GestureDataset(
            id=first.id, name=first.name, gesture_type=first.gesture_type,
            hand_count=first.hand_count, examples=first.examples[:-1],
        )
        self.assertFalse(model_matches_datasets(model, tuple(changed)))

    def test_live_worker_classifies_only_fresh_engine_samples(self) -> None:
        datasets = synthetic_datasets(groups_per_class=5)
        model = self.trainer.train(datasets).model
        hand = DetectedHand(tuple(
            NormalizedLandmark(0.4 + (index % 5) * 0.025,
                               0.4 + (index // 5) * 0.03 + (index % 3) * 0.004,
                               (index % 4) * 0.006)
            for index in range(21)
        ), 0.99)
        expected = model.predict_one(normalize_hand_features((hand,)))

        class Status:
            processed_frames = 1
            hands_detected = True
            hands = (hand,)

        status = Status()
        results = queue.Queue()
        worker = LiveClassifierWorker(model, lambda: status, results)
        worker.start()
        try:
            status.processed_frames = 2
            deadline = time.monotonic() + 1
            while results.empty() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(results.empty())
            sequence, prediction, error = results.get_nowait()
            self.assertEqual(sequence, 2)
            self.assertEqual(prediction, expected)
            self.assertIsNone(error)
        finally:
            worker.stop()
            self.assertTrue(worker.wait(1))

    def test_live_recognition_worker_emits_structured_diagnostic_event(self) -> None:
        model = self.trainer.train(synthetic_datasets(groups_per_class=5)).model
        hand = DetectedHand(tuple(
            NormalizedLandmark(0.4 + (index % 5) * 0.025,
                               0.4 + (index // 5) * 0.03 + (index % 3) * 0.004,
                               (index % 4) * 0.006)
            for index in range(21)
        ), 0.99)

        class Status:
            processed_frames = 1
            hands_detected = False
            hands = ()

        status = Status()
        results = queue.Queue(maxsize=8)
        worker = LiveRecognitionWorker(model, lambda: status, results,
                                       activation_frames=1, cooldown_seconds=0)
        worker.start()
        try:
            status.processed_frames = 2
            status.hands_detected = True
            status.hands = (hand,)
            deadline = time.monotonic() + 1
            while results.empty() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(results.empty())
            event = results.get_nowait()
            self.assertIsInstance(event, RecognitionEvent)
            self.assertEqual(event.frame_sequence, 2)
            self.assertIn(event.prediction, {label.name for label in model.labels} |
                          {"Uncertain — no gesture detected"})
            self.assertEqual(event.score_kind, "svc_ovo_decision_margin")
            self.assertIsNotNone(event.timestamp)
        finally:
            worker.stop()
            self.assertTrue(worker.wait(1))


if __name__ == "__main__":
    unittest.main()
