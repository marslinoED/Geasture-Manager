from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from gesture_manager.dynamic_gestures import (
    DynamicGestureDataset, DynamicGestureExample, DynamicGestureStore,
    DynamicGestureTrainer, DynamicRecordingSession, create_dynamic_draft,
)
from gesture_manager.dynamic_studio import DynamicStudioRecorder
from gesture_manager.hand_landmarks import DetectedHand, NormalizedLandmark


def hand(x: float = 0.0) -> DetectedHand:
    points = tuple(NormalizedLandmark(
        x + 0.35 + (index % 5) * 0.025,
        0.3 + (index // 5) * 0.035 + (index % 3) * 0.004,
        (index % 4) * 0.003,
    ) for index in range(21))
    return DetectedHand(points, 0.99)


def engine_status(sequence: int, *, detected: bool = True, hands=()):
    return SimpleNamespace(processed_frames=sequence, hands_detected=detected,
                           hands=(hands or ((hand(sequence * 0.002),) if detected else ())))


class DynamicStudioIntegrationTests(unittest.TestCase):
    def test_fresh_engine_landmarks_record_and_commit_a_sequence(self):
        session = DynamicRecordingSession(create_dynamic_draft("Wave"))
        recorder = DynamicStudioRecorder(session, last_sequence=0)
        self.assertEqual(recorder.frame_count, 0)
        recorder.begin()
        for sequence in range(1, 9):
            update = recorder.consume(engine_status(sequence))
            self.assertEqual(update.frame_count, sequence)
        example = recorder.finish()
        self.assertEqual(len(example.frames), 8)
        self.assertEqual(len(session.dataset.examples), 1)

    def test_duplicate_snapshot_is_ignored_and_cancel_preserves_completed_takes(self):
        session = DynamicRecordingSession(create_dynamic_draft("Wave"))
        recorder = DynamicStudioRecorder(session)
        recorder.begin()
        recorder.consume(engine_status(1))
        self.assertEqual(recorder.consume(engine_status(1)).added_frames, 1)
        recorder.cancel()
        self.assertFalse(recorder.recording)
        self.assertEqual(session.dataset.examples, ())

    def test_dynamic_storage_is_separate_and_keeps_additional_sequences(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            store = DynamicGestureStore(root / "dynamic")
            session = DynamicRecordingSession(create_dynamic_draft("Wave"))
            for start in (0, 20):
                session.begin_example()
                for sequence in range(start, start + 8):
                    session.add_frame((hand(sequence * 0.002),), frame_sequence=sequence)
                session.finish_example()
                store.save(session.dataset)
            loaded = store.load(session.dataset.id)
            self.assertEqual(len(loaded.examples), 2)
            self.assertFalse((root / "static" / f"{loaded.id}.json").exists())

    def test_saved_dynamic_labels_feed_the_existing_sequence_classifier(self):
        datasets = []
        for label, name in enumerate(("Horizontal", "Vertical")):
            draft = create_dynamic_draft(name)
            examples = []
            for index in range(5):
                rng = np.random.default_rng(label * 100 + index)
                frames = []
                for point in range(12):
                    values = rng.normal(0, 0.01, 66)
                    progress = point / 11
                    values[63 if label == 0 else 64] += progress * 0.5
                    frames.append(tuple(float(value) for value in values))
                examples.append(DynamicGestureExample(
                    str(uuid4()), datetime.now(timezone.utc).isoformat(), tuple(frames)))
            datasets.append(DynamicGestureDataset(draft.id, draft.name, tuple(examples)))
        model = DynamicGestureTrainer().train(tuple(datasets))
        self.assertEqual({item.name for item in model.labels}, {"Horizontal", "Vertical"})
        self.assertEqual(model.evaluation.labels, ("Horizontal", "Vertical"))


if __name__ == "__main__":
    unittest.main()
