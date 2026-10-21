from pathlib import Path
import unittest

import numpy as np

from gesture_manager.hand_landmarks import (
    LiteHandLandmarkModel,
    NormalizedLandmark,
    bgr_frame_to_rgb_tensor,
    decode_palm_candidates,
    generate_palm_anchors,
    project_crop_landmarks,
)


class FakeInterpreter:
    def __init__(self, path: str, _threads: int, positive_palm: bool = False) -> None:
        self.landmark_model = "landmarks_detector" in path
        self.positive_palm = positive_palm
        self.invocations = 0

    def allocate_tensors(self) -> None:
        pass

    def get_input_details(self):
        return [{"index": 1}]

    def get_output_details(self):
        if self.landmark_model:
            return [
                {"index": 10, "shape": [1, 63]},
                {"index": 11, "shape": [1, 1]},
                {"index": 12, "shape": [1, 1]},
                {"index": 13, "shape": [1, 63]},
            ]
        return [{"index": 20, "shape": [1, 2016, 18]}, {"index": 21, "shape": [1, 2016, 1]}]

    def set_tensor(self, _index: int, _value: np.ndarray) -> None:
        pass

    def invoke(self) -> None:
        self.invocations += 1

    def get_tensor(self, index: int) -> np.ndarray:
        if index == 20:
            boxes = np.zeros((1, 2016, 18), dtype=np.float32)
            if self.positive_palm:
                boxes[0, 0, 2:4] = 80
            return boxes
        if index == 21:
            scores = np.full((1, 2016, 1), -100.0, dtype=np.float32)
            if self.positive_palm:
                scores[0, 0, 0] = 10
            return scores
        if index == 10:
            return np.zeros((1, 63), dtype=np.float32)
        return np.ones((1, 1), dtype=np.float32)


class HandLandmarkConversionTests(unittest.TestCase):
    def test_bgr_conversion_returns_rgb_float_tensor(self) -> None:
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[:, :] = (10, 20, 30)
        tensor = bgr_frame_to_rgb_tensor(frame, 1)
        np.testing.assert_allclose(tensor[0, 0, 0], [30 / 255, 20 / 255, 10 / 255])
        self.assertEqual(tensor.dtype, np.float32)

    def test_invalid_frame_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bgr_frame_to_rgb_tensor(np.zeros((5, 5), dtype=np.uint8), 192)

    def test_anchor_count_and_empty_decode(self) -> None:
        anchors = generate_palm_anchors()
        self.assertEqual(len(anchors), 2016)
        self.assertEqual(decode_palm_candidates(np.zeros((1, 2016, 18)),
                                                np.full((1, 2016, 1), -100), anchors), ())

    def test_landmarks_project_into_normalized_frame(self) -> None:
        raw = np.zeros((1, 63), dtype=np.float32)
        points = project_crop_landmarks(raw, image_shape=(480, 640, 3),
                                        crop_center=(320, 240), crop_size=200)
        self.assertEqual(len(points), 21)
        self.assertEqual(points[0], NormalizedLandmark(0.34375, 0.2916666666666667, 0.0))


class HandLandmarkModelTests(unittest.TestCase):
    def test_missing_assets_report_local_paths(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Missing local hand model"):
            LiteHandLandmarkModel(Path("not-installed")).load()

    def test_interpreters_load_once_and_no_palm_returns_no_hands(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "models"
        created: list[FakeInterpreter] = []

        def factory(path: str, threads: int) -> FakeInterpreter:
            self.assertEqual(threads, 2)
            interpreter = FakeInterpreter(path, threads)
            created.append(interpreter)
            return interpreter

        model = LiteHandLandmarkModel(assets, interpreter_factory=factory)
        model.load()
        model.load()
        self.assertEqual(len(created), 2)
        self.assertEqual(model.detect(np.zeros((480, 640, 3), dtype=np.uint8)), ())
        self.assertEqual(created[0].invocations, 1)
        model.close()
        self.assertFalse(model.is_loaded)

    def test_mocked_palm_and_landmark_inference_returns_21_points(self) -> None:
        assets = Path(__file__).resolve().parents[1] / "models"
        created: list[FakeInterpreter] = []

        def factory(path: str, threads: int) -> FakeInterpreter:
            interpreter = FakeInterpreter(path, threads, positive_palm=True)
            created.append(interpreter)
            return interpreter

        model = LiteHandLandmarkModel(assets, interpreter_factory=factory)
        model.load()
        hands = model.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(len(hands), 1)
        self.assertEqual(len(hands[0].landmarks), 21)
        self.assertEqual(created[1].invocations, 1)
        model.close()


if __name__ == "__main__":
    unittest.main()
