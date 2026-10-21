"""Direct local LiteRT execution of the official MediaPipe hand model bundle.

This module contains the image/data conversion and model invocation only. It
does not own a camera, worker, persistence, or GUI resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, sqrt
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class NormalizedLandmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class DetectedHand:
    landmarks: tuple[NormalizedLandmark, ...]
    confidence: float


@dataclass(frozen=True)
class PalmAnchor:
    x: float
    y: float
    width: float = 1.0
    height: float = 1.0


class InterpreterLike(Protocol):
    def allocate_tensors(self) -> None: ...
    def get_input_details(self) -> list[dict[str, object]]: ...
    def get_output_details(self) -> list[dict[str, object]]: ...
    def set_tensor(self, index: int, value: np.ndarray) -> None: ...
    def invoke(self) -> None: ...
    def get_tensor(self, index: int) -> np.ndarray: ...


InterpreterFactory = Callable[[str, int], InterpreterLike]


def generate_palm_anchors(input_size: int = 192) -> tuple[PalmAnchor, ...]:
    """Generate anchors from the official MediaPipe palm detector graph setup."""
    if input_size < 1:
        raise ValueError("input_size must be positive")
    strides = (8, 16, 16, 16)
    min_scale, max_scale = 0.1484375, 0.75
    anchors: list[PalmAnchor] = []
    layer = 0
    while layer < len(strides):
        group_end = layer
        while group_end < len(strides) and strides[group_end] == strides[layer]:
            group_end += 1
        scales: list[float] = []
        for i in range(layer, group_end):
            scale = min_scale + (max_scale - min_scale) * i / (len(strides) - 1)
            next_scale = (
                min_scale + (max_scale - min_scale) * (i + 1) / (len(strides) - 1)
                if i < len(strides) - 1
                else 1.0
            )
            scales.extend((scale, sqrt(scale * next_scale)))
        grid = ceil(input_size / strides[layer])
        for y in range(grid):
            for x in range(grid):
                for _scale in scales:
                    anchors.append(
                        PalmAnchor(
                            x=(x + 0.5) * strides[layer] / input_size,
                            y=(y + 0.5) * strides[layer] / input_size,
                        )
                    )
        layer = group_end
    if len(anchors) != 2016:
        raise RuntimeError(f"Unexpected palm anchor count: {len(anchors)}")
    return tuple(anchors)


def bgr_frame_to_rgb_tensor(frame: object, size: int) -> np.ndarray:
    """Stretch a valid OpenCV BGR frame into an RGB float32 [0, 1] tensor."""
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or not array.size:
        raise ValueError("Camera frame must be a non-empty H×W×3 BGR image")
    if array.dtype != np.uint8:
        raise ValueError("Camera frame must use uint8 pixels")
    rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return (resized.astype(np.float32) / 255.0)[np.newaxis, ...]


def decode_palm_candidates(
    raw_boxes: np.ndarray,
    raw_scores: np.ndarray,
    anchors: tuple[PalmAnchor, ...],
    *,
    threshold: float = 0.5,
    max_hands: int = 2,
    iou_threshold: float = 0.3,
) -> tuple[tuple[float, float, float, float, float], ...]:
    """Decode official SSD palm tensors to (cx, cy, width, height, score)."""
    boxes = np.asarray(raw_boxes).reshape(-1, 18)
    scores = np.asarray(raw_scores).reshape(-1)
    if len(boxes) != len(anchors) or len(scores) != len(anchors):
        raise ValueError("Palm detector output does not match generated anchors")
    logits = np.clip(scores.astype(np.float64), -80.0, 80.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    candidate_indices = np.flatnonzero(probabilities >= threshold)
    if candidate_indices.size == 0:
        return ()
    candidate_indices = candidate_indices[np.argsort(probabilities[candidate_indices])[::-1]]
    selected: list[tuple[float, float, float, float, float]] = []
    for index in candidate_indices:
        anchor = anchors[int(index)]
        raw = boxes[index]
        # Palm graph uses XYWH output order and linear box size decoding.
        cx = float(raw[0]) / 192.0 * anchor.width + anchor.x
        cy = float(raw[1]) / 192.0 * anchor.height + anchor.y
        width = float(raw[2]) / 192.0 * anchor.width
        height = float(raw[3]) / 192.0 * anchor.height
        if width <= 0 or height <= 0:
            continue
        candidate = (cx, cy, width, height, float(probabilities[index]))
        if all(_box_iou(candidate, prior) <= iou_threshold for prior in selected):
            selected.append(candidate)
            if len(selected) >= max_hands:
                break
    return tuple(selected)


def _box_iou(
    first: tuple[float, float, float, float, float],
    second: tuple[float, float, float, float, float],
) -> float:
    ax0, ay0 = first[0] - first[2] / 2, first[1] - first[3] / 2
    ax1, ay1 = first[0] + first[2] / 2, first[1] + first[3] / 2
    bx0, by0 = second[0] - second[2] / 2, second[1] - second[3] / 2
    bx1, by1 = second[0] + second[2] / 2, second[1] + second[3] / 2
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


def project_crop_landmarks(
    raw_landmarks: np.ndarray,
    *,
    image_shape: tuple[int, ...],
    crop_center: tuple[float, float],
    crop_size: float,
) -> tuple[NormalizedLandmark, ...]:
    """Normalize 21 ROI model outputs back into full-frame coordinates."""
    values = np.asarray(raw_landmarks, dtype=np.float32).reshape(-1)
    if values.size != 63:
        raise ValueError("Landmark model must return exactly 21 x/y/z points")
    if len(image_shape) < 2:
        raise ValueError("image_shape must include height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0 or crop_size <= 0:
        raise ValueError("Image and crop sizes must be positive")
    cx, cy = crop_center
    output: list[NormalizedLandmark] = []
    for x_px, y_px, z_px in values.reshape(21, 3):
        local_x = float(x_px) / 224.0
        local_y = float(y_px) / 224.0
        # MediaPipe's landmark graph normalizes screen z by input width and 0.4.
        local_z = float(z_px) / 224.0 * 0.4
        output.append(
            NormalizedLandmark(
                x=(cx - crop_size / 2 + local_x * crop_size) / width,
                y=(cy - crop_size / 2 + local_y * crop_size) / height,
                z=local_z * crop_size / width,
            )
        )
    return tuple(output)


class LiteHandLandmarkModel:
    """Load the two task-bundle TFLite models once and reuse their interpreters."""

    def __init__(
        self,
        model_dir: Path,
        *,
        interpreter_factory: InterpreterFactory | None = None,
        num_threads: int = 2,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self._factory = interpreter_factory or _create_litert_interpreter
        self._palm: InterpreterLike | None = None
        self._landmark: InterpreterLike | None = None
        self._palm_input: int | None = None
        self._landmark_input: int | None = None
        self._palm_outputs: tuple[int, int] | None = None
        self._landmark_outputs: tuple[int, int] | None = None
        self._anchors = generate_palm_anchors()

    def load(self) -> None:
        if self._palm is not None and self._landmark is not None:
            return
        palm_path = self.model_dir / "hand_detector.tflite"
        landmark_path = self.model_dir / "hand_landmarks_detector.tflite"
        missing = [str(path) for path in (palm_path, landmark_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing local hand model asset(s): " + ", ".join(missing))
        try:
            palm = self._factory(str(palm_path), self.num_threads)
            palm.allocate_tensors()
            landmark = self._factory(str(landmark_path), self.num_threads)
            landmark.allocate_tensors()
            self._palm_input = int(palm.get_input_details()[0]["index"])
            self._landmark_input = int(landmark.get_input_details()[0]["index"])
            palm_details = palm.get_output_details()
            landmark_details = landmark.get_output_details()
            palm_box = next(item for item in palm_details if len(item["shape"]) == 3 and item["shape"][-1] == 18)
            palm_score = next(item for item in palm_details if len(item["shape"]) == 3 and item["shape"][-1] == 1)
            landmark_points = next(item for item in landmark_details if int(np.prod(item["shape"])) == 63)
            presence = next(item for item in landmark_details if int(np.prod(item["shape"])) == 1 and item is not landmark_points)
            self._palm_outputs = (int(palm_box["index"]), int(palm_score["index"]))
            self._landmark_outputs = (int(landmark_points["index"]), int(presence["index"]))
            self._palm, self._landmark = palm, landmark
        except Exception as exc:
            self.close()
            raise RuntimeError(f"Could not load the local LiteRT hand models: {exc}") from exc

    @property
    def is_loaded(self) -> bool:
        return self._palm is not None and self._landmark is not None

    def detect(self, frame: object) -> tuple[DetectedHand, ...]:
        if not self.is_loaded:
            raise RuntimeError("Hand model has not been loaded")
        array = np.asarray(frame)
        detector_input = bgr_frame_to_rgb_tensor(array, 192)
        palm = self._palm
        landmark = self._landmark
        palm_outputs = self._palm_outputs
        landmark_outputs = self._landmark_outputs
        assert palm is not None and landmark is not None
        assert self._palm_input is not None and self._landmark_input is not None
        assert palm_outputs is not None and landmark_outputs is not None
        palm.set_tensor(self._palm_input, detector_input)
        palm.invoke()
        boxes = palm.get_tensor(palm_outputs[0])
        scores = palm.get_tensor(palm_outputs[1])
        palms = decode_palm_candidates(boxes, scores, self._anchors)
        if not palms:
            return ()
        height, width = array.shape[:2]
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        hands: list[DetectedHand] = []
        for cx, cy, box_width, box_height, score in palms:
            center_x = cx * width
            center_y = cy * height
            side = min(max(box_width * width, box_height * height) * 2.6, max(width, height) * 1.5)
            center_y -= side * 0.1
            crop = cv2.getRectSubPix(rgb, (max(1, round(side)), max(1, round(side))), (center_x, center_y))
            crop_input = (cv2.resize(crop, (224, 224)).astype(np.float32) / 255.0)[None, ...]
            landmark.set_tensor(self._landmark_input, crop_input)
            landmark.invoke()
            raw_points = landmark.get_tensor(landmark_outputs[0])
            presence = float(np.asarray(landmark.get_tensor(landmark_outputs[1])).reshape(-1)[0])
            if presence < 0.5:
                continue
            hands.append(
                DetectedHand(
                    landmarks=project_crop_landmarks(
                        raw_points,
                        image_shape=array.shape,
                        crop_center=(center_x, center_y),
                        crop_size=side,
                    ),
                    confidence=presence,
                )
            )
        return tuple(hands)

    def close(self) -> None:
        # Interpreter has no public close() method; releasing both references
        # lets LiteRT deallocate native model resources deterministically by GC.
        self._palm = None
        self._landmark = None
        self._palm_outputs = None
        self._landmark_outputs = None
        self._palm_input = None
        self._landmark_input = None


def _create_litert_interpreter(model_path: str, num_threads: int) -> InterpreterLike:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as exc:
        raise RuntimeError(
            "AI Edge LiteRT is not installed. Install the pinned project dependencies."
        ) from exc
    return Interpreter(model_path=model_path, num_threads=num_threads)
