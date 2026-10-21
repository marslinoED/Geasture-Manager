"""Feature-only recording and DTW classification for dynamic hand gestures.

This is intentionally independent of the static-pose dataset/model format.
The first dynamic release supports one hand; later modalities can add their
own modules without changing either hand representation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors

from gesture_manager.gesture_data import normalize_hand_features, validate_name
from gesture_manager.hand_landmarks import DetectedHand


DYNAMIC_DATASET_FORMAT = "gesture-manager-dynamic-dataset"
DYNAMIC_DATASET_VERSION = 1
DYNAMIC_FEATURE_VERSION = "hand-motion-sequence-v1"
DYNAMIC_MODEL_FORMAT = "gesture-manager-dynamic-dtw-model"
DYNAMIC_MODEL_VERSION = 2
FEATURES_PER_FRAME = 66
MIN_SEQUENCE_FRAMES = 8
MAX_SEQUENCE_FRAMES = 60
MAX_EXAMPLES_PER_GESTURE = 500
DUPLICATE_RMS_TOLERANCE = 0.025


@dataclass(frozen=True)
class DynamicGestureLabel:
    dataset_id: str
    name: str


@dataclass(frozen=True)
class DynamicGestureExample:
    id: str
    captured_at: str
    frames: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class DynamicGestureDataset:
    id: str
    name: str
    examples: tuple[DynamicGestureExample, ...] = ()
    format_version: int = DYNAMIC_DATASET_VERSION
    feature_version: str = DYNAMIC_FEATURE_VERSION
    hand_count: int = 1


@dataclass(frozen=True)
class DynamicPrediction:
    label: DynamicGestureLabel | None
    distance: float
    separation: float
    uncertain: bool


@dataclass(frozen=True)
class DynamicEvaluation:
    evaluated: bool
    reliable: bool
    accuracy: float | None
    confusion: tuple[tuple[int, ...], ...] | None
    labels: tuple[str, ...]
    warnings: tuple[str, ...]
    example_counts: tuple[int, ...]
    independent_group_counts: tuple[int, ...]


@dataclass(frozen=True)
class _LabeledSequence:
    label_index: int
    frames: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class DynamicGestureModel:
    labels: tuple[DynamicGestureLabel, ...]
    examples: tuple[_LabeledSequence, ...]
    trained_at: str
    evaluation: DynamicEvaluation
    max_distance_threshold: float = 0.35
    minimum_separation_threshold: float = 0.015

    def predict(self, frames: tuple[tuple[float, ...], ...], *,
                max_distance: float | None = None,
                minimum_separation: float | None = None) -> DynamicPrediction:
        validate_sequence(frames)
        max_distance = (self.max_distance_threshold if max_distance is None else max_distance)
        minimum_separation = (self.minimum_separation_threshold
                              if minimum_separation is None else minimum_separation)
        if not math.isfinite(max_distance) or max_distance < 0:
            raise ValueError("Maximum dynamic distance must be finite and non-negative")
        if not math.isfinite(minimum_separation) or minimum_separation < 0:
            raise ValueError("Minimum separation must be finite and non-negative")
        distances = [math.inf] * len(self.labels)
        for example in self.examples:
            distances[example.label_index] = min(
                distances[example.label_index], dtw_distance(frames, example.frames))
        order = sorted(range(len(distances)), key=distances.__getitem__)
        best, second = order[:2]
        separation = distances[second] - distances[best]
        uncertain = distances[best] > max_distance or separation < minimum_separation
        return DynamicPrediction(
            None if uncertain else self.labels[best],
            float(distances[best]), float(separation), uncertain,
        )

    def with_thresholds(self, *, max_distance: float,
                        minimum_separation: float) -> "DynamicGestureModel":
        """Return a copy with persisted, model-independent rejection limits."""
        updated = replace(self, max_distance_threshold=max_distance,
                          minimum_separation_threshold=minimum_separation)
        validate_model(updated)
        return updated

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": DYNAMIC_MODEL_FORMAT,
            "version": DYNAMIC_MODEL_VERSION,
            "feature_version": DYNAMIC_FEATURE_VERSION,
            "classifier": "nearest-example-dtw-v1",
            "trained_at": self.trained_at,
            "thresholds": {
                "max_distance": self.max_distance_threshold,
                "minimum_separation": self.minimum_separation_threshold,
            },
            "labels": [{"id": item.dataset_id, "name": item.name} for item in self.labels],
            "examples": [
                {"label_index": item.label_index, "frames": [list(frame) for frame in item.frames]}
                for item in self.examples
            ],
            "evaluation": evaluation_to_dict(self.evaluation),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "DynamicGestureModel":
        if not isinstance(raw, dict) or raw.get("format") != DYNAMIC_MODEL_FORMAT:
            raise ValueError("Unrecognized dynamic gesture model format")
        if raw.get("version") != DYNAMIC_MODEL_VERSION:
            raise ValueError("Unsupported dynamic model version")
        if raw.get("feature_version") != DYNAMIC_FEATURE_VERSION:
            raise ValueError("Incompatible dynamic feature version")
        if raw.get("classifier") != "nearest-example-dtw-v1":
            raise ValueError("Unsupported dynamic classifier")
        try:
            labels = tuple(DynamicGestureLabel(str(item["id"]), str(item["name"]))
                           for item in raw["labels"])
            examples = tuple(_LabeledSequence(
                int(item["label_index"]),
                tuple(tuple(float(value) for value in frame) for frame in item["frames"]),
            ) for item in raw["examples"])
            thresholds = raw["thresholds"]
            if not isinstance(thresholds, dict) or set(thresholds) != {
                    "max_distance", "minimum_separation"}:
                raise ValueError("Dynamic model thresholds are missing or unsupported")
            model = cls(
                labels=labels, examples=examples, trained_at=str(raw["trained_at"]),
                evaluation=evaluation_from_dict(raw["evaluation"]),
                max_distance_threshold=float(thresholds["max_distance"]),
                minimum_separation_threshold=float(thresholds["minimum_separation"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Dynamic model fields are invalid: {exc}") from exc
        validate_model(model)
        return model


def dynamic_frame_features(hand: DetectedHand, *, origin: tuple[float, float] | None = None,
                           reference_scale: float | None = None) -> tuple[tuple[float, ...], tuple[float, float], float]:
    """Extract pose-normalized landmarks plus wrist path in hand-size units."""
    if len(hand.landmarks) != 21:
        raise ValueError("Dynamic hand frames require exactly 21 landmarks")
    wrist = hand.landmarks[0]
    middle_mcp = hand.landmarks[9]
    scale = math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y)
    if not math.isfinite(scale) or scale < 1e-6:
        raise ValueError("Dynamic hand frame has degenerate palm scale")
    base = origin or (wrist.x, wrist.y)
    normalizer = reference_scale or scale
    if not math.isfinite(normalizer) or normalizer < 1e-6:
        raise ValueError("Dynamic reference scale is invalid")
    pose = normalize_hand_features((hand,))
    path_x = (wrist.x - base[0]) / normalizer
    path_y = (wrist.y - base[1]) / normalizer
    result = (*pose, path_x, path_y, (wrist.z - hand.landmarks[9].z) / normalizer)
    if not all(math.isfinite(value) and abs(value) <= 20 for value in result):
        raise ValueError("Dynamic features are outside the supported numeric range")
    return tuple(result), base, normalizer


class DynamicRecordingSession:
    """Capture one bounded sequence at a time; only completed takes persist."""

    def __init__(self, dataset: DynamicGestureDataset) -> None:
        validate_dynamic_dataset(dataset)
        self.dataset = dataset
        self._recording: list[tuple[float, ...]] | None = None
        self._origin: tuple[float, float] | None = None
        self._scale: float | None = None
        self._last_frame_sequence: int | None = None

    @property
    def recording(self) -> bool:
        return self._recording is not None

    def begin_example(self) -> None:
        if self.recording:
            raise RuntimeError("A dynamic example is already recording")
        if len(self.dataset.examples) >= MAX_EXAMPLES_PER_GESTURE:
            raise ValueError("Dynamic gesture example limit reached")
        self._recording = []
        self._origin = None
        self._scale = None
        self._last_frame_sequence = None

    def add_frame(self, hands: tuple[DetectedHand, ...], *, frame_sequence: int | None = None) -> None:
        if not self.recording:
            raise RuntimeError("Begin a dynamic example before adding frames")
        if len(hands) != 1:
            raise ValueError("Dynamic gestures currently require exactly one visible hand")
        if frame_sequence is not None and frame_sequence == self._last_frame_sequence:
            return
        features, origin, scale = dynamic_frame_features(
            hands[0], origin=self._origin, reference_scale=self._scale)
        if self._origin is None:
            self._origin, self._scale = origin, scale
        assert self._recording is not None
        if len(self._recording) >= MAX_SEQUENCE_FRAMES:
            raise ValueError(f"A sequence is limited to {MAX_SEQUENCE_FRAMES} fresh frames")
        self._recording.append(features)
        self._last_frame_sequence = frame_sequence

    def finish_example(self) -> DynamicGestureExample:
        if not self.recording or self._recording is None:
            raise RuntimeError("No dynamic example is being recorded")
        if len(self._recording) < MIN_SEQUENCE_FRAMES:
            raise ValueError(f"Record at least {MIN_SEQUENCE_FRAMES} fresh frames before saving a take")
        example = DynamicGestureExample(
            str(uuid4()), datetime.now(timezone.utc).isoformat(), tuple(self._recording))
        self.dataset = DynamicGestureDataset(
            self.dataset.id, self.dataset.name, (*self.dataset.examples, example))
        self.cancel_example()
        return example

    def cancel_example(self) -> None:
        self._recording = None
        self._origin = None
        self._scale = None
        self._last_frame_sequence = None

    def remove_example(self, example_id: str) -> None:
        remaining = tuple(item for item in self.dataset.examples if item.id != example_id)
        if len(remaining) == len(self.dataset.examples):
            raise KeyError(f"Dynamic example not found: {example_id}")
        self.dataset = DynamicGestureDataset(self.dataset.id, self.dataset.name, remaining)
        self.cancel_example()


def create_dynamic_draft(name: str) -> DynamicGestureDataset:
    return DynamicGestureDataset(str(uuid4()), validate_name(name))


def validate_sequence(frames: tuple[tuple[float, ...], ...]) -> None:
    if not isinstance(frames, (tuple, list)) or not MIN_SEQUENCE_FRAMES <= len(frames) <= MAX_SEQUENCE_FRAMES:
        raise ValueError(f"Sequence must contain {MIN_SEQUENCE_FRAMES} to {MAX_SEQUENCE_FRAMES} frames")
    for frame in frames:
        if not isinstance(frame, (tuple, list)) or len(frame) != FEATURES_PER_FRAME or not all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and abs(value) <= 20 for value in frame):
            raise ValueError("Dynamic frame features are malformed")


def validate_dynamic_dataset(dataset: DynamicGestureDataset) -> None:
    try:
        if str(UUID(dataset.id)) != dataset.id.lower():
            raise ValueError("Gesture identifier must be a canonical UUID")
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid dynamic gesture identifier") from exc
    validate_name(dataset.name)
    if dataset.format_version != DYNAMIC_DATASET_VERSION:
        raise ValueError("Unsupported dynamic dataset version")
    if dataset.feature_version != DYNAMIC_FEATURE_VERSION:
        raise ValueError("Incompatible dynamic feature version")
    if dataset.hand_count != 1:
        raise ValueError("This dynamic gesture format supports exactly one hand")
    if len(dataset.examples) > MAX_EXAMPLES_PER_GESTURE:
        raise ValueError("Dynamic gesture contains too many examples")
    ids: set[str] = set()
    for example in dataset.examples:
        try:
            UUID(example.id)
            timestamp = datetime.fromisoformat(example.captured_at)
            if timestamp.tzinfo is None:
                raise ValueError("timestamp must include a timezone")
        except (ValueError, TypeError) as exc:
            raise ValueError("Dynamic example ID or timestamp is invalid") from exc
        if example.id in ids:
            raise ValueError("Dynamic example IDs must be unique")
        ids.add(example.id)
        validate_sequence(example.frames)


def dynamic_dataset_to_dict(dataset: DynamicGestureDataset) -> dict[str, Any]:
    validate_dynamic_dataset(dataset)
    return {
        "format": DYNAMIC_DATASET_FORMAT, "format_version": dataset.format_version,
        "feature_extraction_version": dataset.feature_version, "id": dataset.id,
        "name": dataset.name, "gesture_type": "dynamic_hand", "hand_count": 1,
        "examples": [{"id": item.id, "captured_at": item.captured_at,
                      "frames": [list(frame) for frame in item.frames]}
                     for item in dataset.examples],
    }


def dynamic_dataset_from_dict(raw: Any) -> DynamicGestureDataset:
    if not isinstance(raw, dict) or raw.get("format") != DYNAMIC_DATASET_FORMAT:
        raise ValueError("Unrecognized dynamic dataset format")
    if raw.get("gesture_type") != "dynamic_hand":
        raise ValueError("Dataset is not a dynamic hand gesture")
    try:
        examples = tuple(DynamicGestureExample(
            str(item["id"]), str(item["captured_at"]),
            tuple(tuple(float(value) for value in frame) for frame in item["frames"]),
        ) for item in raw["examples"])
        dataset = DynamicGestureDataset(
            str(raw["id"]), str(raw["name"]), examples,
            int(raw["format_version"]), str(raw["feature_extraction_version"]),
            int(raw["hand_count"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Dynamic dataset fields are invalid: {exc}") from exc
    validate_dynamic_dataset(dataset)
    return dataset


class DynamicGestureStore:
    """Atomic JSON store separate from static pose datasets and models."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def path_for(self, gesture_id: str) -> Path:
        try:
            canonical_id = str(UUID(gesture_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid dynamic gesture identifier") from exc
        return self.root / f"{canonical_id}.json"

    def list(self) -> tuple[DynamicGestureDataset, ...]:
        self.root.mkdir(parents=True, exist_ok=True)
        return tuple(sorted((self.load(path.stem) for path in self.root.glob("*.json")),
                            key=lambda item: item.name.casefold()))

    def load(self, gesture_id: str) -> DynamicGestureDataset:
        try:
            raw = json.loads(self.path_for(gesture_id).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read dynamic dataset: {exc}") from exc
        return dynamic_dataset_from_dict(raw)

    def save(self, dataset: DynamicGestureDataset) -> None:
        validate_dynamic_dataset(dataset)
        self.root.mkdir(parents=True, exist_ok=True)
        for existing in self.list():
            if existing.id != dataset.id and existing.name.casefold() == dataset.name.casefold():
                raise ValueError(f"A dynamic gesture named {dataset.name!r} already exists")
        path = self.path_for(dataset.id)
        temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.root,
                    prefix=f".{dataset.id}.", suffix=".tmp", delete=False) as stream:
                temp = Path(stream.name)
                json.dump(dynamic_dataset_to_dict(dataset), stream, ensure_ascii=False,
                          allow_nan=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    def delete(self, gesture_id: str) -> None:
        path = self.path_for(gesture_id)
        if path.is_symlink():
            path.unlink()
        else:
            path.resolve().relative_to(self.root)
            path.unlink()


def dtw_distance(first: tuple[tuple[float, ...], ...],
                 second: tuple[tuple[float, ...], ...], *, window: int = 12) -> float:
    """Length-normalized banded DTW distance; lower means more similar."""
    if type(window) is not int or window < 0:
        raise ValueError("DTW window must be a non-negative integer")
    validate_sequence(first)
    validate_sequence(second)
    n, m = len(first), len(second)
    band = max(window, abs(n - m))
    previous = [math.inf] * (m + 1)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = [math.inf] * (m + 1)
        for j in range(max(1, i - band), min(m, i + band) + 1):
            cost = math.sqrt(sum((a - b) ** 2 for a, b in zip(first[i - 1], second[j - 1], strict=True))
                             / FEATURES_PER_FRAME)
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    # Typical path length lies near the longer sequence; using max avoids
    # making long recordings automatically less similar than short ones.
    return previous[m] / max(n, m)


def _resample(frames: tuple[tuple[float, ...], ...], count: int = 12) -> np.ndarray:
    source = np.asarray(frames, dtype=np.float64)
    source_positions = np.linspace(0.0, 1.0, len(source))
    target_positions = np.linspace(0.0, 1.0, count)
    return np.stack([np.interp(target_positions, source_positions, source[:, col])
                     for col in range(source.shape[1])], axis=1)


def _groups(examples: tuple[_LabeledSequence, ...]) -> np.ndarray:
    embedded = np.stack([_resample(item.frames).reshape(-1) for item in examples])
    model = NearestNeighbors(radius=DUPLICATE_RMS_TOLERANCE * math.sqrt(embedded.shape[1]),
                             algorithm="brute", metric="euclidean", n_jobs=1).fit(embedded)
    adjacent = model.radius_neighbors(embedded, return_distance=False)
    parents = list(range(len(examples)))
    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value
    for i, neighbors in enumerate(adjacent):
        for neighbor in neighbors:
            j = int(neighbor)
            if j > i:
                parents[find(j)] = find(i)
    roots: dict[int, int] = {}
    return np.asarray([roots.setdefault(find(i), len(roots)) for i in range(len(examples))])


def _predict_label(labels: tuple[DynamicGestureLabel, ...], examples: tuple[_LabeledSequence, ...],
                   frames: tuple[tuple[float, ...], ...]) -> int:
    scores = [math.inf] * len(labels)
    for example in examples:
        scores[example.label_index] = min(scores[example.label_index], dtw_distance(frames, example.frames))
    return int(np.argmin(scores))


class DynamicGestureTrainer:
    """Train a nearest-example DTW model and evaluate on grouped sequences."""

    def __init__(self, *, max_distance_threshold: float = 0.35,
                 minimum_separation_threshold: float = 0.015) -> None:
        if (not math.isfinite(max_distance_threshold) or max_distance_threshold < 0
                or not math.isfinite(minimum_separation_threshold)
                or minimum_separation_threshold < 0):
            raise ValueError("Dynamic rejection thresholds must be finite and non-negative")
        self.max_distance_threshold = max_distance_threshold
        self.minimum_separation_threshold = minimum_separation_threshold

    def train(self, datasets: tuple[DynamicGestureDataset, ...]) -> DynamicGestureModel:
        if len(datasets) < 2:
            raise ValueError("Create at least two dynamic gestures before training")
        ordered = tuple(sorted(datasets, key=lambda item: (item.name.casefold(), item.id)))
        labels = tuple(DynamicGestureLabel(item.id, item.name) for item in ordered)
        examples: list[_LabeledSequence] = []
        counts: list[int] = []
        for index, dataset in enumerate(ordered):
            validate_dynamic_dataset(dataset)
            if len(dataset.examples) < 3:
                raise ValueError(f"{dataset.name!r} needs at least 3 completed sequences")
            counts.append(len(dataset.examples))
            examples.extend(_LabeledSequence(index, item.frames) for item in dataset.examples)
        if len(examples) > 2000:
            raise ValueError("Dynamic training is limited to 2000 sequences")
        rows = tuple(examples)
        y = np.asarray([item.label_index for item in rows], dtype=np.int64)
        groups = _groups(rows)
        group_counts = tuple(len(set(groups[y == i])) for i in range(len(labels)))
        warnings: list[str] = []
        if any(value < 5 for value in counts):
            warnings.append("Each dynamic gesture should have at least 5 independent sequences.")
        if any(value < 5 for value in group_counts):
            warnings.append("Near-duplicate takes reduce independent validation examples.")
        split: tuple[np.ndarray, np.ndarray] | None = None
        expected = set(range(len(labels)))
        splitter = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=2026)
        for train_idx, test_idx in splitter.split(np.zeros(len(rows)), y, groups):
            if set(y[train_idx]) == expected and set(y[test_idx]) == expected:
                split = (train_idx, test_idx)
                break
        confusion = None
        accuracy = None
        evaluated = split is not None
        if not evaluated:
            warnings.append("Could not form a leakage-safe holdout containing every dynamic gesture.")
        else:
            train_idx, test_idx = split
            matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
            train_rows = tuple(rows[int(i)] for i in train_idx)
            for index in test_idx:
                actual = int(y[index])
                predicted = _predict_label(labels, train_rows, rows[int(index)].frames)
                matrix[actual, predicted] += 1
            confusion = tuple(tuple(int(x) for x in row) for row in matrix)
            accuracy = float(np.trace(matrix) / max(1, matrix.sum()))
        reliable = (evaluated and all(count >= 5 for count in counts)
                    and all(count >= 5 for count in group_counts))
        evaluation = DynamicEvaluation(evaluated, reliable, accuracy, confusion,
            tuple(label.name for label in labels), tuple(warnings), tuple(counts), group_counts)
        model = DynamicGestureModel(
            labels, rows, datetime.now(timezone.utc).isoformat(), evaluation,
            self.max_distance_threshold, self.minimum_separation_threshold)
        validate_model(model)
        return model


class DynamicGestureModelStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def save(self, model: DynamicGestureModel) -> None:
        validate_model(model)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                    prefix=f".{self.path.name}.", suffix=".tmp", delete=False) as stream:
                temp = Path(stream.name)
                json.dump(model.to_dict(), stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self.path)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)

    def load(self) -> DynamicGestureModel | None:
        if not self.path.exists():
            return None
        if self.path.stat().st_size > 100 * 1024 * 1024:
            raise ValueError("Dynamic model exceeds the supported file size")
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Dynamic model file is invalid: {exc}") from exc
        return DynamicGestureModel.from_dict(raw)


def validate_model(model: DynamicGestureModel) -> None:
    if not 2 <= len(model.labels) <= 64:
        raise ValueError("Dynamic model must contain 2 to 64 labels")
    ids = [item.dataset_id for item in model.labels]
    names = [item.name.casefold() for item in model.labels]
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("Dynamic model label mapping is duplicated")
    for item in model.labels:
        try:
            UUID(item.dataset_id)
        except (ValueError, TypeError) as exc:
            raise ValueError("Dynamic model label ID is invalid") from exc
        validate_name(item.name)
    if not model.examples or any(item.label_index < 0 or item.label_index >= len(model.labels)
                                 for item in model.examples):
        raise ValueError("Dynamic model training examples have invalid labels")
    if len(model.examples) > 2000:
        raise ValueError("Dynamic model contains too many training sequences")
    if any(not any(example.label_index == index for example in model.examples)
           for index in range(len(model.labels))):
        raise ValueError("Every dynamic model label needs training examples")
    for item in model.examples:
        validate_sequence(item.frames)
    try:
        datetime.fromisoformat(model.trained_at)
    except ValueError as exc:
        raise ValueError("Dynamic model training time is invalid") from exc
    if len(model.evaluation.example_counts) != len(model.labels):
        raise ValueError("Dynamic evaluation metadata does not match labels")
    if (len(model.evaluation.independent_group_counts) != len(model.labels)
            or len(model.evaluation.labels) != len(model.labels)
            or any(count < 0 for count in (*model.evaluation.example_counts,
                                           *model.evaluation.independent_group_counts))):
        raise ValueError("Dynamic evaluation class metadata is invalid")
    if any(not math.isfinite(value) or value < 0 for value in (
            model.max_distance_threshold, model.minimum_separation_threshold)):
        raise ValueError("Dynamic model thresholds must be finite and non-negative")
    if model.evaluation.accuracy is not None and (
            not math.isfinite(model.evaluation.accuracy)
            or not 0 <= model.evaluation.accuracy <= 1):
        raise ValueError("Dynamic evaluation accuracy is invalid")
    if model.evaluation.confusion is not None and (
            len(model.evaluation.confusion) != len(model.labels)
            or any(len(row) != len(model.labels) for row in model.evaluation.confusion)):
        raise ValueError("Dynamic confusion matrix dimensions do not match labels")


def evaluation_to_dict(evaluation: DynamicEvaluation) -> dict[str, Any]:
    return {"evaluated": evaluation.evaluated, "reliable": evaluation.reliable,
            "accuracy": evaluation.accuracy,
            "confusion": [list(row) for row in evaluation.confusion] if evaluation.confusion else None,
            "labels": list(evaluation.labels), "warnings": list(evaluation.warnings),
            "example_counts": list(evaluation.example_counts),
            "independent_group_counts": list(evaluation.independent_group_counts)}


def evaluation_from_dict(raw: Any) -> DynamicEvaluation:
    if not isinstance(raw, dict):
        raise ValueError("Dynamic evaluation metadata must be an object")
    confusion_raw = raw.get("confusion")
    return DynamicEvaluation(bool(raw["evaluated"]), bool(raw["reliable"]),
        None if raw["accuracy"] is None else float(raw["accuracy"]),
        None if confusion_raw is None else tuple(tuple(int(x) for x in row) for row in confusion_raw),
        tuple(str(x) for x in raw["labels"]), tuple(str(x) for x in raw["warnings"]),
        tuple(int(x) for x in raw["example_counts"]),
        tuple(int(x) for x in raw["independent_group_counts"]))
