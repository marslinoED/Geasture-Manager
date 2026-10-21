"""Versioned local storage and normalized feature extraction for static poses."""

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

from gesture_manager.hand_landmarks import DetectedHand


DATASET_FORMAT = "gesture-manager-dataset"
DATASET_VERSION = 1
FEATURE_VERSION = "hand-pose-normalized-v1"
FEATURES_PER_HAND = 63
GESTURE_TYPE = "static_pose"
MAX_DATASET_FILE_BYTES = 50 * 1024 * 1024


def default_gesture_directory() -> Path:
    """Return the per-user Windows data directory without an extra runtime dependency."""
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / "GestureManager" / "gestures"


@dataclass(frozen=True)
class TrainingExample:
    id: str
    captured_at: str
    features: tuple[float, ...]


@dataclass(frozen=True)
class GestureDataset:
    id: str
    name: str
    gesture_type: str
    hand_count: int
    examples: tuple[TrainingExample, ...] = ()
    format_version: int = DATASET_VERSION
    feature_version: str = FEATURE_VERSION

    @property
    def is_trainable(self) -> bool:
        return self.hand_count in (1, 2) and len(self.examples) >= 3 and all(
            len(example.features) == self.hand_count * FEATURES_PER_HAND
            and all(math.isfinite(value) for value in example.features)
            for example in self.examples
        )


def create_draft(name: str, hand_count: int) -> GestureDataset:
    normalized_name = validate_name(name)
    if hand_count not in (1, 2):
        raise ValueError("Choose a supported hand configuration: one or two hands")
    return GestureDataset(id=str(uuid4()), name=normalized_name,
                          gesture_type=GESTURE_TYPE, hand_count=hand_count)


def validate_name(name: str) -> str:
    normalized = name.strip()
    if not normalized or len(normalized) > 40:
        raise ValueError("Gesture name must contain 1 to 40 characters")
    if any(ord(character) < 32 for character in normalized):
        raise ValueError("Gesture name cannot contain control characters")
    return normalized


def normalize_hand_features(hands: tuple[DetectedHand, ...]) -> tuple[float, ...]:
    """Normalize one or two hands for position, scale, and in-plane rotation.

    Wrist (landmark 0) is the origin for one hand; the midpoint between wrists
    is the origin for two. Average wrist-to-middle-MCP length sets scale, and
    the average wrist-to-middle-MCP vector defines the rotation frame. Two-hand
    inputs are ordered by wrist x position. The resulting 63 values per hand
    are local numeric features; no source image is retained.
    """
    if len(hands) not in (1, 2):
        raise ValueError("A recording example must contain exactly one or two hands")
    if any(len(hand.landmarks) != 21 for hand in hands):
        raise ValueError("Each hand must contain exactly 21 landmarks")
    ordered = tuple(sorted(hands, key=lambda hand: hand.landmarks[0].x))
    xy = [[(point.x, point.y, point.z) for point in hand.landmarks] for hand in ordered]
    if not all(math.isfinite(value) for hand in xy for point in hand for value in point):
        raise ValueError("Landmarks must contain only finite coordinates")
    origin_x = sum(hand[0][0] for hand in xy) / len(xy)
    origin_y = sum(hand[0][1] for hand in xy) / len(xy)
    vectors = [(hand[9][0] - hand[0][0], hand[9][1] - hand[0][1]) for hand in xy]
    scale = sum(math.hypot(x, y) for x, y in vectors) / len(vectors)
    if scale < 1e-6:
        raise ValueError("Hand landmarks are too close together to normalize")
    axis_x = sum(vector[0] for vector in vectors)
    axis_y = sum(vector[1] for vector in vectors)
    if math.hypot(axis_x, axis_y) < 1e-6:
        raise ValueError("Hand orientation is ambiguous for this example")
    angle = math.atan2(axis_y, axis_x)
    cosine, sine = math.cos(angle), math.sin(angle)
    output: list[float] = []
    for hand in xy:
        for x, y, z in hand:
            dx, dy = x - origin_x, y - origin_y
            output.extend(((dx * cosine + dy * sine) / scale,
                           (-dx * sine + dy * cosine) / scale,
                           z / scale))
    return tuple(output)


class GestureRecordingSession:
    """In-memory draft. Cancelling an unfinished take never writes a dataset."""

    def __init__(self, dataset: GestureDataset) -> None:
        self.dataset = dataset
        self._baseline = dataset

    def add_hands(self, hands: tuple[DetectedHand, ...]) -> TrainingExample:
        if len(hands) != self.dataset.hand_count:
            raise ValueError(f"This gesture requires exactly {self.dataset.hand_count} hand(s)")
        example = TrainingExample(
            id=str(uuid4()),
            captured_at=datetime.now(timezone.utc).isoformat(),
            features=normalize_hand_features(hands),
        )
        self.dataset = replace(self.dataset, examples=(*self.dataset.examples, example))
        return example

    def remove_example(self, example_id: str) -> None:
        remaining = tuple(example for example in self.dataset.examples if example.id != example_id)
        if len(remaining) == len(self.dataset.examples):
            raise KeyError(f"Example not found: {example_id}")
        self.dataset = replace(self.dataset, examples=remaining)

    def cancel(self) -> None:
        self.dataset = self._baseline


class GestureStore:
    """Read and atomically write versioned feature-only JSON datasets."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def _ensure_root(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset_id: str) -> Path:
        try:
            canonical_id = str(UUID(dataset_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Invalid gesture identifier") from exc
        return self.root / f"{canonical_id}.json"

    def list(self) -> tuple[GestureDataset, ...]:
        self._ensure_root()
        datasets = [self.load(path.stem) for path in self.root.glob("*.json")]
        return tuple(sorted(datasets, key=lambda dataset: dataset.name.casefold()))

    def load(self, dataset_id: str) -> GestureDataset:
        self._ensure_root()
        path = self.path_for(dataset_id)
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Gesture dataset path must remain inside its storage directory")
        try:
            if path.stat().st_size > MAX_DATASET_FILE_BYTES:
                raise ValueError("Gesture dataset file is larger than the supported limit")
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Gesture dataset is not valid JSON: {path.name}") from exc
        return dataset_from_dict(raw)

    def save(self, dataset: GestureDataset) -> None:
        self._ensure_root()
        validate_dataset(dataset)
        path = self.path_for(dataset.id)
        for existing in self.list():
            if existing.id != dataset.id and existing.name.casefold() == dataset.name.casefold():
                raise ValueError(f"A gesture named {dataset.name!r} already exists")
        payload = json.dumps(dataset_to_dict(dataset), ensure_ascii=False, indent=2, allow_nan=False)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.root,
                                             prefix=f".{dataset.id}.", suffix=".tmp", delete=False) as stream:
                temp_path = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def delete(self, dataset_id: str) -> None:
        path = self.path_for(dataset_id)
        if path.is_symlink():
            path.unlink()
        else:
            path.resolve().relative_to(self.root)
            path.unlink()

    def rename(self, dataset_id: str, name: str) -> GestureDataset:
        updated = replace(self.load(dataset_id), name=validate_name(name))
        self.save(updated)
        return updated

    def remove_example(self, dataset_id: str, example_id: str) -> GestureDataset:
        dataset = self.load(dataset_id)
        session = GestureRecordingSession(dataset)
        session.remove_example(example_id)
        self.save(session.dataset)
        return session.dataset


def validate_dataset(dataset: GestureDataset) -> None:
    try:
        canonical_id = str(UUID(dataset.id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid gesture identifier") from exc
    if canonical_id != dataset.id.lower():
        raise ValueError("Gesture identifier must be a canonical UUID")
    validate_name(dataset.name)
    if dataset.format_version != DATASET_VERSION:
        raise ValueError(f"Unsupported dataset version: {dataset.format_version}")
    if dataset.feature_version != FEATURE_VERSION:
        raise ValueError(f"Unsupported feature version: {dataset.feature_version}")
    if dataset.gesture_type != GESTURE_TYPE:
        raise ValueError(f"Unsupported gesture type: {dataset.gesture_type}")
    if dataset.hand_count not in (1, 2):
        raise ValueError("Hand count must be one or two")
    example_ids: set[str] = set()
    expected = dataset.hand_count * FEATURES_PER_HAND
    for example in dataset.examples:
        try:
            UUID(example.id)
            timestamp = datetime.fromisoformat(example.captured_at)
            if timestamp.tzinfo is None:
                raise ValueError("timestamp needs a timezone")
        except (ValueError, TypeError) as exc:
            raise ValueError("Example ID or timestamp is invalid") from exc
        if example.id in example_ids:
            raise ValueError("Example identifiers must be unique")
        example_ids.add(example.id)
        if len(example.features) != expected:
            raise ValueError(f"Feature vector must contain exactly {expected} values")
        if not all(
            isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and abs(value) <= 10
            for value in example.features
        ):
            raise ValueError("Feature values must be finite and within [-10, 10]")


def dataset_to_dict(dataset: GestureDataset) -> dict[str, Any]:
    return {
        "format": DATASET_FORMAT,
        "format_version": dataset.format_version,
        "feature_extraction_version": dataset.feature_version,
        "id": dataset.id,
        "name": dataset.name,
        "gesture_type": dataset.gesture_type,
        "hand_count": dataset.hand_count,
        "examples": [
            {"id": example.id, "captured_at": example.captured_at,
             "features": list(example.features)}
            for example in dataset.examples
        ],
    }


def dataset_from_dict(raw: Any) -> GestureDataset:
    if not isinstance(raw, dict) or raw.get("format") != DATASET_FORMAT:
        raise ValueError("Unrecognized gesture dataset format")
    expected_fields = {"format", "format_version", "feature_extraction_version", "id",
                       "name", "gesture_type", "hand_count", "examples"}
    if set(raw) != expected_fields:
        raise ValueError("Gesture dataset fields are invalid")
    if (type(raw["format_version"]) is not int or
            type(raw["hand_count"]) is not int or
            not all(isinstance(raw[key], str) for key in
                    ("feature_extraction_version", "id", "name", "gesture_type"))):
        raise ValueError("Gesture dataset field types are invalid")
    try:
        examples_raw = raw["examples"]
        if not isinstance(examples_raw, list):
            raise TypeError("examples must be an array")
        examples_list = []
        for item in examples_raw:
            if (not isinstance(item, dict) or set(item) != {"id", "captured_at", "features"}
                    or not isinstance(item["id"], str)
                    or not isinstance(item["captured_at"], str)
                    or not isinstance(item["features"], list)
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           for value in item["features"])):
                raise ValueError("Gesture example field types are invalid")
            examples_list.append(TrainingExample(
                id=item["id"], captured_at=item["captured_at"],
                features=tuple(float(value) for value in item["features"]),
            ))
        examples = tuple(examples_list)
        dataset = GestureDataset(
            id=raw["id"], name=raw["name"],
            gesture_type=raw["gesture_type"], hand_count=raw["hand_count"],
            examples=examples, format_version=raw["format_version"],
            feature_version=raw["feature_extraction_version"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Gesture dataset fields are invalid: {exc}") from exc
    validate_dataset(dataset)
    return dataset
