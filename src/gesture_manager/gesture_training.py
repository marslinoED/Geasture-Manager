"""Local custom-label training, grouped evaluation, and safe JSON model files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
from uuid import UUID

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

from gesture_manager.gesture_data import (
    FEATURES_PER_HAND,
    FEATURE_VERSION,
    GestureDataset,
    dataset_to_dict,
    validate_dataset,
)


MODEL_FORMAT = "gesture-manager-svc-model"
MODEL_VERSION = 1
CLASSIFIER_VERSION = "portable-rbf-svc-v1"
CLASSIFIER_TYPE = "scikit-learn SVC (RBF)"
MAX_MODEL_BYTES = 50 * 1024 * 1024
MAX_TRAINING_EXAMPLES = 10_000
NEAR_DUPLICATE_RMS_TOLERANCE = 0.015


@dataclass(frozen=True)
class GestureLabel:
    dataset_id: str
    name: str


@dataclass(frozen=True)
class ClassifierPrediction:
    label: GestureLabel
    decision_margin: float
    vote_share: float
    vote_tie: bool


@dataclass(frozen=True)
class GestureEvaluation:
    label: GestureLabel
    support: int
    correct: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class EvaluationReport:
    evaluated: bool
    reliable: bool
    accuracy: float | None
    macro_f1: float | None
    classes: tuple[GestureEvaluation, ...]
    confusion: tuple[tuple[int, ...], ...] | None
    confusion_labels: tuple[str, ...]
    warnings: tuple[str, ...]
    sample_counts: tuple[int, ...]
    independent_group_counts: tuple[int, ...]
    validation_counts: tuple[int, ...] = ()


@dataclass(frozen=True)
class PortableGestureModel:
    labels: tuple[GestureLabel, ...]
    feature_version: str
    feature_size: int
    sklearn_version: str
    trained_at: str
    dataset_fingerprints: tuple[tuple[str, str], ...]
    training_sample_count: int
    training_group_count: int
    label_training_counts: tuple[int, ...]
    label_group_counts: tuple[int, ...]
    evaluation_reliable: bool
    evaluation_accuracy: float | None
    evaluation_macro_f1: float | None
    evaluation_warnings: tuple[str, ...]
    support_vectors: tuple[tuple[float, ...], ...]
    dual_coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    support_counts: tuple[int, ...]
    gamma: float

    def predict_one(self, features: tuple[float, ...] | list[float]) -> GestureLabel:
        return self.predict_with_scores(features).label

    def predict_with_scores(
        self, features: tuple[float, ...] | list[float], *,
        enabled_label_ids: set[str] | frozenset[str] | None = None,
    ) -> ClassifierPrediction:
        """Return the SVC label and raw pairwise decision diagnostics.

        ``decision_margin`` is an uncalibrated difference of aggregated OVO
        decision values. It is not a probability or a comparable score across
        separately trained models.
        """
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 1 or values.size != self.feature_size:
            raise ValueError(f"Expected exactly {self.feature_size} numeric features")
        if not np.isfinite(values).all() or np.abs(values).max(initial=0.0) > 10:
            raise ValueError("Live feature values are invalid")
        enabled_ids = ({label.dataset_id for label in self.labels}
                       if enabled_label_ids is None else set(enabled_label_ids))
        unknown_ids = enabled_ids - {label.dataset_id for label in self.labels}
        if unknown_ids:
            raise ValueError("Enabled gesture set contains an unknown label ID")
        enabled_indices = [index for index, label in enumerate(self.labels)
                           if label.dataset_id in enabled_ids]
        if len(enabled_indices) < 2:
            raise ValueError("At least two gestures must be enabled for live classification")
        support = np.asarray(self.support_vectors, dtype=np.float64)
        coefficients = np.asarray(self.dual_coefficients, dtype=np.float64)
        kernels = np.exp(-self.gamma * np.square(support - values).sum(axis=1))
        offsets = np.cumsum((0, *self.support_counts))
        votes = np.zeros(len(self.labels), dtype=np.int32)
        evidence = np.zeros(len(self.labels), dtype=np.float64)
        pair_index = 0
        for first in range(len(self.labels)):
            for second in range(first + 1, len(self.labels)):
                if first not in enabled_indices or second not in enabled_indices:
                    pair_index += 1
                    continue
                decision = float(
                    np.dot(coefficients[second - 1, offsets[first]:offsets[first + 1]],
                           kernels[offsets[first]:offsets[first + 1]])
                    + np.dot(coefficients[first, offsets[second]:offsets[second + 1]],
                             kernels[offsets[second]:offsets[second + 1]])
                    + self.intercepts[pair_index]
                )
                # LIBSVM's raw binary decision sign favors class 0 when
                # positive, while sklearn intentionally flips the binary
                # decision function so positive favors classes_[1]. The
                # multiclass OVO path retains LIBSVM's pair orientation.
                # Match sklearn.predict in the portable live path.
                if len(self.labels) == 2:
                    decision = -decision
                evidence[first] += decision
                evidence[second] -= decision
                votes[first if decision > 0 else second] += 1
                pair_index += 1
        active_votes = votes[enabled_indices]
        winner = enabled_indices[int(np.argmax(active_votes))]
        runner_up = enabled_indices[int(np.argsort(active_votes)[-2])]
        pair_count = len(enabled_indices) * (len(enabled_indices) - 1) // 2
        return ClassifierPrediction(
            label=self.labels[winner],
            decision_margin=float(evidence[winner] - evidence[runner_up]),
            vote_share=float(votes[winner] / max(1, pair_count)),
            vote_tie=int(np.sum(active_votes == votes[winner])) > 1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": MODEL_FORMAT,
            "model_version": MODEL_VERSION,
            "classifier_type": "svc-rbf",
            "classifier_version": CLASSIFIER_VERSION,
            "feature_version": self.feature_version,
            "feature_size": self.feature_size,
            "sklearn_version": self.sklearn_version,
            "trained_at": self.trained_at,
            "training_metadata": {
                "sample_count": self.training_sample_count,
                "independent_group_count": self.training_group_count,
                "label_training_counts": list(self.label_training_counts),
                "label_group_counts": list(self.label_group_counts),
                "evaluation_reliable": self.evaluation_reliable,
                "evaluation_accuracy": self.evaluation_accuracy,
                "evaluation_macro_f1": self.evaluation_macro_f1,
                "warnings": list(self.evaluation_warnings),
            },
            "labels": [{"dataset_id": label.dataset_id, "name": label.name} for label in self.labels],
            "dataset_fingerprints": dict(self.dataset_fingerprints),
            "svc": {
                "support_vectors": [list(row) for row in self.support_vectors],
                "dual_coefficients": [list(row) for row in self.dual_coefficients],
                "intercepts": list(self.intercepts),
                "support_counts": list(self.support_counts),
                "gamma": self.gamma,
            },
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "PortableGestureModel":
        if not isinstance(raw, dict) or raw.get("format") != MODEL_FORMAT:
            raise ValueError("Unrecognized custom gesture model format")
        if type(raw.get("model_version")) is not int or raw.get("model_version") != MODEL_VERSION:
            raise ValueError(f"Unsupported model version: {raw.get('model_version')}")
        if raw.get("classifier_type") != "svc-rbf" or raw.get("classifier_version") != CLASSIFIER_VERSION:
            raise ValueError("Unsupported classifier format")
        if raw.get("feature_version") != FEATURE_VERSION:
            raise ValueError(f"Incompatible feature version: {raw.get('feature_version')}")
        try:
            feature_size = int(raw["feature_size"])
            raw_labels = raw["labels"]
            metadata = raw["training_metadata"]
            if (not isinstance(metadata, dict)
                    or type(metadata.get("evaluation_reliable")) is not bool):
                raise ValueError("Model reliability metadata must be boolean")
            if feature_size not in (FEATURES_PER_HAND, 2 * FEATURES_PER_HAND):
                raise ValueError("Unsupported model feature size")
            labels = tuple(GestureLabel(str(item["dataset_id"]), str(item["name"])) for item in raw_labels)
            svc = raw["svc"]
            model = cls(
                labels=labels,
                feature_version=str(raw["feature_version"]),
                feature_size=feature_size,
                sklearn_version=str(raw["sklearn_version"]),
                trained_at=str(raw["trained_at"]),
                dataset_fingerprints=tuple(sorted((str(key), str(value)) for key, value in raw["dataset_fingerprints"].items())),
                training_sample_count=int(metadata["sample_count"]),
                training_group_count=int(metadata["independent_group_count"]),
                label_training_counts=tuple(int(value) for value in metadata["label_training_counts"]),
                label_group_counts=tuple(int(value) for value in metadata["label_group_counts"]),
                evaluation_reliable=metadata["evaluation_reliable"],
                evaluation_accuracy=(None if metadata["evaluation_accuracy"] is None else float(metadata["evaluation_accuracy"])),
                evaluation_macro_f1=(None if metadata["evaluation_macro_f1"] is None else float(metadata["evaluation_macro_f1"])),
                evaluation_warnings=tuple(str(item) for item in metadata["warnings"]),
                support_vectors=tuple(tuple(float(value) for value in row) for row in svc["support_vectors"]),
                dual_coefficients=tuple(tuple(float(value) for value in row) for row in svc["dual_coefficients"]),
                intercepts=tuple(float(value) for value in svc["intercepts"]),
                support_counts=tuple(int(value) for value in svc["support_counts"]),
                gamma=float(svc["gamma"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
            raise ValueError(f"Model fields are invalid: {exc}") from exc
        validate_model(model)
        return model


@dataclass(frozen=True)
class TrainingOutcome:
    model: PortableGestureModel
    report: EvaluationReport
    model_path: Path
    classifier_version: str = CLASSIFIER_VERSION


def validate_model(model: PortableGestureModel) -> None:
    class_count = len(model.labels)
    if class_count < 2 or class_count > 64:
        raise ValueError("A model must contain 2 to 64 custom gesture labels")
    ids = [label.dataset_id for label in model.labels]
    names = [label.name.casefold() for label in model.labels]
    try:
        for label in model.labels:
            UUID(label.dataset_id)
            if not label.name.strip() or len(label.name) > 40:
                raise ValueError("Model label name is invalid")
    except (ValueError, TypeError) as exc:
        raise ValueError("Model label mapping is invalid") from exc
    if len(set(ids)) != class_count or len(set(names)) != class_count:
        raise ValueError("Model label mapping contains duplicate labels")
    if not model.feature_version or not model.sklearn_version:
        raise ValueError("Model version metadata is missing")
    if model.training_sample_count < 1 or model.training_group_count < 1:
        raise ValueError("Model training metadata is invalid")
    if (len(model.label_training_counts) != class_count
            or len(model.label_group_counts) != class_count
            or any(count < 1 for count in (*model.label_training_counts, *model.label_group_counts))):
        raise ValueError("Per-gesture training metadata is invalid")
    if sum(model.label_training_counts) != model.training_sample_count:
        raise ValueError("Total and per-gesture training counts do not match")
    for metric in (model.evaluation_accuracy, model.evaluation_macro_f1):
        if metric is not None and (not math.isfinite(metric) or not 0 <= metric <= 1):
            raise ValueError("Model evaluation metadata is invalid")
    if len(model.support_counts) != class_count or any(count < 1 for count in model.support_counts):
        raise ValueError("Model support-vector counts are invalid")
    support_total = sum(model.support_counts)
    if support_total > MAX_TRAINING_EXAMPLES:
        raise ValueError("Model contains too many support vectors")
    if len(model.support_vectors) != support_total or any(len(row) != model.feature_size for row in model.support_vectors):
        raise ValueError("Model support-vector dimensions are invalid")
    expected_coeff_rows = class_count - 1
    if len(model.dual_coefficients) != expected_coeff_rows or any(len(row) != support_total for row in model.dual_coefficients):
        raise ValueError("Model coefficient dimensions are invalid")
    if len(model.intercepts) != class_count * (class_count - 1) // 2:
        raise ValueError("Model intercept dimensions are invalid")
    numeric = [value for rows in (model.support_vectors, model.dual_coefficients)
               for row in rows for value in row]
    numeric.extend((*model.intercepts, model.gamma))
    if not all(math.isfinite(value) for value in numeric) or any(abs(value) > 1e6 for value in numeric):
        raise ValueError("Model contains invalid numeric parameters")
    if any(abs(value) > 10 for row in model.support_vectors for value in row):
        raise ValueError("Model support vector contains out-of-range feature values")
    if model.gamma <= 0:
        raise ValueError("Model gamma must be positive")
    try:
        datetime.fromisoformat(model.trained_at)
    except ValueError as exc:
        raise ValueError("Model training timestamp is invalid") from exc
    if any(len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint)
           for _dataset_id, fingerprint in model.dataset_fingerprints):
        raise ValueError("Dataset fingerprint metadata is invalid")
    if {dataset_id for dataset_id, _fingerprint in model.dataset_fingerprints} != set(ids):
        raise ValueError("Saved dataset fingerprints do not match the label mapping")


class GestureModelStore:
    """Persist a validated, data-only model. Never unpickle estimator files."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def save(self, model: PortableGestureModel) -> None:
        validate_model(model)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(model.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                             prefix=f".{self.path.name}.", suffix=".tmp", delete=False) as stream:
                temp_path = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def load(self) -> PortableGestureModel | None:
        if not self.path.exists():
            return None
        if self.path.stat().st_size > MAX_MODEL_BYTES:
            raise ValueError("Model file is larger than the supported limit")
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("Saved model file is not valid JSON") from exc
        return PortableGestureModel.from_dict(raw)

    @staticmethod
    def path_from_gesture_directory(gesture_directory: Path) -> Path:
        return Path(gesture_directory).parent / "models" / "custom_gestures.json"


class GestureTrainer:
    """Train custom gesture labels; evaluation holds near-duplicate groups out."""

    def __init__(self, model_store: GestureModelStore) -> None:
        self.model_store = model_store

    def train(self, datasets: tuple[GestureDataset, ...]) -> TrainingOutcome:
        x, y, groups, labels, counts = make_training_data(datasets)
        group_counts = tuple(len(set(groups[y == index])) for index in range(len(labels)))
        warnings: list[str] = []
        if any(count < 5 for count in counts):
            warnings.append("Some gestures have fewer than 5 examples; validation estimates will be unreliable.")
        if any(count < 3 for count in group_counts):
            warnings.append("Some gestures have fewer than 3 independent pose groups; validation is weak.")
        group_labels: dict[int, set[int]] = {}
        for group, label in zip(groups, y, strict=True):
            group_labels.setdefault(int(group), set()).add(int(label))
        if any(len(group_classes) > 1 for group_classes in group_labels.values()):
            warnings.append("Near-identical feature samples have different custom labels; those labels may be visually indistinguishable.")

        split = grouped_validation_split(x, y, groups)
        report: EvaluationReport
        if split is None:
            warnings.append("Could not make a leakage-safe holdout containing every gesture; no validation metrics are reported.")
            report = EvaluationReport(False, False, None, None, (), None,
                                      tuple(label.name for label in labels), tuple(warnings), counts, group_counts)
        else:
            train_indices, validation_indices = split
            validation_counts = tuple(int(np.sum(y[validation_indices] == index)) for index in range(len(labels)))
            validation_group_counts = tuple(
                len(set(groups[validation_indices][y[validation_indices] == index]))
                for index in range(len(labels))
            )
            try:
                evaluation_model = _new_svc()
                evaluation_model.fit(x[train_indices], y[train_indices])
                predictions = evaluation_model.predict(x[validation_indices])
            except Exception as exc:
                raise ValueError(f"Grouped validation could not train this dataset: {exc}") from exc
            cm = confusion_matrix(y[validation_indices], predictions,
                                  labels=np.arange(len(labels)))
            precision, recall, f1, support = precision_recall_fscore_support(
                y[validation_indices], predictions, labels=np.arange(len(labels)),
                zero_division=0,
            )
            per_class = tuple(GestureEvaluation(
                label=labels[index], support=int(support[index]),
                correct=int(cm[index, index]), precision=float(precision[index]),
                recall=float(recall[index]), f1=float(f1[index]),
            ) for index in range(len(labels)))
            reliability = (
                all(count >= 5 for count in counts)
                and all(count >= 5 for count in group_counts)
                and all(count >= 2 for count in validation_counts)
                and all(count >= 2 for count in validation_group_counts)
            )
            if not reliability:
                warnings.append("Holdout metrics are preliminary: each gesture needs more independent examples for a dependable estimate.")
            report = EvaluationReport(
                evaluated=True, reliable=reliability,
                accuracy=float(np.trace(cm) / max(1, cm.sum())),
                macro_f1=float(np.mean(f1)), classes=per_class,
                confusion=tuple(tuple(int(value) for value in row) for row in cm),
                confusion_labels=tuple(label.name for label in labels),
                warnings=tuple(warnings), sample_counts=counts,
                independent_group_counts=group_counts,
                validation_counts=validation_counts,
            )

        # The saved estimator learns from all validated user examples. Its
        # evaluation report above is from a disjoint group holdout.
        final_model = _new_svc()
        final_model.fit(x, y)
        portable = _export_model(final_model, labels, datasets, x.shape[1], report)
        self.model_store.save(portable)
        return TrainingOutcome(portable, report, self.model_store.path)

    def load_saved(self) -> PortableGestureModel | None:
        return self.model_store.load()


def make_training_data(
    datasets: tuple[GestureDataset, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[GestureLabel, ...], tuple[int, ...]]:
    if len(datasets) < 2:
        raise ValueError("Create at least two custom gestures before training")
    ordered = tuple(sorted(datasets, key=lambda item: (item.name.casefold(), item.id)))
    feature_sizes = {item.hand_count * FEATURES_PER_HAND for item in ordered}
    if len(feature_sizes) != 1:
        raise ValueError("All gestures in one model must use the same one-hand or two-hand configuration")
    labels = tuple(GestureLabel(item.id, item.name) for item in ordered)
    if len({label.name.casefold() for label in labels}) != len(labels):
        raise ValueError("Custom gesture names must be unique")
    features: list[tuple[float, ...]] = []
    label_indices: list[int] = []
    example_ids: list[str] = []
    sample_counts: list[int] = []
    for label_index, dataset in enumerate(ordered):
        validate_dataset(dataset)
        if len(dataset.examples) < 3:
            raise ValueError(f"{dataset.name!r} needs at least 3 valid examples before training")
        sample_counts.append(len(dataset.examples))
        for example in dataset.examples:
            features.append(example.features)
            label_indices.append(label_index)
            example_ids.append(example.id)
    if len(features) > MAX_TRAINING_EXAMPLES:
        raise ValueError(f"Training is limited to {MAX_TRAINING_EXAMPLES} examples per model")
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or np.abs(x).max(initial=0.0) > 10:
        raise ValueError("Feature samples are invalid or outside the supported range")
    if any(len(np.unique(x[np.asarray(label_indices) == index], axis=0)) < 2
           for index in range(len(labels))):
        raise ValueError("Each gesture needs at least two different feature examples")
    y = np.asarray(label_indices, dtype=np.int64)
    groups = group_near_duplicates(x)
    return x, y, groups, labels, tuple(sample_counts)


def group_near_duplicates(
    x: np.ndarray, *, rms_tolerance: float = NEAR_DUPLICATE_RMS_TOLERANCE,
) -> np.ndarray:
    """Connect samples within RMS feature distance; connected rows share a fold."""
    count, dimensions = x.shape
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        a, b = find(first), find(second)
        if a != b:
            parent[b] = a

    radius = rms_tolerance * math.sqrt(dimensions)
    neighbors = NearestNeighbors(radius=radius, algorithm="brute", metric="euclidean", n_jobs=1)
    neighbors.fit(x)
    lists = neighbors.radius_neighbors(x, return_distance=False)
    for index, adjacent in enumerate(lists):
        for neighbor in adjacent:
            if int(neighbor) > index:
                union(index, int(neighbor))
    roots: dict[int, int] = {}
    return np.asarray([roots.setdefault(find(index), len(roots)) for index in range(count)], dtype=np.int64)


def grouped_validation_split(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    class_count = len(np.unique(y))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        cv = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=2026)
        folds.extend(cv.split(x, y, groups))
    except ValueError:
        pass
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    expected = set(range(class_count))
    for train, validation in folds:
        if set(y[train]) == expected and set(y[validation]) == expected:
            candidates.append((train, validation))
    if not candidates:
        splitter = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=2026)
        for train, validation in splitter.split(x, y, groups):
            if set(y[train]) == expected and set(y[validation]) == expected:
                candidates.append((train, validation))
    if not candidates:
        return None
    return min(candidates, key=lambda pair: abs(len(pair[1]) / len(y) - 0.25))


def _new_svc() -> SVC:
    return SVC(C=10.0, kernel="rbf", gamma="scale", class_weight="balanced",
               decision_function_shape="ovo", cache_size=64)


def _export_model(
    estimator: SVC,
    labels: tuple[GestureLabel, ...],
    datasets: tuple[GestureDataset, ...],
    feature_size: int,
    report: EvaluationReport,
) -> PortableGestureModel:
    return PortableGestureModel(
        labels=labels, feature_version=FEATURE_VERSION, feature_size=feature_size,
        sklearn_version=sklearn.__version__, trained_at=datetime.now(timezone.utc).isoformat(),
        dataset_fingerprints=compute_dataset_fingerprints(datasets),
        training_sample_count=sum(len(dataset.examples) for dataset in datasets),
        training_group_count=sum(report.independent_group_counts),
        label_training_counts=tuple(len(dataset.examples) for dataset in sorted(datasets, key=lambda item: (item.name.casefold(), item.id))),
        label_group_counts=report.independent_group_counts,
        evaluation_reliable=report.reliable,
        evaluation_accuracy=report.accuracy,
        evaluation_macro_f1=report.macro_f1,
        evaluation_warnings=report.warnings,
        support_vectors=tuple(tuple(float(value) for value in row) for row in estimator.support_vectors_),
        dual_coefficients=tuple(tuple(float(value) for value in row) for row in estimator.dual_coef_),
        intercepts=tuple(float(value) for value in estimator.intercept_),
        support_counts=tuple(int(value) for value in estimator.n_support_),
        gamma=float(estimator._gamma),
    )


def compute_dataset_fingerprints(
    datasets: tuple[GestureDataset, ...],
) -> tuple[tuple[str, str], ...]:
    result = []
    for dataset in sorted(datasets, key=lambda item: item.id):
        encoded = json.dumps(dataset_to_dict(dataset), ensure_ascii=False,
                             sort_keys=True, separators=(",", ":")).encode("utf-8")
        result.append((dataset.id, hashlib.sha256(encoded).hexdigest()))
    return tuple(result)


def model_matches_datasets(
    model: PortableGestureModel, datasets: tuple[GestureDataset, ...],
) -> bool:
    return model.dataset_fingerprints == compute_dataset_fingerprints(datasets)


class LiveClassifierWorker:
    """Classify only new camera-engine outputs on a dedicated worker thread."""

    def __init__(
        self,
        model: PortableGestureModel,
        status_provider: Callable[[], Any],
        output_queue: Any,
    ) -> None:
        from threading import Event
        self.model = model
        self.status_provider = status_provider
        self.output_queue = output_queue
        self._stop = Event()
        self._thread = None
        initial_status = status_provider()
        self._last_sequence = getattr(initial_status, "processed_frames", -1)

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        from threading import Thread
        if self.is_alive:
            return False
        self._stop.clear()
        self._thread = Thread(target=self._run, name="GestureManagerLiveClassifier", daemon=False)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()

    def wait(self, timeout: float | None = None) -> bool:
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop.wait(0.03):
            status = self.status_provider()
            sequence = getattr(status, "processed_frames", -1)
            if sequence <= self._last_sequence:
                continue
            self._last_sequence = sequence
            if getattr(status, "hands_detected", None) is not True:
                continue
            hands = getattr(status, "hands", ())
            if len(hands) * FEATURES_PER_HAND != self.model.feature_size:
                self.output_queue.put((sequence, None, "Live hand configuration does not match the saved model."))
                continue
            try:
                from gesture_manager.gesture_data import normalize_hand_features
                features = normalize_hand_features(tuple(hands))
                prediction = self.model.predict_one(features)
            except Exception as exc:
                self.output_queue.put((sequence, None, f"Live sample could not be classified: {exc}"))
            else:
                self.output_queue.put((sequence, prediction, None))
