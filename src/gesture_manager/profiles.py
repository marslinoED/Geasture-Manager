"""Versioned, local gesture profiles with safe, self-contained ZIP import/export."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any
from uuid import UUID, uuid4
import zipfile

from gesture_manager.actions import ActionMapping
from gesture_manager.gesture_data import (
    GestureDataset, dataset_from_dict, dataset_to_dict, validate_dataset, validate_name,
)
from gesture_manager.gesture_training import PortableGestureModel, validate_model


PROFILE_FORMAT = "gesture-manager-profile"
PROFILE_VERSION = 4
MAX_PROFILE_FILE_BYTES = 50 * 1024 * 1024
MAX_PROFILE_ARCHIVE_BYTES = 100 * 1024 * 1024
MAX_PROFILE_MEMBERS = 256
_UNSET = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_profile_directory() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / "GestureManager" / "profiles"


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Profile asset path must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("Profile asset path escapes its profile directory")
    return path.as_posix()


def validate_profile(raw: object) -> dict[str, Any]:
    """Validate structure and safe values; asset contents are checked separately."""
    base_fields = {
        "format", "version", "id", "name", "metadata", "gesture_definitions",
        "dataset_refs", "model_ref", "action_mappings", "thresholds",
    }
    execution_fields = {"dry_run", "stop_gesture_id"}
    control_fields = {"pause_gesture_id", "resume_gesture_id"}
    if not isinstance(raw, dict):
        raise ValueError("Profile schema fields are invalid")
    version = raw.get("version")
    if version == 1:
        if set(raw) != base_fields:
            raise ValueError("Profile schema fields are invalid")
        dry_run, stop_gesture_id = True, None
        pause_gesture_id, resume_gesture_id = None, None
        theme_mode = "light"
    elif version == 2:
        if set(raw) != base_fields | execution_fields:
            raise ValueError("Profile schema fields are invalid")
        dry_run, stop_gesture_id = raw["dry_run"], raw["stop_gesture_id"]
        pause_gesture_id, resume_gesture_id = None, None
        if type(dry_run) is not bool:
            raise ValueError("Profile Dry-run setting must be a boolean")
        theme_mode = "light"
    elif version == 3:
        if set(raw) != base_fields | execution_fields | {"theme_mode"}:
            raise ValueError("Profile schema fields are invalid")
        dry_run, stop_gesture_id = raw["dry_run"], raw["stop_gesture_id"]
        pause_gesture_id, resume_gesture_id = None, None
        theme_mode = raw["theme_mode"]
        if type(dry_run) is not bool:
            raise ValueError("Profile Dry-run setting must be a boolean")
        if theme_mode not in ("light", "dark"):
            raise ValueError("Profile theme mode must be light or dark")
    elif version == PROFILE_VERSION:
        if set(raw) != base_fields | execution_fields | control_fields | {"theme_mode"}:
            raise ValueError("Profile schema fields are invalid")
        dry_run, stop_gesture_id = raw["dry_run"], raw["stop_gesture_id"]
        pause_gesture_id, resume_gesture_id = raw["pause_gesture_id"], raw["resume_gesture_id"]
        theme_mode = raw["theme_mode"]
        if type(dry_run) is not bool:
            raise ValueError("Profile Dry-run setting must be a boolean")
        if theme_mode not in ("light", "dark"):
            raise ValueError("Profile theme mode must be light or dark")
    else:
        raise ValueError("Unsupported profile format or version")
    if raw["format"] != PROFILE_FORMAT or type(version) is not int:
        raise ValueError("Unsupported profile format or version")
    try:
        profile_id = str(UUID(raw["id"]))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Profile ID must be a UUID") from exc
    name = validate_name(raw["name"])
    if not isinstance(raw["metadata"], dict) or set(raw["metadata"]) != {"created_at", "updated_at"}:
        raise ValueError("Profile metadata is invalid")
    for value in raw["metadata"].values():
        if not isinstance(value, str) or len(value) > 80:
            raise ValueError("Profile timestamps are invalid")
    definitions = raw["gesture_definitions"]
    refs = raw["dataset_refs"]
    if not isinstance(definitions, list) or not isinstance(refs, dict) or len(definitions) > 1000:
        raise ValueError("Profile gesture definitions or dataset references are invalid")
    ids: set[str] = set()
    for definition in definitions:
        if not isinstance(definition, dict) or set(definition) != {"id", "name", "gesture_type", "hand_count"}:
            raise ValueError("Gesture definition fields are invalid")
        try:
            gesture_id = str(UUID(definition["id"]))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Gesture definition ID must be a UUID") from exc
        if gesture_id in ids:
            raise ValueError("Profile contains duplicate gesture IDs")
        ids.add(gesture_id)
        validate_name(definition["name"])
        if definition["gesture_type"] != "static_pose" or definition["hand_count"] not in (1, 2):
            raise ValueError("Profile gesture type or hand configuration is unsupported")
    if set(refs) != ids:
        raise ValueError("Every custom gesture must have exactly one dataset reference")
    control_ids = {
        "Stop Gesture": stop_gesture_id,
        "Pause Gesture": pause_gesture_id,
        "Resume Gesture": resume_gesture_id,
    }
    normalized_controls: dict[str, str] = {}
    for control_name, control_id in control_ids.items():
        if control_id is None:
            continue
        try:
            normalized_id = str(UUID(control_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"{control_name} ID must be a UUID") from exc
        if normalized_id != control_id or normalized_id not in ids:
            raise ValueError(f"{control_name} must reference a custom gesture in this profile")
        normalized_controls[control_name] = normalized_id
    if (normalized_controls.get("Stop Gesture") in {
            normalized_controls.get("Pause Gesture"),
            normalized_controls.get("Resume Gesture"),
    } and normalized_controls.get("Stop Gesture") is not None):
        raise ValueError("Stop Gesture cannot use the same gesture as Pause or Resume")
    for gesture_id, reference in refs.items():
        try:
            if str(UUID(gesture_id)) != gesture_id:
                raise ValueError
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Dataset reference ID is invalid") from exc
        expected = f"datasets/{gesture_id}.json"
        if _safe_relative_path(reference) != expected:
            raise ValueError("Dataset references must use their matching local dataset filename")
    model_ref = raw["model_ref"]
    if model_ref is not None and _safe_relative_path(model_ref) != "models/custom_gestures.json":
        raise ValueError("Model reference must point to the local profile model")
    mappings = raw["action_mappings"]
    if not isinstance(mappings, list) or len(mappings) > 1000:
        raise ValueError("Action mappings are invalid")
    seen_mappings: set[str] = set()
    for item in mappings:
        if not isinstance(item, dict) or set(item) != {
            "gesture_id", "gesture_name", "action_type", "shortcut", "enabled",
        }:
            raise ValueError("Action mapping fields are invalid")
        mapping = ActionMapping(**item)
        if mapping.gesture_id not in ids or mapping.gesture_id in seen_mappings:
            raise ValueError("Action mapping refers to an unknown or duplicate gesture")
        seen_mappings.add(mapping.gesture_id)
    thresholds = raw["thresholds"]
    if not isinstance(thresholds, dict) or not set(thresholds).issubset(ids):
        raise ValueError("Gesture threshold IDs are invalid")
    for values in thresholds.values():
        if not isinstance(values, dict) or set(values) != {"minimum_margin", "cooldown_seconds"}:
            raise ValueError("Threshold configuration fields are invalid")
        margin, cooldown = values["minimum_margin"], values["cooldown_seconds"]
        if (isinstance(margin, bool) or not isinstance(margin, (int, float))
                or not 0 <= margin <= 1_000_000):
            raise ValueError("Gesture decision threshold is outside the supported range")
        if (isinstance(cooldown, bool) or not isinstance(cooldown, (int, float))
                or not 0 <= cooldown <= 3600):
            raise ValueError("Gesture cooldown is outside the supported range")
    normalized = dict(raw)
    normalized.update(id=profile_id, name=name, version=PROFILE_VERSION,
                      dry_run=dry_run, stop_gesture_id=stop_gesture_id,
                      pause_gesture_id=pause_gesture_id,
                      resume_gesture_id=resume_gesture_id,
                      theme_mode=theme_mode)
    return normalized


class ProfileStore:
    """Manage profile manifests and their feature-only datasets/data-only model."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or default_profile_directory()).expanduser().resolve()

    def _directory(self, profile_id: str) -> Path:
        try:
            canonical = str(UUID(profile_id))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("Invalid profile ID") from exc
        return self.root / canonical

    def _manifest_path(self, profile_id: str) -> Path:
        return self._directory(profile_id) / "profile.json"

    def list(self) -> tuple[dict[str, Any], ...]:
        if not self.root.exists():
            return ()
        profiles = []
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            try:
                if not self._manifest_path(path.name).is_file():
                    continue
                profiles.append(self.load(path.name, validate_assets=False))
            except (OSError, ValueError):
                # A damaged profile must not prevent the tray application from starting.
                continue
        return tuple(sorted(profiles, key=lambda item: item["name"].casefold()))

    def active_id(self) -> str | None:
        path = self.root / "active.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            profile_id = str(UUID(value["profile_id"]))
            self.load(profile_id, validate_assets=False)
            return profile_id
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def set_active(self, profile_id: str) -> None:
        profile = self.load(profile_id, validate_assets=False)
        self._atomic_write(self.root / "active.json",
                           json.dumps({"profile_id": profile["id"]}).encode("utf-8"))

    def load(self, profile_id: str, *, validate_assets: bool = True) -> dict[str, Any]:
        path = self._manifest_path(profile_id)
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Profile manifest is too large")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(f"Profile manifest could not be read: {exc}") from exc
        profile = validate_profile(raw)
        if validate_assets:
            self.validate_assets(profile)
        return profile

    def create(self, name: str) -> dict[str, Any]:
        normalized_name = validate_name(name)
        if any(item["name"].casefold() == normalized_name.casefold() for item in self.list()):
            raise ValueError("A profile with that name already exists")
        profile_id = str(uuid4())
        now = _utc_now()
        profile = {
            "format": PROFILE_FORMAT, "version": PROFILE_VERSION, "id": profile_id,
            "name": normalized_name, "metadata": {"created_at": now, "updated_at": now},
            "gesture_definitions": [], "dataset_refs": {}, "model_ref": None,
            "action_mappings": [], "thresholds": {},
            "dry_run": True, "stop_gesture_id": None, "theme_mode": "light",
            "pause_gesture_id": None, "resume_gesture_id": None,
        }
        self.save(profile)
        return profile

    def save(self, raw: object) -> dict[str, Any]:
        profile = validate_profile(raw)
        directory = self._directory(profile["id"])
        directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(profile, ensure_ascii=False, allow_nan=False, indent=2)
        self._atomic_write(directory / "profile.json", payload.encode("utf-8"))
        return profile

    def rename(self, profile_id: str, name: str) -> dict[str, Any]:
        profile = self.load(profile_id, validate_assets=False)
        normalized_name = validate_name(name)
        if any(item["id"] != profile["id"] and item["name"].casefold() == normalized_name.casefold()
               for item in self.list()):
            raise ValueError("A profile with that name already exists")
        profile["name"] = normalized_name
        profile["metadata"]["updated_at"] = _utc_now()
        return self.save(profile)

    def update_configuration(self, profile_id: str, *, datasets: tuple[GestureDataset, ...],
                             model: PortableGestureModel | None,
                             mappings: dict[str, ActionMapping],
                             thresholds: dict[str, dict[str, float]],
                             copy_assets: bool = True) -> dict[str, Any]:
        try:
            profile = validate_profile(json.loads(self._manifest_path(profile_id).read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(f"Profile manifest could not be read: {exc}") from exc
        directory = self._directory(profile_id)
        dataset_dir = directory / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        definitions: list[dict[str, Any]] = []
        refs: dict[str, str] = {}
        for dataset in datasets:
            validate_dataset(dataset)
            dataset_path = dataset_dir / f"{dataset.id}.json"
            if copy_assets:
                self._atomic_write(dataset_path, json.dumps(dataset_to_dict(dataset), ensure_ascii=False,
                                                            allow_nan=False, indent=2).encode("utf-8"))
            definitions.append({"id": dataset.id, "name": dataset.name,
                                "gesture_type": dataset.gesture_type, "hand_count": dataset.hand_count})
            refs[dataset.id] = f"datasets/{dataset.id}.json"
        existing_ids = set(refs)
        for path in dataset_dir.glob("*.json"):
            if path.stem not in existing_ids:
                path.unlink()
        if model is not None:
            validate_model(model)
            model_path = directory / "models" / "custom_gestures.json"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if copy_assets:
                self._atomic_write(model_path, json.dumps(model.to_dict(), ensure_ascii=False,
                                                           allow_nan=False, separators=(",", ":")).encode("utf-8"))
            profile["model_ref"] = "models/custom_gestures.json"
        else:
            profile["model_ref"] = None
        profile["gesture_definitions"] = definitions
        profile["dataset_refs"] = refs
        if profile["stop_gesture_id"] not in existing_ids:
            profile["stop_gesture_id"] = None
        if profile["pause_gesture_id"] not in existing_ids:
            profile["pause_gesture_id"] = None
        if profile["resume_gesture_id"] not in existing_ids:
            profile["resume_gesture_id"] = None
        profile["action_mappings"] = [mapping.to_dict() for mapping in mappings.values()
                                       if mapping.gesture_id in existing_ids]
        profile["thresholds"] = {key: value for key, value in thresholds.items()
                                  if key in existing_ids}
        profile["metadata"]["updated_at"] = _utc_now()
        return self.save(profile)

    def update_settings(self, profile_id: str, *, mappings: dict[str, ActionMapping],
                        thresholds: dict[str, dict[str, float]]) -> dict[str, Any]:
        """Update recognition/action settings without loading feature/model assets."""
        profile = self.load(profile_id, validate_assets=False)
        gesture_ids = set(profile["dataset_refs"])
        profile["action_mappings"] = [mapping.to_dict() for key, mapping in mappings.items()
                                      if key in gesture_ids]
        profile["thresholds"] = {key: value for key, value in thresholds.items()
                                  if key in gesture_ids}
        profile["metadata"]["updated_at"] = _utc_now()
        return self.save(profile)

    def update_execution_settings(self, profile_id: str, *, dry_run: bool,
                                  stop_gesture_id: str | None,
                                  pause_gesture_id: str | None | object = _UNSET,
                                  resume_gesture_id: str | None | object = _UNSET) -> dict[str, Any]:
        """Persist execution preference and independent static control gesture IDs."""
        if type(dry_run) is not bool:
            raise ValueError("Profile Dry-run setting must be a boolean")
        profile = self.load(profile_id, validate_assets=False)
        proposed = {
            "stop_gesture_id": stop_gesture_id,
            "pause_gesture_id": profile["pause_gesture_id"] if pause_gesture_id is _UNSET else pause_gesture_id,
            "resume_gesture_id": profile["resume_gesture_id"] if resume_gesture_id is _UNSET else resume_gesture_id,
        }
        for field, gesture_id in proposed.items():
            if gesture_id is None:
                continue
            try:
                normalized = str(UUID(gesture_id))
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(f"{field.replace('_', ' ').title()} must be a UUID") from exc
            if normalized != gesture_id or normalized not in profile["dataset_refs"]:
                raise ValueError(f"{field.replace('_', ' ').title()} must reference a gesture in this profile")
        if (proposed["stop_gesture_id"] is not None and proposed["stop_gesture_id"] in {
                proposed["pause_gesture_id"], proposed["resume_gesture_id"]}):
            raise ValueError("Stop Gesture cannot use the same gesture as Pause or Resume")
        profile["dry_run"] = dry_run
        profile.update(proposed)
        profile["metadata"]["updated_at"] = _utc_now()
        return self.save(profile)

    def update_theme_mode(self, profile_id: str, *, theme_mode: str) -> dict[str, Any]:
        """Persist a validated light/dark appearance choice in its profile."""
        if theme_mode not in ("light", "dark"):
            raise ValueError("Profile theme mode must be light or dark")
        profile = self.load(profile_id, validate_assets=False)
        profile["theme_mode"] = theme_mode
        profile["metadata"]["updated_at"] = _utc_now()
        return self.save(profile)

    def validate_assets(self, profile: dict[str, Any]) -> tuple[GestureDataset, ...]:
        directory = self._directory(profile["id"])
        datasets: list[GestureDataset] = []
        for gesture_id, reference in profile["dataset_refs"].items():
            path = directory.joinpath(*PurePosixPath(_safe_relative_path(reference)).parts).resolve()
            if not path.is_relative_to(directory.resolve()) or path.stat().st_size > MAX_PROFILE_FILE_BYTES:
                raise ValueError("Dataset asset path or size is invalid")
            try:
                dataset = dataset_from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                raise ValueError(f"Profile dataset {gesture_id} is invalid: {exc}") from exc
            if dataset.id != gesture_id:
                raise ValueError("Dataset ID does not match its profile reference")
            definition = next(item for item in profile["gesture_definitions"] if item["id"] == gesture_id)
            if (dataset.name != definition["name"] or dataset.gesture_type != definition["gesture_type"]
                    or dataset.hand_count != definition["hand_count"]):
                raise ValueError("Dataset definition does not match profile metadata")
            datasets.append(dataset)
        if profile["model_ref"] is not None:
            path = directory.joinpath(*PurePosixPath(_safe_relative_path(profile["model_ref"])).parts).resolve()
            if not path.is_relative_to(directory.resolve()) or path.stat().st_size > MAX_PROFILE_FILE_BYTES:
                raise ValueError("Model asset path or size is invalid")
            try:
                model = PortableGestureModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
                validate_model(model)
            except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                raise ValueError(f"Profile model is invalid: {exc}") from exc
            if not {label.dataset_id for label in model.labels}.issubset({dataset.id for dataset in datasets}):
                raise ValueError("Profile model labels are missing a referenced dataset")
        return tuple(datasets)

    def delete(self, profile_id: str) -> None:
        shutil.rmtree(self._directory(profile_id))

    def export(self, profile_id: str, destination: Path) -> Path:
        profile = self.load(profile_id)
        directory = self._directory(profile_id)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("profile.json", json.dumps(profile, ensure_ascii=False, indent=2))
            for relative in (*profile["dataset_refs"].values(), *((profile["model_ref"],) if profile["model_ref"] else ())):
                safe = _safe_relative_path(relative)
                path = directory.joinpath(*PurePosixPath(safe).parts).resolve()
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError("Profile asset escapes its profile directory")
                archive.write(path, safe)
        return destination

    def import_archive(self, source: Path) -> dict[str, Any]:
        source = Path(source)
        if source.stat().st_size > MAX_PROFILE_ARCHIVE_BYTES:
            raise ValueError("Profile archive is larger than the supported limit")
        stage_parent = self.root
        stage_parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".profile-import-", dir=stage_parent))
        try:
            with zipfile.ZipFile(source, "r") as archive:
                infos = archive.infolist()
                if len(infos) > MAX_PROFILE_MEMBERS:
                    raise ValueError("Profile archive has too many files")
                total = 0
                names: set[str] = set()
                for info in infos:
                    name = _safe_relative_path(info.filename)
                    if name in names or info.is_dir():
                        raise ValueError("Profile archive contains duplicate or directory entries")
                    names.add(name)
                    mode = info.external_attr >> 16
                    if mode & 0o170000 == 0o120000:
                        raise ValueError("Profile archive cannot contain symbolic links")
                    total += info.file_size
                    if info.file_size > MAX_PROFILE_FILE_BYTES or total > MAX_PROFILE_ARCHIVE_BYTES:
                        raise ValueError("Profile archive expands beyond the supported size")
                    output = stage.joinpath(*PurePosixPath(name).parts).resolve()
                    if not output.is_relative_to(stage.resolve()):
                        raise ValueError("Profile archive path escapes its import directory")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(archive.read(info))
            manifest_path = stage / "profile.json"
            if not manifest_path.is_file():
                raise ValueError("Profile archive is missing profile.json")
            profile = validate_profile(json.loads(manifest_path.read_text(encoding="utf-8")))
            existing_names = {item["name"].casefold() for item in self.list()}
            base_name = profile["name"]
            suffix = 2
            while profile["name"].casefold() in existing_names:
                ending = f" ({suffix})"
                profile["name"] = f"{base_name[:40-len(ending)]}{ending}"
                suffix += 1
            if profile["name"] != base_name:
                profile["metadata"]["updated_at"] = _utc_now()
            self._atomic_write(manifest_path, json.dumps(profile, ensure_ascii=False,
                                                          allow_nan=False, indent=2).encode("utf-8"))
            expected_members = {"profile.json", *profile["dataset_refs"].values()}
            if profile["model_ref"]:
                expected_members.add(profile["model_ref"])
            if names != expected_members:
                raise ValueError("Profile archive contains missing or unsupported files")
            # Validate each bundled file before it reaches the profile store.
            for gesture_id, relative in profile["dataset_refs"].items():
                path = stage.joinpath(*PurePosixPath(_safe_relative_path(relative)).parts)
                dataset = dataset_from_dict(json.loads(path.read_text(encoding="utf-8")))
                if dataset.id != gesture_id:
                    raise ValueError("Imported dataset ID does not match its reference")
            if profile["model_ref"]:
                model_path = stage.joinpath(*PurePosixPath(profile["model_ref"]).parts)
                model = PortableGestureModel.from_dict(json.loads(model_path.read_text(encoding="utf-8")))
                validate_model(model)
            destination = self._directory(profile["id"])
            if destination.exists():
                profile["id"] = str(uuid4())
                profile["metadata"]["updated_at"] = _utc_now()
                destination = self._directory(profile["id"])
                self._atomic_write(manifest_path, json.dumps(profile, ensure_ascii=False,
                                                               allow_nan=False, indent=2).encode("utf-8"))
            shutil.copytree(stage, destination)
            return self.load(profile["id"], validate_assets=False)
        except (zipfile.BadZipFile, UnicodeError, json.JSONDecodeError, RuntimeError, RecursionError) as exc:
            raise ValueError(f"Profile archive is malformed: {exc}") from exc
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
