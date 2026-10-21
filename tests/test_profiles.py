from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
import zipfile

from gesture_manager.actions import ActionMapping
from gesture_manager.gesture_data import GestureDataset, TrainingExample
from gesture_manager.gesture_training import GestureModelStore, GestureTrainer
from gesture_manager.profiles import ProfileStore, validate_profile


def one_dataset() -> GestureDataset:
    return GestureDataset(
        id=str(uuid4()), name="Personal pose", gesture_type="static_pose", hand_count=1,
        examples=(TrainingExample(str(uuid4()), datetime.now(timezone.utc).isoformat(), (0.0,) * 63),),
    )


class ProfileTests(unittest.TestCase):
    def test_create_rename_select_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory))
            profile = store.create("Default")
            store.set_active(profile["id"])
            renamed = store.rename(profile["id"], "Work")
            self.assertEqual(store.active_id(), profile["id"])
            self.assertEqual(renamed["name"], "Work")
            other = store.create("Gaming")
            store.set_active(other["id"])
            store.delete(profile["id"])
            self.assertEqual(store.list()[0]["name"], "Gaming")

    def test_export_import_round_trip_includes_local_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ProfileStore(root / "profiles")
            profile = store.create("Custom")
            dataset = one_dataset()
            mapping = ActionMapping(dataset.id, dataset.name, "keyboard_shortcut", "Ctrl+V", True)
            store.update_configuration(profile["id"], datasets=(dataset,), model=None,
                                       mappings={dataset.id: mapping}, thresholds={dataset.id: {
                                           "minimum_margin": 0.25, "cooldown_seconds": 2.0}})
            exported = store.export(profile["id"], root / "custom.gmp")
            imported = store.import_archive(exported)
            self.assertEqual(imported["gesture_definitions"][0]["id"], dataset.id)
            self.assertEqual(imported["thresholds"][dataset.id]["cooldown_seconds"], 2.0)
            self.assertEqual(imported["action_mappings"][0]["shortcut"], "Ctrl+V")
            self.assertEqual(store.validate_assets(imported), (dataset,))

    def test_profile_saves_dry_run_and_all_control_gestures_and_exports_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ProfileStore(root / "profiles")
            profile = store.create("Configured")
            dataset = one_dataset()
            pause = one_dataset()
            resume = one_dataset()
            store.update_configuration(profile["id"], datasets=(dataset, pause, resume), model=None,
                                       mappings={}, thresholds={})
            configured = store.update_execution_settings(
                profile["id"], dry_run=False, stop_gesture_id=dataset.id,
                pause_gesture_id=pause.id, resume_gesture_id=resume.id)
            configured = store.update_theme_mode(profile["id"], theme_mode="dark")
            self.assertFalse(configured["dry_run"])
            self.assertEqual(configured["stop_gesture_id"], dataset.id)
            self.assertEqual(configured["pause_gesture_id"], pause.id)
            self.assertEqual(configured["resume_gesture_id"], resume.id)
            self.assertEqual(configured["theme_mode"], "dark")
            # A different profile update must preserve the saved appearance.
            configured = store.update_execution_settings(
                profile["id"], dry_run=False, stop_gesture_id=dataset.id)
            self.assertEqual(configured["theme_mode"], "dark")
            archive = store.export(profile["id"], root / "configured.gmp")
            imported = store.import_archive(archive)
            self.assertFalse(imported["dry_run"])
            self.assertEqual(imported["stop_gesture_id"], dataset.id)
            self.assertEqual(imported["pause_gesture_id"], pause.id)
            self.assertEqual(imported["resume_gesture_id"], resume.id)
            self.assertEqual(imported["theme_mode"], "dark")

    def test_profile_rejects_unsupported_theme_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory) / "profiles")
            profile = store.create("Configured")
            with self.assertRaisesRegex(ValueError, "theme mode"):
                store.update_theme_mode(profile["id"], theme_mode="system")

    def test_stop_gesture_must_reference_profile_gesture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory) / "profiles")
            profile = store.create("Config")
            with self.assertRaisesRegex(ValueError, "gesture in this profile"):
                store.update_execution_settings(profile["id"], dry_run=True,
                                                stop_gesture_id=str(uuid4()))

    def test_pause_and_resume_may_share_gesture_but_stop_must_differ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory) / "profiles")
            profile = store.create("Control config")
            first, second = one_dataset(), one_dataset()
            store.update_configuration(profile["id"], datasets=(first, second), model=None,
                                       mappings={}, thresholds={})
            configured = store.update_execution_settings(
                profile["id"], dry_run=True, stop_gesture_id=None,
                pause_gesture_id=first.id, resume_gesture_id=first.id)
            self.assertEqual(configured["pause_gesture_id"], first.id)
            self.assertEqual(configured["resume_gesture_id"], first.id)
            with self.assertRaisesRegex(ValueError, "Stop Gesture cannot use"):
                store.update_execution_settings(
                    profile["id"], dry_run=True, stop_gesture_id=first.id,
                    pause_gesture_id=first.id, resume_gesture_id=first.id)

    def test_profile_archive_bundles_and_validates_portable_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = []
            for class_index, center in enumerate((-1.0, 1.0)):
                gesture_id = str(uuid4())
                examples = tuple(TrainingExample(
                    str(uuid4()), datetime.now(timezone.utc).isoformat(),
                    (center + index * 0.1,) * 63,
                ) for index in range(8))
                datasets.append(GestureDataset(gesture_id, f"Pose {class_index}", "static_pose", 1, examples))
            trainer = GestureTrainer(GestureModelStore(root / "build-model.json"))
            model = trainer.train(tuple(datasets)).model
            store = ProfileStore(root / "profiles")
            profile = store.create("Model bundle")
            store.update_configuration(profile["id"], datasets=tuple(datasets), model=model,
                                       mappings={}, thresholds={})
            archive = store.export(profile["id"], root / "bundle.gmp")
            imported = store.import_archive(archive)
            self.assertEqual(imported["model_ref"], "models/custom_gestures.json")
            self.assertTrue((store._directory(imported["id"]) / "models" / "custom_gestures.json").is_file())
            store.validate_assets(imported)

    def test_rejects_path_traversal_and_unknown_executable_members(self) -> None:
        profile = {
            "format": "gesture-manager-profile", "version": 1, "id": str(uuid4()),
            "name": "Bad", "metadata": {"created_at": "now", "updated_at": "now"},
            "gesture_definitions": [], "dataset_refs": {}, "model_ref": None,
            "action_mappings": [], "thresholds": {}, "script": "print('no')",
        }
        with self.assertRaises(ValueError):
            validate_profile(profile)
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory) / "profiles")
            archive = Path(directory) / "bad.gmp"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("../escape.py", "print('no')")
            with self.assertRaises(ValueError):
                store.import_archive(archive)
            safe_archive = Path(directory) / "script.gmp"
            manifest = {
                "format": "gesture-manager-profile", "version": 1, "id": str(uuid4()),
                "name": "No scripts", "metadata": {"created_at": "now", "updated_at": "now"},
                "gesture_definitions": [], "dataset_refs": {}, "model_ref": None,
                "action_mappings": [], "thresholds": {},
            }
            with zipfile.ZipFile(safe_archive, "w") as zip_file:
                import json
                zip_file.writestr("profile.json", json.dumps(manifest))
                zip_file.writestr("scripts/start.py", "print('never run')")
            with self.assertRaisesRegex(ValueError, "unsupported files"):
                store.import_archive(safe_archive)

    def test_rejects_unsafe_thresholds_and_dataset_reference(self) -> None:
        profile = {
            "format": "gesture-manager-profile", "version": 1, "id": str(uuid4()),
            "name": "Unsafe", "metadata": {"created_at": "now", "updated_at": "now"},
            "gesture_definitions": [], "dataset_refs": {}, "model_ref": "../../model.pkl",
            "action_mappings": [], "thresholds": {},
        }
        with self.assertRaisesRegex(ValueError, "escapes|Model reference"):
            validate_profile(profile)

