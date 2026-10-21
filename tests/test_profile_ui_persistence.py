from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
from types import SimpleNamespace

from gesture_manager.gesture_data import GestureDataset, TrainingExample
from gesture_manager.profiles import ProfileStore
from gesture_manager.gesture_controls import GestureControlRouter
from gesture_manager.ui import GestureManagerUI


class ProfileUISettingsPersistenceTests(unittest.TestCase):
    def test_loading_profile_then_saving_mode_preserves_stop_gesture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory) / "profiles")
            profile = store.create("Configured")
            dataset = GestureDataset(
                id=str(uuid4()), name="Close service", gesture_type="static_pose",
                hand_count=1,
                examples=(TrainingExample(
                    str(uuid4()), datetime.now(timezone.utc).isoformat(), (0.0,) * 63,
                ),),
            )
            store.update_configuration(profile["id"], datasets=(dataset,), model=None,
                                       mappings={}, thresholds={})
            profile = store.update_execution_settings(
                profile["id"], dry_run=True, stop_gesture_id=dataset.id)
            profile = store.update_theme_mode(profile["id"], theme_mode="dark")

            # Exercise the same profile-load and execution-mode-save methods
            # used by the UI without constructing a Tk window.
            ui = object.__new__(GestureManagerUI)
            ui.root = SimpleNamespace(theme_mode="light")
            ui.profile_store = store
            ui._active_profile = profile
            ui._profile_mode_preference = True
            ui._profile_mode_error = None
            ui._stop_gesture_id = None
            ui._pause_gesture_id = None
            ui._resume_gesture_id = None
            ui._gesture_controls = GestureControlRouter()
            ui._stop_gesture_error = None
            ui._configure_profile_storage()

            self.assertEqual(ui._stop_gesture_id, dataset.id)
            self.assertEqual(ui.root.theme_mode, "dark")
            ui._save_profile_mode_preference(False)
            reloaded = store.load(profile["id"], validate_assets=False)
            self.assertEqual(reloaded["stop_gesture_id"], dataset.id)
            self.assertFalse(reloaded["dry_run"])
            self.assertEqual(reloaded["theme_mode"], "dark")


if __name__ == "__main__":
    unittest.main()
