import json
from pathlib import Path
import tempfile
import unittest

from gesture_manager.execution_mode import ExecutionModeStore


class ExecutionModeStoreTests(unittest.TestCase):
    def test_profile_mode_defaults_to_dry_run_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ExecutionModeStore(Path(temp) / "settings" / "execution_mode.json")
            self.assertTrue(store.load_dry_run())
            store.save_dry_run(False)
            self.assertFalse(store.load_dry_run())
            store.save_dry_run(True)
            self.assertTrue(store.load_dry_run())

    def test_rejects_invalid_or_unsupported_data(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "execution_mode.json"
            store = ExecutionModeStore(path)
            with self.assertRaises(ValueError):
                store.save_dry_run(0)  # type: ignore[arg-type]
            path.write_text(json.dumps({
                "format": ExecutionModeStore.FORMAT,
                "version": 1,
                "dry_run": "false",
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                store.load_dry_run()


if __name__ == "__main__":
    unittest.main()
