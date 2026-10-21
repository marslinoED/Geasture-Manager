from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from gesture_manager.actions import (
    ActionDispatcher,
    ActionMapping,
    ActionMappingStore,
    PynputKeyboardBackend,
    normalize_shortcut,
    pynput_key_token,
    windows_vk_token,
)


class MockKeyboardBackend:
    def __init__(self, error: Exception | None = None) -> None:
        self.sent: list[str] = []
        self.error = error

    def send_shortcut(self, shortcut: str) -> None:
        self.sent.append(shortcut)
        if self.error is not None:
            raise self.error


class MockController:
    def __init__(self, fail_release: object | None = None) -> None:
        self.pressed: list[object] = []
        self.released: list[object] = []
        self.fail_release = fail_release

    def press(self, key: object) -> None:
        self.pressed.append(key)

    def release(self, key: object) -> None:
        self.released.append(key)
        if key is self.fail_release:
            raise OSError("release failed")


class ActionMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = ActionMapping("gesture-1", "My custom pose", "keyboard_shortcut", "ctrl+c", True)
        self.event = SimpleNamespace(gesture_id="gesture-1", trigger_eligible=True)

    def test_supported_chords_are_normalized_and_validated(self) -> None:
        self.assertEqual(normalize_shortcut("ctrl+c"), "Ctrl+C")
        self.assertEqual(normalize_shortcut("Alt+Tab"), "Alt+Tab")
        self.assertEqual(normalize_shortcut("shift+ctrl+z"), "Ctrl+Shift+Z")
        self.assertEqual(normalize_shortcut("a"), "A")
        self.assertEqual(normalize_shortcut("Caps Lock"), "CapsLock")
        self.assertEqual(normalize_shortcut(";"), "Semicolon")
        for invalid in ("Ctrl+C+V", "Ctrl+Ctrl+C", "Ctrl+Shell", "Alt+",
                        "Ctrl", "C+V",
                        "Ctrl+Alt+Delete"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_shortcut(invalid)

    def test_recorder_key_conversion_supports_windows_and_control_key_events(self) -> None:
        key_enum = SimpleNamespace(
            cmd="windows-key", cmd_l="left-windows-key", cmd_r="right-windows-key",
            ctrl="control-key", ctrl_l="left-control", ctrl_r="right-control",
            alt="alt-key", alt_l="left-alt", alt_r="right-alt", alt_gr="alt-gr",
            shift="shift-key", shift_l="left-shift", shift_r="right-shift",
            tab="tab-key", enter="enter-key", esc="escape-key", space="space-key",
            backspace="backspace-key", delete="delete-key", insert="insert-key",
            home="home-key", end="end-key", page_up="page-up-key", page_down="page-down-key",
            left="left-key", right="right-key", up="up-key", down="down-key",
            caps_lock="caps-lock-key", print_screen="print-screen-key",
            f1="f1-key", f12="f12-key",
            media_volume_up="volume-up-key", media_play_pause="play-pause-key",
        )
        self.assertEqual(pynput_key_token("windows-key", key_enum), "Win")
        self.assertEqual(pynput_key_token(SimpleNamespace(char="d", vk=68), key_enum), "D")
        self.assertEqual(pynput_key_token(SimpleNamespace(char="\x04", vk=68), key_enum), "D")
        self.assertEqual(pynput_key_token("tab-key", key_enum), "Tab")
        self.assertEqual(pynput_key_token("caps-lock-key", key_enum), "CapsLock")
        self.assertEqual(pynput_key_token("f1-key", key_enum), "F1")
        self.assertEqual(pynput_key_token("f12-key", key_enum), "F12")
        self.assertEqual(pynput_key_token("volume-up-key", key_enum), "VolumeUp")
        self.assertEqual(windows_vk_token(0xAF), "VolumeUp")
        self.assertEqual(normalize_shortcut("media play pause"), "MediaPlayPause")
        self.assertEqual(pynput_key_token(SimpleNamespace(char=";", vk=0xBA), key_enum), "Semicolon")
        self.assertEqual(normalize_shortcut("Win+D"), "Win+D")

    def test_windows_low_level_recorder_key_conversion(self) -> None:
        self.assertEqual(windows_vk_token(0x5B), "Win")
        self.assertEqual(windows_vk_token(0x44), "D")
        self.assertEqual(windows_vk_token(0x09), "Tab")
        self.assertEqual(windows_vk_token(0x70), "F1")
        self.assertEqual(windows_vk_token(0xBA), "Semicolon")
        self.assertEqual(windows_vk_token(0x5D), "Menu")
        self.assertIsNone(windows_vk_token(0xFF))

    def test_mappings_persist_separately_as_versioned_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ActionMappingStore(Path(directory) / "settings" / "actions.json")
            store.save({self.mapping.gesture_id: self.mapping})
            loaded = store.load()
            self.assertEqual(loaded, {self.mapping.gesture_id: self.mapping})
            self.assertEqual(store.path.parent.name, "settings")

    def test_clearing_mappings_persists_an_empty_mapping_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ActionMappingStore(Path(directory) / "settings" / "actions.json")
            store.save({self.mapping.gesture_id: self.mapping})
            store.save({})
            self.assertEqual(store.load(), {})

    def test_store_rejects_unallowlisted_action_and_unsupported_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actions.json"
            store = ActionMappingStore(path)
            path.write_text('{"format":"gesture-manager-actions","version":1,"mappings":['
                            '{"gesture_id":"g","gesture_name":"x","action_type":"shell_command",'
                            '"shortcut":"Ctrl+C","enabled":true}]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported action type"):
                store.load()
            path.write_text('{"format":"gesture-manager-actions","version":999,"mappings":[]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "version"):
                store.load()

    def test_store_rejects_arbitrary_profile_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actions.json"
            path.write_text(
                '{"format":"gesture-manager-actions","version":1,"mappings":['
                '{"gesture_id":"g","gesture_name":"x","action_type":"keyboard_shortcut",'
                '"shortcut":"Ctrl+C","enabled":true,"command":"calc.exe"}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                ActionMappingStore(path).load()

    def test_real_input_cannot_be_enabled_without_confirmation(self) -> None:
        dispatcher = ActionDispatcher(MockKeyboardBackend())
        with self.assertRaisesRegex(RuntimeError, "Confirm real input"):
            dispatcher.set_dry_run(False)
        with self.assertRaisesRegex(ValueError, "not confirmed"):
            dispatcher.confirm_real_input(confirmed=False)
        self.assertTrue(dispatcher.dry_run)

    def test_returning_to_dry_run_requires_confirmation_for_new_enabled_mappings(self) -> None:
        backend = MockKeyboardBackend()
        dispatcher = ActionDispatcher(backend)
        dispatcher.set_backend(backend)
        dispatcher.confirm_real_input(confirmed=True)
        self.assertFalse(dispatcher.dry_run)

        # Profile switching/import uses this reset before replacing mappings.
        dispatcher.set_dry_run(True)
        enabled = ActionMapping("gesture-1", "Imported gesture", "keyboard_shortcut", "Win+D", True)
        result = dispatcher.dispatch(self.event, {enabled.gesture_id: enabled}, now=10)
        self.assertEqual(result.status, "dry_run")
        self.assertEqual(backend.sent, [])
        with self.assertRaisesRegex(RuntimeError, "Confirm real input"):
            dispatcher.set_dry_run(False)

    def test_dry_run_is_default_and_never_calls_backend(self) -> None:
        backend = MockKeyboardBackend()
        dispatcher = ActionDispatcher(backend)
        result = dispatcher.dispatch(self.event, {self.mapping.gesture_id: self.mapping}, now=1)
        self.assertEqual(result.status, "dry_run")
        self.assertIn("Ctrl+C", result.message)
        self.assertEqual(backend.sent, [])

    def test_disabled_mapping_and_ineligible_event_do_not_send(self) -> None:
        backend = MockKeyboardBackend()
        dispatcher = ActionDispatcher(backend)
        dispatcher.set_backend(backend)
        dispatcher.confirm_real_input(confirmed=True)
        disabled = ActionMapping("gesture-1", "My custom pose", "keyboard_shortcut", "Ctrl+C", False)
        self.assertEqual(dispatcher.dispatch(self.event, {"gesture-1": disabled}).status, "disabled")
        self.assertEqual(dispatcher.dispatch(SimpleNamespace(gesture_id="gesture-1", trigger_eligible=False),
                                             {"gesture-1": self.mapping}).status, "ignored")
        self.assertEqual(backend.sent, [])

    def test_confirmed_real_dispatch_uses_mock_and_rate_limits_repeats(self) -> None:
        backend = MockKeyboardBackend()
        dispatcher = ActionDispatcher(backend)
        dispatcher.set_backend(backend)
        dispatcher.confirm_real_input(confirmed=True)
        first = dispatcher.dispatch(self.event, {"gesture-1": self.mapping}, now=5)
        repeat = dispatcher.dispatch(self.event, {"gesture-1": self.mapping}, now=5.1)
        self.assertEqual(first.status, "sent")
        self.assertEqual(repeat.status, "cooldown")
        self.assertEqual(backend.sent, ["Ctrl+C"])

    def test_global_rate_limit_applies_across_different_gestures(self) -> None:
        backend = MockKeyboardBackend()
        dispatcher = ActionDispatcher(backend)
        dispatcher.set_backend(backend)
        dispatcher.confirm_real_input(confirmed=True)
        other = ActionMapping("gesture-2", "Second pose", "keyboard_shortcut", "Ctrl+V", True)
        self.assertEqual(dispatcher.dispatch(self.event, {"gesture-1": self.mapping}, now=3).status,
                         "sent")
        second_event = SimpleNamespace(gesture_id="gesture-2", trigger_eligible=True)
        result = dispatcher.dispatch(second_event, {"gesture-2": other}, now=3.1)
        self.assertEqual(result.status, "cooldown")
        self.assertEqual(backend.sent, ["Ctrl+C"])

    def test_global_emergency_stop_blocks_dispatch_and_rearm_returns_to_dry_run(self) -> None:
        backend = MockKeyboardBackend()
        dispatcher = ActionDispatcher(backend)
        dispatcher.set_backend(backend)
        dispatcher.confirm_real_input(confirmed=True)
        dispatcher.emergency_stop()
        self.assertEqual(dispatcher.dispatch(self.event, {"gesture-1": self.mapping}).status,
                         "emergency_stopped")
        dispatcher.rearm()
        self.assertTrue(dispatcher.dry_run)
        self.assertEqual(dispatcher.dispatch(self.event, {"gesture-1": self.mapping}, now=10).status,
                         "dry_run")
        self.assertEqual(backend.sent, [])

    def test_backend_releases_every_pressed_key_when_a_release_fails(self) -> None:
        backend = PynputKeyboardBackend.__new__(PynputKeyboardBackend)
        backend._controller = MockController()
        key_values = {"Ctrl": object(), "Alt": object(), "C": object()}
        backend._controller.fail_release = key_values["Alt"]
        backend._resolve = lambda token: key_values[token]
        with self.assertRaisesRegex(OSError, "release failed"):
            backend.send_shortcut("Ctrl+Alt+C")
        self.assertEqual(backend._controller.released,
                         [key_values["C"], key_values["Alt"], key_values["Ctrl"]])


if __name__ == "__main__":
    unittest.main()
