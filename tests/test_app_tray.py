import unittest

from gesture_manager.app import AppCommand, GestureManagerApp
from gesture_manager.runtime import CommandQueue, RuntimeController, RuntimeState
from gesture_manager.assets import load_app_icon
from gesture_manager.tray_icons import create_status_icons


class FakeRoot:
    def __init__(self):
        self.scheduled = []
        self.visible = False

    def after(self, delay, callback):
        self.scheduled.append((delay, callback))

    def deiconify(self):
        self.visible = True

    def lift(self):
        pass


class FakeUI:
    def __init__(self):
        self.refreshes = 0
        self.gesture_actions_paused = False

    def refresh_state(self):
        self.refreshes += 1


class FakeIcon:
    def __init__(self):
        self.refreshes = 0
        self.notifications = []
        self.icon = None
        self.title = ""

    def update_menu(self):
        self.refreshes += 1

    def notify(self, message, title=None):
        self.notifications.append((title, message))


class TrayCommandTests(unittest.TestCase):
    @staticmethod
    def make_app():
        app = GestureManagerApp.__new__(GestureManagerApp)
        app.root = FakeRoot()
        app.ui = FakeUI()
        app.icon = FakeIcon()
        app.controller = RuntimeController()
        app._commands = CommandQueue()
        app._closing = False
        app._tray_icon_images = create_status_icons(load_app_icon())
        app._tray_visual_state = "stopped"
        app._tray_visual_generation = 0
        app._last_runtime_state = RuntimeState.STOPPED
        app._last_shortcut_paused = False
        app.controller._on_change = app._runtime_state_changed
        return app

    def test_runtime_state_callback_refreshes_ui_and_tray(self):
        app = self.make_app()
        app._runtime_state_changed(RuntimeState.STOPPED, None)
        self.assertEqual(app.ui.refreshes, 1)
        self.assertEqual(app.icon.refreshes, 1)

    def test_tray_commands_continue_after_opening_settings(self):
        app = self.make_app()
        app._queue_show(None, None)
        app._process_commands()
        self.assertTrue(app.root.visible)
        self.assertTrue(app.root.scheduled)

        app._queue_start(None, None)
        app._process_commands()
        self.assertEqual(app.controller.state, RuntimeState.ACTIVE)
        self.assertGreaterEqual(app.ui.refreshes, 2)
        self.assertGreaterEqual(app.icon.refreshes, 1)
        self.assertEqual(app.root.scheduled[-1][0], 250)

    def test_runtime_changes_update_tinted_icon_and_notify(self):
        app = self.make_app()
        app.controller.start()
        self.assertEqual(app._tray_visual_state, "active")
        self.assertEqual(app.icon.icon.size, (64, 64))
        self.assertIn(("Active", "Gesture recognition is active."), app.icon.notifications)
        app.controller.pause()
        self.assertEqual(app._tray_visual_state, "paused")
        app.controller.resume()
        self.assertEqual(app._tray_visual_state, "unpaused")
        self.assertEqual(app.root.scheduled[-1][0], 1000)
        app.controller.stop()
        self.assertEqual(app._tray_visual_state, "stopped")
        self.assertIn(("Stopped", "Gesture recognition is stopped."), app.icon.notifications)

    def test_shortcut_pause_and_resume_notify_and_change_tray_filter(self):
        app = self.make_app()
        app.ui.gesture_actions_paused = True
        app._gesture_action_state_changed()
        self.assertEqual(app._tray_visual_state, "paused")
        app.ui.gesture_actions_paused = False
        app._gesture_action_state_changed()
        self.assertEqual(app._tray_visual_state, "unpaused")
        self.assertIn(("Shortcuts paused", "Ordinary shortcut actions are blocked."), app.icon.notifications)
        self.assertIn(("Shortcuts resumed", "Eligible shortcut actions are available."), app.icon.notifications)


if __name__ == "__main__":
    unittest.main()
