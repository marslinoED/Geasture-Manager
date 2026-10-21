"""Tray-first application lifecycle and runtime command routing."""

from __future__ import annotations

from enum import Enum
import logging
import os
from pathlib import Path
import tkinter as tk

from PIL import Image, ImageTk
import pystray
import ttkbootstrap as ttk

from gesture_manager.assets import app_icon_ico_path, load_app_icon
from gesture_manager.tray_icons import create_status_icons
from gesture_manager.theme import (
    DARK_THEME, LIGHT_THEME, apply_readable_text_styles, install_brand_themes,
)
from gesture_manager.runtime import (
    CommandQueue,
    QueueResult,
    RuntimeCommand,
    RuntimeController,
    RuntimeState,
)
from gesture_manager.camera_manager import CameraManager
from gesture_manager.hand_landmark_engine import HandLandmarkEngine
from gesture_manager.ui import GestureManagerUI

logger = logging.getLogger(__name__)


class AppCommand(str, Enum):
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    EMERGENCY_STOP = "emergency_stop"
    SHOW_WINDOW = "show_window"
    EXIT = "exit"


class GestureManagerApp:
    """Keep tray callbacks off Tk and coordinate clean application shutdown."""

    def __init__(self, root: ttk.Window) -> None:
        self.root = root
        # Runtime commands are handled by Tk's main-thread command queue. The
        # callbacks only coordinate the already-owned UI and camera services.
        self.controller = RuntimeController(
            start_action=self._start_runtime,
            pause_action=self._pause_runtime,
            resume_action=self._resume_runtime,
            stop_action=self._stop_runtime,
            on_change=self._runtime_state_changed,
        )
        self.camera_manager = CameraManager()
        self.hand_engine = HandLandmarkEngine(self.camera_manager)
        self._commands: CommandQueue[AppCommand] = CommandQueue()
        self._closing = False
        self._tray_ready = False
        self._tray_failed = False
        self.icon: pystray.Icon | None = None
        self._tray_icon_images: dict[str, Image.Image] = {}
        self._tray_visual_state = "stopped"
        self._tray_visual_generation = 0
        self._last_runtime_state = self.controller.state
        self._last_shortcut_paused = False

        self.root.title("Gesture Manager")
        self._app_icon_image = load_app_icon()
        self._tray_icon_images = create_status_icons(self._app_icon_image)
        self._window_icon_images = [
            ImageTk.PhotoImage(self._app_icon_image.resize((size, size), Image.Resampling.LANCZOS),
                               master=self.root)
            for size in (16, 32, 48, 256)
        ]
        self.root.iconphoto(True, *self._window_icon_images)
        icon_path = app_icon_ico_path()
        if icon_path.is_file():
            try:
                # Set this window directly as well as Tk's class default. On
                # Windows the default alone can be ignored by an existing root.
                self.root.iconbitmap(str(icon_path))
                self.root.iconbitmap(default=str(icon_path))
            except tk.TclError:
                logger.exception("Could not set the native Windows application icon")
        self.root.minsize(960, 620)
        self.root.geometry("1280x720")
        install_brand_themes(self.root.style)
        self.root.set_theme_modes(light=LIGHT_THEME, dark=DARK_THEME)
        self.root.theme_mode = "light"
        apply_readable_text_styles(self.root.style, "light")
        self.root.protocol("WM_DELETE_WINDOW", self.hide_window)
        self.ui = GestureManagerUI(
            self.root,
            self.controller,
            request_command=self._queue_runtime_command,
            tray_status=self._tray_status,
            camera_manager=self.camera_manager,
            hand_engine=self.hand_engine,
            on_gesture_action_state=self._gesture_action_state_changed,
        )
        self.root.withdraw()

        try:
            self.icon = pystray.Icon(
                "gesture-manager",
                self._tray_icon_images["stopped"],
                "Gesture Manager",
                menu=pystray.Menu(
                    pystray.MenuItem(
                        "Open Gesture Manager", self._queue_show, default=True
                    ),
                    pystray.MenuItem(
                        self._tray_state_text,
                        self._queue_noop,
                        enabled=lambda _item: False,
                    ),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem("Start", self._queue_start),
                    pystray.MenuItem("Pause", self._queue_pause),
                    pystray.MenuItem("Resume", self._queue_resume),
                    pystray.MenuItem("Stop", self._queue_stop),
                    pystray.Menu.SEPARATOR,
                    pystray.MenuItem("Exit", self._queue_exit),
                ),
            )
            self.icon.run_detached()
            # pystray starts its Win32 message loop on a worker thread. A
            # successful return only means that thread was launched; the
            # native window and notification icon are created later.
            self.root.after(1500, self._check_tray_startup)
        except Exception as exc:
            logger.exception("Could not start the system tray icon")
            self._tray_failed = True
            self.controller.report_error(f"System tray unavailable: {exc}")
            self.show_window()

        self.ui.refresh_state()
        self.root.after(250, self._process_commands)

    def _check_tray_startup(self) -> None:
        """Fall back to the window if pystray's detached Win32 loop died."""
        if self._closing or self.icon is None:
            return
        tray_thread = getattr(self.icon, "_thread", None)
        if tray_thread is not None and tray_thread.is_alive():
            self._tray_ready = True
            self.ui.refresh_state()
            return

        logger.error("System tray message loop did not start; showing the main window")
        self._tray_failed = True
        self.controller.report_error(
            "The system tray could not start. Gesture Manager is open in its window."
        )
        self.show_window()

    def _create_icon_image(self) -> Image.Image:
        return self._app_icon_image.resize((64, 64), Image.Resampling.LANCZOS)

    def _notify_tray(self, title: str, message: str) -> None:
        icon = getattr(self, "icon", None)
        notify = getattr(icon, "notify", None)
        if callable(notify):
            try:
                notify(message, title=title)
            except Exception:
                logger.exception("Could not show tray notification")

    def _set_tray_icon_state(self, status: str) -> None:
        if status not in self._tray_icon_images:
            status = "stopped"
        self._tray_visual_state = status
        self._tray_visual_generation += 1
        icon = getattr(self, "icon", None)
        if icon is not None:
            try:
                icon.icon = self._tray_icon_images[status]
                icon.title = f"Gesture Manager · {status.title()}"
            except Exception:
                logger.exception("Could not update the tray icon")

    def _show_unpaused_tray_state(self) -> None:
        """Briefly highlight Resume in red, then return to the current state."""
        self._set_tray_icon_state("unpaused")
        generation = self._tray_visual_generation

        def restore() -> None:
            if generation == self._tray_visual_generation:
                self._set_tray_icon_state(self._current_tray_state())

        root = getattr(self, "root", None)
        if root is not None and callable(getattr(root, "after", None)):
            root.after(1000, restore)

    def _current_tray_state(self) -> str:
        ui = getattr(self, "ui", None)
        if (self.controller.state is RuntimeState.PAUSED
                or (ui is not None and getattr(ui, "gesture_actions_paused", False))):
            return "paused"
        return "active" if self.controller.state is RuntimeState.ACTIVE else "stopped"

    def _state_notification(self, state: RuntimeState) -> tuple[str, str]:
        messages = {
            RuntimeState.STARTING: ("Starting", "Gesture recognition is starting."),
            RuntimeState.ACTIVE: ("Active", "Gesture recognition is active."),
            RuntimeState.PAUSED: ("Paused", "Gesture recognition is paused."),
            RuntimeState.STOPPING: ("Stopping", "Gesture recognition is stopping."),
            RuntimeState.STOPPED: ("Stopped", "Gesture recognition is stopped."),
            RuntimeState.ERROR: ("Error", self.controller.error or "Gesture Manager encountered an error."),
        }
        return messages[state]

    def _tray_status(self) -> str:
        if self._tray_ready:
            return "Initialized"
        return "Unavailable" if self._tray_failed else "Starting"

    def _start_runtime(self) -> None:
        self.ui.start_runtime()

    def _pause_runtime(self) -> None:
        self.ui.pause_runtime()

    def _resume_runtime(self) -> None:
        self.ui.resume_runtime()

    def _stop_runtime(self) -> None:
        self.ui.stop_runtime()

    def _runtime_state_changed(self, _state: object, _error: str | None) -> None:
        """Keep UI and tray status synchronized for UI, tray, and gesture stops."""
        ui = getattr(self, "ui", None)
        try:
            state = RuntimeState(_state)
        except (TypeError, ValueError):
            state = self.controller.state
        previous = getattr(self, "_last_runtime_state", None)
        self._last_runtime_state = state
        if previous is not state:
            title, message = self._state_notification(state)
            self._notify_tray(title, message)
        if state is RuntimeState.ACTIVE and previous is RuntimeState.PAUSED:
            self._show_unpaused_tray_state()
        else:
            self._set_tray_icon_state(self._current_tray_state())
        if ui is not None:
            try:
                ui.refresh_state()
            except Exception:
                logger.exception("Could not refresh the UI after a runtime state change")
        icon = getattr(self, "icon", None)
        if icon is not None:
            try:
                icon.update_menu()
            except Exception:
                logger.exception("Could not refresh the tray menu after a runtime state change")

    def _gesture_action_state_changed(self) -> None:
        """Refresh the tray's active/shortcut-paused status on control changes."""
        ui = getattr(self, "ui", None)
        paused = bool(ui is not None and getattr(ui, "gesture_actions_paused", False))
        previous_paused = getattr(self, "_last_shortcut_paused", False)
        self._last_shortcut_paused = paused
        if paused != previous_paused:
            if paused:
                self._notify_tray("Shortcuts paused", "Ordinary shortcut actions are blocked.")
                self._set_tray_icon_state("paused")
            else:
                self._notify_tray("Shortcuts resumed", "Eligible shortcut actions are available.")
                self._show_unpaused_tray_state()
        icon = getattr(self, "icon", None)
        if icon is not None:
            try:
                icon.update_menu()
            except Exception:
                logger.exception("Could not refresh tray status after gesture control change")

    def _tray_state_text(self, _item: pystray.MenuItem) -> str:
        state = self.controller.state.value.title()
        ui = getattr(self, "ui", None)
        if state == "Active" and ui is not None and ui.gesture_actions_paused:
            return "Runtime: Active · shortcuts paused"
        return f"Runtime: {state}"

    def _queue_runtime_command(self, command: RuntimeCommand) -> None:
        self._commands.submit(AppCommand(command.value))

    def _queue_show(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        self._commands.submit(AppCommand.SHOW_WINDOW)

    def _queue_start(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        self._commands.submit(AppCommand.START)

    def _queue_pause(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        self._commands.submit(AppCommand.PAUSE)

    def _queue_resume(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        self._commands.submit(AppCommand.RESUME)

    def _queue_stop(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        self._commands.submit(AppCommand.STOP)

    def _queue_emergency_stop(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        self._commands.submit(AppCommand.EMERGENCY_STOP)

    def _queue_exit(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        self._commands.submit(AppCommand.EXIT)

    def _queue_noop(self, _icon: pystray.Icon, _item: pystray.MenuItem) -> None:
        return

    def _handle_command(self, command: AppCommand) -> QueueResult:
        if command is AppCommand.SHOW_WINDOW:
            self.show_window()
        elif command is AppCommand.EXIT:
            self.exit_application()
            return QueueResult.STOP
        elif command is AppCommand.EMERGENCY_STOP:
            self.show_window()
            self.ui.show_page("Actions")
            self.ui.emergency_stop_global()
        else:
            self.controller.handle_command(RuntimeCommand(command.value))
        return QueueResult.CONTINUE

    def _process_commands(self) -> None:
        """Run commands and Tk updates on the Tk main thread."""
        try:
            count = self._commands.process(self._handle_command)
            if self._closing:
                return
            if count:
                self.ui.refresh_state()
                if self.icon is not None:
                    try:
                        self.icon.update_menu()
                    except Exception:
                        logger.exception("Could not refresh the tray menu")
        except Exception:
            logger.exception("Could not process an application command")
            self.controller.report_error(
                "An application command failed. See the log for details."
            )
            self.ui.refresh_state()

        if not self._closing:
            self.root.after(250, self._process_commands)

    def show_window(self) -> None:
        self.root.deiconify()
        self.root.lift()

    def hide_window(self) -> None:
        self.root.withdraw()

    def exit_application(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.controller.stop()
        self.ui.stop_background_tasks()
        self.hand_engine.stop()
        self.camera_manager.stop()
        self.root.after(25, self._finish_shutdown)

    def _finish_shutdown(self) -> None:
        """Wait asynchronously for the camera worker to release its device."""
        if (not self.hand_engine.wait_for_stop(0)
                or not self.camera_manager.wait_for_stop(0)
                or not self.ui.background_tasks_stopped()):
            self.root.after(50, self._finish_shutdown)
            return
        if self.icon is not None:
            try:
                self.icon.stop()
            except Exception:
                logger.exception("Could not stop the system tray icon cleanly")
        self.root.destroy()


def main() -> None:
    log_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "GestureManager" / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=log_dir / "app.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            encoding="utf-8",
        )
    except OSError:
        logging.basicConfig(level=logging.INFO)
        logger.exception("Could not create the application log file")
    root = ttk.Window(
        title="Gesture Manager",
        theme="nord-light",
        size=(1180, 740),
        minsize=(800, 560),
        high_dpi=True,
    )
    GestureManagerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

