"""Polished, camera-independent Gesture Manager desktop interface."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
import logging
import platform
import math
import json
import queue
from pathlib import Path
import shutil
import sys
from threading import Thread
from time import monotonic
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from PIL import Image, ImageTk
import ttkbootstrap as ttk

from gesture_manager import __version__
from gesture_manager.assets import load_app_icon, model_assets_directory
from gesture_manager.theme import apply_readable_text_styles
from gesture_manager.window_chrome import apply_windows_titlebar_theme
from gesture_manager.actions import (
    ActionDispatcher, ActionMapping, ActionMappingStore, PynputKeyboardBackend,
    normalize_shortcut, pynput_key_token, windows_vk_token,
)
from gesture_manager.runtime import RuntimeCommand, RuntimeController, RuntimeState
from gesture_manager.camera_manager import CameraConfig, CameraDevice, CameraManager, CameraState
from gesture_manager.camera_preview import CameraPreview
from gesture_manager.hand_landmark_engine import EngineConfig, EngineState, HandLandmarkEngine
from gesture_manager.gesture_data import (
    GestureDataset,
    GestureRecordingSession,
    GestureStore,
    create_draft,
    default_gesture_directory,
)
from gesture_manager.gesture_training import (
    EvaluationReport,
    GestureModelStore,
    GestureTrainer,
    PortableGestureModel,
    TrainingOutcome,
    model_matches_datasets,
)
from gesture_manager.dynamic_gestures import (
    DynamicGestureDataset, DynamicGestureModel, DynamicGestureModelStore,
    DynamicGestureStore, DynamicGestureTrainer, DynamicRecordingSession,
    create_dynamic_draft,
)
from gesture_manager.dynamic_studio import DynamicStudioRecorder
from gesture_manager.recognition import LiveRecognitionWorker, RecognitionEvent
from gesture_manager.profiles import ProfileStore
from gesture_manager.stop_gesture import (
    StopGestureStore, StopGestureTrigger, describe_stop_candidate,
)
from gesture_manager.gesture_controls import GestureControlRouter
from gesture_manager.execution_mode import ExecutionModeStore
from gesture_manager.scheduler import AdaptiveScheduler, WindowsFocusDetector
from gesture_manager.performance import ProcessMetricsSampler
from gesture_manager.wake_detection import WakeMotionWorker
from gesture_manager.inactivity import FrameInactivityTracker

logger = logging.getLogger(__name__)


PAGES = (
    ("Dashboard", "⌂"),
    ("My Gestures", "☷"),
    ("Gesture Studio", "✦"),
    ("Actions", "⌨"),
    ("Profiles", "▤"),
    ("Camera", "◉"),
    ("General Settings", "⚙"),
    ("About & Diagnostics", "ⓘ"),
)

PAGE_COPY = {
    "Dashboard": "A clear view of the local runtime and what is available in this build.",
    "My Gestures": "Your custom gesture collection.",
    "Gesture Studio": "Create and prepare a custom gesture for local recognition.",
    "Actions": "Map trained custom gestures to supported keyboard shortcuts.",
    "Profiles": "Organize sets of your custom gesture configuration.",
    "Camera": "Camera integration status and preview area.",
    "General Settings": "Appearance and application behavior.",
    "About & Diagnostics": "Application details and local environment information.",
}


def dashboard_column_count(width: int) -> int:
    """Choose a readable dashboard card grid for the available window width."""
    if width >= 1150:
        return 3
    if width >= 860:
        return 2
    return 1


class GestureManagerUI:
    """Own the single Settings window and render its navigable pages."""

    def __init__(
        self,
        root: ttk.Window,
        controller: RuntimeController,
        request_command: Callable[[RuntimeCommand], None],
        tray_status: Callable[[], str],
        camera_manager: CameraManager,
        hand_engine: HandLandmarkEngine,
        on_gesture_action_state: Callable[[], None] | None = None,
    ) -> None:
        self.root = root
        self.controller = controller
        self.request_command = request_command
        self.tray_status = tray_status
        self.camera_manager = camera_manager
        self.hand_engine = hand_engine
        self.on_gesture_action_state = on_gesture_action_state or (lambda: None)
        legacy_gestures = GestureStore(default_gesture_directory())
        legacy_model_store = GestureModelStore(
            GestureModelStore.path_from_gesture_directory(legacy_gestures.root)
        )
        legacy_action_store = ActionMappingStore(ActionMappingStore.default_path())
        self.profile_store = ProfileStore()
        profiles = self.profile_store.list()
        if not profiles:
            default_profile = self.profile_store.create("Default")
            legacy_datasets = legacy_gestures.list()
            try:
                legacy_mappings = legacy_action_store.load()
            except (OSError, ValueError):
                legacy_mappings = {}
            self.profile_store.update_configuration(
                default_profile["id"], datasets=legacy_datasets, model=None,
                mappings=legacy_mappings, thresholds={},
            )
            if legacy_model_store.path.is_file():
                profile_model_path = (self.profile_store._directory(default_profile["id"])
                                      / "models" / "custom_gestures.json")
                profile_model_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(legacy_model_store.path, profile_model_path)
                default_profile["model_ref"] = "models/custom_gestures.json"
                self.profile_store.save(default_profile)
            profiles = self.profile_store.list()
        selected_id = self.profile_store.active_id()
        self._active_profile = next((item for item in profiles if item["id"] == selected_id), profiles[0])
        self.profile_store.set_active(self._active_profile["id"])
        self._profile_mode_preference = True
        self._profile_mode_error: str | None = None
        # Must exist before profile storage loads the saved Stop Gesture.
        self._stop_gesture_id: str | None = None
        self._pause_gesture_id: str | None = None
        self._resume_gesture_id: str | None = None
        self._stop_gesture_error: str | None = None
        self._gesture_controls = GestureControlRouter()
        self._gesture_actions_paused = False
        self._gesture_actions_paused_since: float | None = None
        self._configure_profile_storage()
        apply_readable_text_styles(self.root.style, self.root.theme_mode)
        self.action_dispatcher = ActionDispatcher()
        self._action_mappings: dict[str, ActionMapping] = {}
        self._action_store_error: str | None = None
        try:
            self._action_mappings = self.action_store.load()
        except (OSError, ValueError) as exc:
            self._action_store_error = str(exc)
        self._service_events: queue.Queue[tuple[object, ...]] = queue.Queue()
        self._live_results: queue.Queue[RecognitionEvent] = queue.Queue(maxsize=8)
        self._model_load_thread: Thread | None = None
        self._model_load_threads: list[Thread] = []
        self._training_thread: Thread | None = None
        self._training_running = False
        self._training_message = "Checking for a saved local model…"
        self._saved_model: PortableGestureModel | None = None
        self._model_current = False
        self._training_outcome: TrainingOutcome | None = None
        self._live_classifier: LiveRecognitionWorker | None = None
        self._recognition_enabled_vars: dict[str, tk.BooleanVar] = {}
        self._recognition_threshold_vars: dict[str, tk.StringVar] = {}
        self._recognition_cooldown_vars: dict[str, tk.StringVar] = {}
        self._recognition_cooldown_preferences: dict[str, float] = {}
        self._activation_frames_var: tk.StringVar | None = None
        self._cooldown_var: tk.StringVar | None = None
        self._recognition_paused = False
        self._recognition_emergency_stopped = False
        self._live_test_active = False
        self._camera_index = 0
        self._recognition_preferences: dict[str, tuple[bool, float]] = {}
        self._recognition_activation_frames = 3
        self._recognition_cooldown_seconds = 1.0
        self._training_status_label: ttk.Label | None = None
        self._live_status_label: ttk.Label | None = None
        self._service_poll_scheduled = False
        self._studio_session: GestureRecordingSession | None = None
        self._studio_stage = "details"
        self._studio_preview: CameraPreview | None = None
        self._studio_preview_var: tk.BooleanVar | None = None
        self._studio_status_label: ttk.Label | None = None
        self._recording_progress_label: ttk.Label | None = None
        self._studio_error_var: tk.StringVar | None = None
        self._studio_name_var: tk.StringVar | None = None
        self._studio_hand_count_var: tk.StringVar | None = None
        self._studio_type_var: tk.StringVar | None = None
        self._dynamic_label_var: tk.StringVar | None = None
        self._dynamic_session: DynamicRecordingSession | None = None
        self._dynamic_recorder: DynamicStudioRecorder | None = None
        self._dynamic_training_thread: Thread | None = None
        self._dynamic_training_running = False
        self._dynamic_status_label: ttk.Label | None = None
        self._dynamic_report_label: ttk.Label | None = None
        self._dynamic_saved_model: DynamicGestureModel | None = None
        self._last_recorded_processed_frames = 0
        self._scan_results: queue.Queue[tuple[CameraDevice, ...]] = queue.Queue(maxsize=1)
        self._camera_devices: tuple[CameraDevice, ...] = ()
        self._camera_status_label: ttk.Label | None = None
        self._camera_message: ttk.Label | None = None
        self._camera_preview: CameraPreview | None = None
        self._camera_preview_var: tk.BooleanVar | None = None
        self._camera_index_var: tk.StringVar | None = None
        self._camera_resolution_var: tk.StringVar | None = None
        self._processing_rate_var: tk.StringVar | None = None
        self._engine_status_label: ttk.Label | None = None
        self._camera_last_sequence = -1
        self._camera_poll_scheduled = False
        self.current_page = "Dashboard"
        self._nav_buttons: dict[str, ttk.Button] = {}
        self._theme_mode = tk.StringVar(root, value=root.theme_mode)
        self._theme_notice = tk.StringVar(root, value="")
        self._state_badge: ttk.Label | None = None
        self._theme_button: ttk.Button | None = None
        self._action_shortcut_vars: dict[str, tk.StringVar] = {}
        self._action_enabled_vars: dict[str, tk.BooleanVar] = {}
        self._action_status_label: ttk.Label | None = None
        self._dry_run_var = tk.BooleanVar(root, value=True)
        self._real_input_confirmed = False
        self._stop_gesture_trigger = StopGestureTrigger()
        self._stop_gesture_choice: tk.StringVar | None = None
        self._pause_gesture_choice: tk.StringVar | None = None
        self._resume_gesture_choice: tk.StringVar | None = None
        self._stop_gesture_choices: dict[str, str | None] = {}
        self._gesture_control_status_label: ttk.Label | None = None
        self._gesture_control_toggle_button: ttk.Button | None = None
        self._action_record_dialog: tk.Toplevel | None = None
        self._action_record_target: tk.StringVar | None = None
        self._action_record_status: ttk.Label | None = None
        self._action_record_button: ttk.Button | None = None
        self._action_recorded_shortcut: str | None = None
        self._action_record_queue: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=1)
        self._action_record_listener: object | None = None
        self._focus_detector = WindowsFocusDetector()
        self._adaptive_poll_scheduled = False
        self._adaptive_mode = "stopped"
        self._inactivity_tracker = FrameInactivityTracker(timeout_seconds=300.0)
        self._inactivity_last_frame_sequence: int | None = None
        self._performance_sampler = ProcessMetricsSampler()
        self._fps_sample_time = monotonic()
        self._fps_sample_frames = 0
        self._observed_fps = 0.0
        self._profile_name_var: tk.StringVar | None = None
        self._profile_status: ttk.Label | None = None
        self._adaptive_status_label: ttk.Label | None = None
        self._profile_generation = 0
        self._profile_io_thread: Thread | None = None
        self._wake_detection_var = tk.BooleanVar(root, value=False)
        self._wake_worker: WakeMotionWorker | None = None
        self._wake_resume_pending = False
        self._strict_runtime_pause = False
        self._last_runtime_action = "No gesture action triggered yet."
        self._layout_refresh_pending = False
        self._dashboard_layout: tuple[int, int] | None = None
        self._dashboard_reflow_pending = False

        self.root.option_add("*Font", ("Segoe UI", 10))
        self._build_shell()
        self._bind_shortcuts()
        self.show_page("Dashboard")
        self._load_saved_model_async()
        self._schedule_service_poll()
        # Profile preference is persistent, but real input is never enabled
        # silently on restart. Ask after the window has been constructed.
        self.root.after(750, self._apply_saved_execution_preference)

    def _configure_profile_storage(self) -> None:
        profile_dir = self.profile_store._directory(self._active_profile["id"])
        self.gesture_store = GestureStore(profile_dir / "datasets")
        self.model_store = GestureModelStore(profile_dir / "models" / "custom_gestures.json")
        self.trainer = GestureTrainer(self.model_store)
        self.dynamic_store = DynamicGestureStore(profile_dir / "dynamic_gestures" / "datasets")
        self.dynamic_model_store = DynamicGestureModelStore(profile_dir / "dynamic_gestures" / "model.json")
        self.action_store = ActionMappingStore(profile_dir / "settings" / "actions.json")
        self.execution_mode_store = ExecutionModeStore(profile_dir / "settings" / "execution_mode.json")
        self.stop_gesture_store = StopGestureStore(profile_dir / "settings" / "stop_gesture.json")
        current_theme = self._validated_theme_mode(
            getattr(getattr(self, "root", None), "theme_mode", "light"))
        try:
            manifest = json.loads(self.profile_store._manifest_path(self._active_profile["id"])
                                  .read_text(encoding="utf-8"))
            if manifest.get("version") == 1:
                # Migrate legacy per-profile settings into the version-2
                # version-3 profile manifest without enabling real input.
                try:
                    dry_run = self.execution_mode_store.load_dry_run()
                    self._profile_mode_error = None
                except (OSError, ValueError) as exc:
                    dry_run = True
                    self._profile_mode_error = str(exc)
                try:
                    stop_gesture_id = self.stop_gesture_store.load()
                    self._stop_gesture_error = None
                except (OSError, ValueError) as exc:
                    stop_gesture_id = None
                    self._stop_gesture_error = str(exc)
                if self._profile_mode_error is None and self._stop_gesture_error is None:
                    self._active_profile = self.profile_store.update_execution_settings(
                        self._active_profile["id"], dry_run=dry_run,
                        stop_gesture_id=stop_gesture_id)
                    self._active_profile = self.profile_store.update_theme_mode(
                        self._active_profile["id"],
                        theme_mode=current_theme)
                self._profile_mode_preference = dry_run
                self._stop_gesture_id = stop_gesture_id
                self._pause_gesture_id = None
                self._resume_gesture_id = None
            elif manifest.get("version") == 2:
                # Older profiles had no saved theme; preserve the theme active
                # when opening them and add it to their version-3 manifest.
                self._active_profile = self.profile_store.update_theme_mode(
                    self._active_profile["id"],
                    theme_mode=current_theme)
                self._profile_mode_preference = self._active_profile["dry_run"]
                self._stop_gesture_id = self._active_profile["stop_gesture_id"]
                self._pause_gesture_id = self._active_profile.get("pause_gesture_id")
                self._resume_gesture_id = self._active_profile.get("resume_gesture_id")
                self._profile_mode_error = None
                self._stop_gesture_error = None
            else:
                self._profile_mode_preference = self._active_profile["dry_run"]
                self._stop_gesture_id = self._active_profile["stop_gesture_id"]
                self._pause_gesture_id = self._active_profile.get("pause_gesture_id")
                self._resume_gesture_id = self._active_profile.get("resume_gesture_id")
                self._profile_mode_error = None
                self._stop_gesture_error = None
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError) as exc:
            self._profile_mode_preference = True
            self._profile_mode_error = str(exc)
            self._stop_gesture_id = None
            self._pause_gesture_id = None
            self._resume_gesture_id = None
            self._stop_gesture_error = str(exc)
        self._gesture_controls.configure(
            pause_gesture_id=self._pause_gesture_id,
            resume_gesture_id=self._resume_gesture_id,
            stop_gesture_id=self._stop_gesture_id,
        )
        if not self._profile_mode_error:
            theme_mode = self._active_profile.get("theme_mode", "light")
            root = getattr(self, "root", None)
            if root is not None:
                root.theme_mode = theme_mode
                apply_windows_titlebar_theme(root, theme_mode)
                style = getattr(root, "style", None)
                if style is not None:
                    apply_readable_text_styles(style, theme_mode)
            theme_var = getattr(self, "_theme_mode", None)
            if theme_var is not None:
                theme_var.set(theme_mode)
        try:
            self._action_mappings = {
                item["gesture_id"]: ActionMapping(**item)
                for item in self._active_profile["action_mappings"]
            }
            self.action_store.save(self._action_mappings)
            self._action_store_error = None
        except (OSError, ValueError) as exc:
            self._action_mappings = {}
            self._action_store_error = str(exc)

    @staticmethod
    def _validated_theme_mode(mode: object) -> str:
        return mode if mode in ("light", "dark") else "light"

    def _build_shell(self) -> None:
        self.shell = ttk.Frame(self.root, padding=0)
        self.shell.pack(fill="both", expand=True)
        self.shell.columnconfigure(1, weight=1)
        self.shell.rowconfigure(0, weight=1)

        self.sidebar = ttk.Frame(self.shell, padding=(20, 22), bootstyle="secondary")
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.columnconfigure(0, weight=1)
        self.sidebar.rowconfigure(3, weight=1)

        brand = ttk.Frame(self.sidebar, bootstyle="secondary")
        brand.grid(row=0, column=0, sticky="ew", pady=(0, 28))
        self._brand_icon_photo = ImageTk.PhotoImage(
            load_app_icon().resize((52, 52), Image.Resampling.LANCZOS), master=self.root
        )
        ttk.Label(brand, image=self._brand_icon_photo).pack(side="left", padx=(0, 11))
        brand_words = ttk.Frame(brand, bootstyle="secondary")
        brand_words.pack(side="left", fill="y")
        ttk.Label(
            brand_words,
            text="Gesture Manager",
            bootstyle="inverse-secondary",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            brand_words,
            text="LOCAL DESKTOP",
            bootstyle="inverse-secondary",
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", pady=(2, 0))

        ttk.Label(
            self.sidebar,
            text="WORKSPACE",
            bootstyle="inverse-secondary",
            font=("Segoe UI", 8, "bold"),
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        nav = ttk.Frame(self.sidebar, bootstyle="secondary")
        nav.grid(row=2, column=0, sticky="new")
        for row, (page, glyph) in enumerate(PAGES):
            button = ttk.Button(
                nav,
                text=f"{glyph}   {page}",
                command=lambda name=page: self.show_page(name),
                style="GMNav.TButton",
                width=24,
                takefocus=True,
                padding=(12, 10),
            )
            button.grid(row=row, column=0, sticky="ew", pady=3)
            self._nav_buttons[page] = button

        footer = ttk.Frame(self.sidebar, bootstyle="secondary")
        footer.grid(row=3, column=0, sticky="sew", pady=(28, 0))
        ttk.Separator(footer).pack(fill="x", pady=(0, 12))
        ttk.Label(
            footer,
            text="●  Local runtime",
            bootstyle="inverse-secondary",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            footer,
            text="Camera access is available on the Camera page",
            bootstyle="inverse-secondary",
            wraplength=195,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(5, 0))

        self.main = ttk.Frame(self.shell, padding=(28, 22, 30, 24))
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.columnconfigure(0, weight=1)
        self.main.rowconfigure(1, weight=1)

        self.header = ttk.Frame(self.main)
        self.header.grid(row=0, column=0, sticky="ew", pady=(0, 22))
        self.header.columnconfigure(0, weight=1)
        title_area = ttk.Frame(self.header)
        title_area.grid(row=0, column=0, sticky="w")
        self.page_title = ttk.Label(
            title_area,
            text="",
            font=("Segoe UI", 22, "bold"),
        )
        self.page_title.pack(anchor="w")
        self.page_subtitle = ttk.Label(title_area, text="", bootstyle="secondary")
        self.page_subtitle.pack(anchor="w", pady=(4, 0))

        tools = ttk.Frame(self.header)
        tools.grid(row=0, column=1, sticky="e", padx=(16, 0))
        self._state_badge = ttk.Label(
            tools,
            text="",
            bootstyle="secondary",
            padding=(11, 7),
        )
        self._state_badge.pack(side="left", padx=(0, 9))
        self._theme_button = ttk.Button(
            tools,
            text="",
            command=self.toggle_theme,
            style="GMReadable.TButton",
            takefocus=True,
            padding=(11, 7),
        )
        self._theme_button.pack(side="left")

        self.page_viewport = ttk.Frame(self.main)
        self.page_viewport.grid(row=1, column=0, sticky="nsew")
        self.page_viewport.columnconfigure(0, weight=1)
        self.page_viewport.rowconfigure(0, weight=1)
        self.page_canvas = tk.Canvas(
            self.page_viewport, highlightthickness=0, borderwidth=0,
            background=self.root.style.colors.bg,
        )
        page_scrollbar = ttk.Scrollbar(
            self.page_viewport, orient="vertical", command=self.page_canvas.yview
        )
        self.page_canvas.configure(yscrollcommand=page_scrollbar.set)
        self.page_canvas.grid(row=0, column=0, sticky="nsew")
        page_scrollbar.grid(row=0, column=1, sticky="ns")
        self.page_body = ttk.Frame(self.page_canvas)
        self._page_window = self.page_canvas.create_window(
            (0, 0), window=self.page_body, anchor="nw"
        )
        self.page_body.columnconfigure(0, weight=1)
        self.page_body.bind("<Configure>", lambda _event: self.page_canvas.configure(
            scrollregion=self.page_canvas.bbox("all")
        ))
        self.page_canvas.bind("<Configure>", self._on_page_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._scroll_page)
        self.root.bind("<Configure>", self._on_window_configure, add="+")

        self._refresh_header_state()
        self._refresh_theme_button()

    def _bind_shortcuts(self) -> None:
        for number, (page, _glyph) in enumerate(PAGES, start=1):
            self.root.bind_all(
                f"<Control-Key-{number}>",
                lambda _event, selected=page: self.show_page(selected),
            )
        self.root.bind_all("<Control-Shift-T>", lambda _event: self.toggle_theme())

    def _scroll_page(self, event: tk.Event) -> str:
        if event.widget.winfo_toplevel() is not self.root:
            return ""
        widget = event.widget
        while widget is not None:
            if isinstance(widget, ttk.Treeview):
                return ""
            widget = getattr(widget, "master", None)
        units = int(-event.delta / 120)
        if units == 0 and event.delta:
            units = -1 if event.delta > 0 else 1
        self.page_canvas.yview_scroll(units, "units")
        return "break"

    def _on_window_configure(self, event: tk.Event) -> None:
        if event.widget is not self.root or self.current_page != "Dashboard":
            return
        if event.width < 100:
            return
        width = event.width
        layout = (dashboard_column_count(width), 2 if width >= 1040 else 1)
        if (self._dashboard_layout is not None and layout != self._dashboard_layout
                and not self._dashboard_reflow_pending):
            self._dashboard_reflow_pending = True
            self.root.after_idle(self._reflow_dashboard)

    def _reflow_dashboard(self) -> None:
        self._dashboard_reflow_pending = False
        if self.current_page != "Dashboard":
            return
        width = max(self.root.winfo_width(), 800)
        layout = (dashboard_column_count(width), 2 if width >= 1040 else 1)
        if layout != self._dashboard_layout:
            self.show_page("Dashboard")

    def _on_page_canvas_configure(self, event: tk.Event) -> None:
        self.page_canvas.itemconfigure(self._page_window, width=event.width)
        if not self._layout_refresh_pending:
            self._layout_refresh_pending = True
            self.root.after_idle(self._refresh_page_layout)

    def _refresh_page_layout(self) -> None:
        """Fit text to its current card width after the window is resized."""
        self._layout_refresh_pending = False
        if not self.page_body.winfo_exists():
            return
        self.page_canvas.configure(background=self.root.style.colors.bg)
        pending = list(self.page_body.winfo_children())
        while pending:
            widget = pending.pop()
            pending.extend(widget.winfo_children())
            try:
                current = int(widget.cget("wraplength"))
            except (tk.TclError, TypeError, ValueError):
                continue
            if current <= 0:
                continue
            available = widget.master.winfo_width() - 32
            if available <= 0:
                continue
            target = max(150, available)
            if abs(current - target) > 3:
                try:
                    widget.configure(wraplength=target)
                except tk.TclError:
                    pass

    def show_page(self, page: str) -> None:
        if page not in PAGE_COPY:
            return
        if self.current_page == "Gesture Studio" and page != "Gesture Studio":
            if self._dynamic_recorder is not None and self._dynamic_recorder.recording:
                self._dynamic_recorder.cancel()
            if self._studio_stage in ("record", "dynamic_record") and not self._live_test_active:
                self.hand_engine.stop()
        self.current_page = page
        self.page_title.configure(text=page)
        self.page_subtitle.configure(text=PAGE_COPY[page])
        for name, button in self._nav_buttons.items():
            button.configure(
                style="GMNavActive.TButton" if name == page else "GMNav.TButton"
            )
        for child in self.page_body.winfo_children():
            child.destroy()
        for row in range(20):
            self.page_body.rowconfigure(row, weight=0)
            self.page_body.columnconfigure(row, weight=0)
        self.page_body.columnconfigure(0, weight=1)
        self.page_body.grid_propagate(True)
        self._studio_preview = None
        self._studio_status_label = None
        self._recording_progress_label = None
        self._dynamic_status_label = None
        self._dynamic_report_label = None

        renderer = {
            "Dashboard": self._render_dashboard,
            "My Gestures": self._render_gestures,
            "Gesture Studio": self._render_studio,
            "Actions": self._render_actions,
            "Profiles": self._render_profiles,
            "Camera": self._render_camera,
            "General Settings": self._render_general_settings,
            "About & Diagnostics": self._render_about,
        }[page]
        renderer()
        # ttkbootstrap builds semantic outline styles lazily as widgets are
        # created; reapply contrast overrides after each page is rendered.
        apply_readable_text_styles(self.root.style, self.root.theme_mode)
        if page == "Dashboard":
            width = max(self.root.winfo_width(), 800)
            self._dashboard_layout = (dashboard_column_count(width),
                                      2 if width >= 1040 else 1)
        self.page_canvas.yview_moveto(0)
        self.root.after_idle(self._refresh_page_layout)
        self._refresh_header_state()
        if page in ("Camera", "Gesture Studio"):
            self._schedule_camera_poll()

    def refresh_state(self) -> None:
        """Refresh state after an explicit runtime command, not on a timer."""
        self._refresh_header_state()
        if self.current_page == "Dashboard":
            self.show_page("Dashboard")
        elif self.current_page == "About & Diagnostics":
            self._refresh_diagnostic_state()

    def _refresh_diagnostic_state(self) -> None:
        state_label = getattr(self, "_diagnostic_state", None)
        if state_label is not None:
            state_label.configure(text=self.controller.state.value.title())

    def _refresh_header_state(self) -> None:
        if self._state_badge is None:
            return
        state = self.controller.state
        styles = {
            RuntimeState.ACTIVE: "success",
            RuntimeState.PAUSED: "warning",
            RuntimeState.STARTING: "info",
            RuntimeState.STOPPING: "warning",
            RuntimeState.ERROR: "danger",
            RuntimeState.STOPPED: "secondary",
        }
        title = ("Active · shortcuts paused" if state is RuntimeState.ACTIVE
                 and self._gesture_actions_paused else state.value.title())
        style = ("warning" if state is RuntimeState.ACTIVE and self._gesture_actions_paused
                 else styles.get(state, "secondary"))
        self._state_badge.configure(text=f"\u25cf  {title}",
                                    bootstyle=style)

    @property
    def gesture_actions_paused(self) -> bool:
        return self._gesture_actions_paused

    def _refresh_gesture_control_state(self) -> None:
        self._gesture_controls.set_actions_paused(self._gesture_actions_paused)
        status = getattr(self, "_gesture_control_status_label", None)
        if status is not None and status.winfo_exists():
            running = self._live_test_active
            status.configure(
                text=("Stopped · shortcut state resets to Active on next start." if not running else
                      "Paused · ordinary shortcuts blocked; camera and recognition remain active."
                      if self._gesture_actions_paused else
                      "Active · eligible ordinary shortcut mappings may run."),
                bootstyle=("secondary" if not running else
                           "warning" if self._gesture_actions_paused else "success"))
        toggle = getattr(self, "_gesture_control_toggle_button", None)
        if toggle is not None and toggle.winfo_exists():
            toggle.configure(text="Resume shortcuts" if self._gesture_actions_paused
                              else "Pause shortcuts",
                              state="normal" if self._live_test_active else "disabled")
        self._refresh_header_state()
        if self.current_page == "Dashboard":
            self.show_page("Dashboard")
        self.on_gesture_action_state()

    def _toggle_gesture_actions_manually(self) -> None:
        if not self._live_test_active:
            if self._gesture_control_status_label is not None:
                self._gesture_control_status_label.configure(
                    text="Start runtime before pausing or resuming shortcut execution.",
                    bootstyle="warning")
            return
        if (self._gesture_actions_paused
                and self._gesture_actions_paused_since is not None
                and monotonic() - self._gesture_actions_paused_since < 1.0):
            remaining = max(0.0, 1.0 - (monotonic() - self._gesture_actions_paused_since))
            if self._gesture_control_status_label is not None:
                self._gesture_control_status_label.configure(
                    text=f"Resume is available in {remaining:.1f} seconds.", bootstyle="warning")
            return
        self._gesture_actions_paused = not self._gesture_actions_paused
        self._gesture_actions_paused_since = monotonic() if self._gesture_actions_paused else None
        self._gesture_controls.set_actions_paused(self._gesture_actions_paused)
        self._last_runtime_action = (
            "Manual pause · ordinary shortcuts blocked; recognition remains active."
            if self._gesture_actions_paused else
            "Manual resume · ordinary shortcut eligibility restored.")
        self._refresh_gesture_control_state()

    def _save_pause_resume_gestures(self) -> None:
        pause_name = self._pause_gesture_choice.get() if self._pause_gesture_choice else "Disabled"
        resume_name = self._resume_gesture_choice.get() if self._resume_gesture_choice else "Disabled"
        if pause_name not in self._stop_gesture_choices or resume_name not in self._stop_gesture_choices:
            messagebox.showerror("Invalid control gesture", "Select gestures from the current profile.",
                                 parent=self.root)
            return
        pause_id = self._stop_gesture_choices[pause_name]
        resume_id = self._stop_gesture_choices[resume_name]
        if self._stop_gesture_id is not None and self._stop_gesture_id in {pause_id, resume_id}:
            messagebox.showerror(
                "Stop Gesture must be distinct",
                "Stop Gesture must use a different custom gesture. Pause and Resume may share one gesture; it toggles between the two states.",
                parent=self.root)
            return
        try:
            updated = self.profile_store.update_execution_settings(
                self._active_profile["id"],
                dry_run=self._profile_mode_preference,
                stop_gesture_id=self._stop_gesture_id,
                pause_gesture_id=pause_id,
                resume_gesture_id=resume_id,
            )
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not save control gestures", str(exc), parent=self.root)
            return
        self._active_profile = updated
        self._pause_gesture_id = pause_id
        self._resume_gesture_id = resume_id
        was_paused = self._gesture_actions_paused
        self._gesture_controls.configure(
            pause_gesture_id=pause_id,
            resume_gesture_id=resume_id,
            stop_gesture_id=self._stop_gesture_id,
        )
        self._gesture_actions_paused = was_paused
        self._gesture_controls.set_actions_paused(was_paused)
        worker = self._live_classifier
        if worker is not None and worker.is_alive:
            for gesture_id in (pause_id, resume_id):
                if gesture_id is not None:
                    worker.gate.update_gesture(gesture_id, enabled=True)
                    self._recognition_preferences[gesture_id] = (
                        True, self._recognition_preferences.get(gesture_id, (True, 0.0))[1])
                    variable = self._recognition_enabled_vars.get(gesture_id)
                    if variable is not None:
                        variable.set(True)
        self._sync_active_profile()
        self._refresh_gesture_control_state()

    def _refresh_theme_button(self) -> None:
        if self._theme_button is None:
            return
        mode = self.root.theme_mode
        self._theme_mode.set(mode)
        self._theme_button.configure(
            text="Dark mode" if mode == "light" else "Light mode"
        )
        self.page_canvas.configure(background=self.root.style.colors.bg)

    def toggle_theme(self) -> None:
        self._theme_mode.set("dark" if self.root.theme_mode == "light" else "light")
        self._apply_theme_choice()
        self._refresh_theme_button()
        if self.current_page == "General Settings":
            self.show_page("General Settings")

    def _card(self, parent: ttk.Frame, title: str, **grid_options: object) -> ttk.Labelframe:
        card = ttk.Labelframe(
            parent,
            text=title,
            padding=17,
            bootstyle="secondary",
        )
        card.grid(**grid_options)
        card.columnconfigure(0, weight=1)
        return card

    def _render_dashboard(self) -> None:
        self.page_body.columnconfigure(0, weight=1)
        overview = ttk.Labelframe(
            self.page_body,
            text="Runtime overview",
            padding=(20, 17),
            bootstyle="primary",
        )
        overview.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        overview.columnconfigure(0, weight=1)
        state = self.controller.state
        ttk.Label(
            overview,
            text=("Active · shortcuts paused · recognition running"
                  if state is RuntimeState.ACTIVE and self._gesture_actions_paused else
                  f"{state.value.title()} · local gesture runtime"),
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w")
        engine_state = self.hand_engine.status.state
        detail = (self.controller.error or
                  (self.hand_engine.status.error if engine_state is EngineState.ERROR else None) or
                  self._runtime_description(state))
        if state is RuntimeState.ACTIVE and self._gesture_actions_paused:
            detail = "Camera and recognition stay active. Ordinary shortcuts are paused; use Resume Gesture or Resume shortcuts to continue."
        self._runtime_overview_detail = ttk.Label(
            overview,
            text=detail,
            wraplength=760,
            justify="left",
        )
        self._runtime_overview_detail.grid(row=1, column=0, sticky="w", pady=(6, 0))

        actions = ttk.Frame(overview)
        actions.grid(row=2, column=0, sticky="w", pady=(15, 0))
        if state in (RuntimeState.STOPPED, RuntimeState.ERROR):
            self._action_button(actions, "Start runtime", RuntimeCommand.START, "primary")
        elif state is RuntimeState.ACTIVE:
            self._action_button(
                actions,
                "Resume shortcuts" if self._gesture_actions_paused else "Pause shortcuts",
                self._toggle_gesture_actions_manually,
                "success" if self._gesture_actions_paused else "warning",
            )
            self._action_button(actions, "Pause service", RuntimeCommand.PAUSE, "secondary outline")
        elif state is RuntimeState.PAUSED:
            self._action_button(actions, "Resume service", RuntimeCommand.RESUME, "primary")
        if state is not RuntimeState.STOPPED:
            self._action_button(actions, "Stop service", RuntimeCommand.STOP, "secondary outline")

        cards = ttk.Frame(self.page_body)
        cards.grid(row=1, column=0, sticky="ew", pady=(0, 16))
        card_width = max(self.root.winfo_width(), 800)
        columns = dashboard_column_count(card_width)
        for col in range(columns):
            cards.columnconfigure(col, weight=1, uniform="dashboard")
        self._summary_card(
            cards,
            0,
            "Local camera",
            "Available on request",
            "Capture and hand landmark preview start only when you request them.",
            columns=columns,
        )
        try:
            gesture_count = len(self.gesture_store.list())
            gesture_value = f"{gesture_count} saved"
            gesture_detail = "Your custom feature datasets are stored on this device."
        except (OSError, ValueError):
            gesture_count = 0
            gesture_value = "Unavailable"
            gesture_detail = "The local gesture dataset folder could not be read."
        self._summary_card(
            cards,
            1,
            "Gesture library",
            gesture_value,
            gesture_detail,
            columns=columns,
        )
        profile_value, profile_detail = self._dashboard_profile_summary(gesture_count)
        self._summary_card(
            cards,
            2,
            "Active profile",
            profile_value,
            profile_detail,
            columns=columns,
        )

        lower = ttk.Frame(self.page_body)
        lower.grid(row=2, column=0, sticky="ew")
        lower_columns = 2 if card_width >= 1040 else 1
        for col in range(lower_columns):
            lower.columnconfigure(col, weight=1, uniform="dashboard-lower")
        activity = self._card(lower, "Recent activity", row=0, column=0, sticky="nsew",
                              padx=(0, 8) if lower_columns == 2 else 0,
                              pady=(0, 0) if lower_columns == 2 else (0, 12))
        self._dashboard_activity = ttk.Label(
            activity,
            text=self._last_runtime_action,
            wraplength=360,
        )
        self._dashboard_activity.grid(row=0, column=0, sticky="w", pady=(2, 0))
        shortcuts = self._card(lower, "Quick access", row=0,
                               column=1 if lower_columns == 2 else 0,
                               sticky="nsew", padx=(8, 0) if lower_columns == 2 else 0)
        ttk.Button(
            shortcuts,
            text="Create a custom gesture  →",
            command=lambda: self.show_page("My Gestures"),
            bootstyle="primary outline",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 7))
        ttk.Button(
            shortcuts,
            text="Open camera setup  →",
            command=lambda: self.show_page("Camera"),
            bootstyle="secondary outline",
        ).grid(row=1, column=0, sticky="ew")

    def _summary_card(
        self, parent: ttk.Frame, column: int, title: str, value: str, detail: str,
        *, columns: int,
    ) -> None:
        card = self._card(
            parent,
            title,
            row=column // columns,
            column=column % columns,
            sticky="nsew",
            padx=(0 if column % columns == 0 else 6,
                  0 if column % columns == columns - 1 else 6),
            pady=(0, 10) if column >= columns else 0,
        )
        ttk.Label(card, text=value, font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(card, text=detail, wraplength=215, justify="left").grid(
            row=1, column=0, sticky="nw", pady=(7, 0)
        )

    def _dashboard_profile_summary(self, gesture_count: int) -> tuple[str, str]:
        """Describe the loaded profile using the state available locally."""
        name = str(self._active_profile.get("name", "Unnamed profile"))
        if self._profile_mode_error:
            execution = "execution preference unavailable"
        else:
            execution = "Dry-run preference" if self._profile_mode_preference else (
                "Real-input preference · confirmation required")

        if self._stop_gesture_error:
            stop = "Stop Gesture settings unavailable"
        elif self._stop_gesture_id is None:
            stop = "Stop Gesture not selected"
        else:
            stop_name = next((item.get("name") for item in
                              self._active_profile.get("gesture_definitions", ())
                              if item.get("id") == self._stop_gesture_id), None)
            stop = f"Stop Gesture: {stop_name}" if stop_name else "Stop Gesture unavailable"

        def control_name(gesture_id: str | None) -> str:
            if gesture_id is None:
                return "off"
            return next((str(item.get("name")) for item in
                         self._active_profile.get("gesture_definitions", ())
                         if item.get("id") == gesture_id), "unavailable")
        pause_resume = (f"Pause: {control_name(self._pause_gesture_id)} · "
                        f"Resume: {control_name(self._resume_gesture_id)}")

        if self._training_running:
            model = "model training"
        elif self._model_current:
            model = "model ready"
        elif self._saved_model is not None:
            model = "model needs retraining"
        elif any(thread.is_alive() for thread in self._model_load_threads):
            model = "model loading"
        else:
            model = "model not trained"

        details = (f"{gesture_count} custom gestures · {model} · {stop} · "
                   f"{pause_resume} · {execution}")
        return name, details

    @staticmethod
    def _runtime_description(state: RuntimeState) -> str:
        if state is RuntimeState.ACTIVE:
            return "Local camera capture, custom gesture recognition, and configured actions are active."
        if state is RuntimeState.PAUSED:
            return "The application lifecycle is paused. No camera is active."
        if state is RuntimeState.STARTING:
            return "Starting the local application runtime."
        if state is RuntimeState.STOPPING:
            return "Stopping the local runtime and releasing its resources."
        return "The application runtime is stopped. No camera is active."

    def _action_button(
        self,
        parent: ttk.Frame,
        label: str,
        command: RuntimeCommand | Callable[[], None],
        style: str,
    ) -> None:
        callback = (lambda: self.request_command(command)
                    if isinstance(command, RuntimeCommand) else command())
        ttk.Button(
            parent,
            text=label,
            command=callback,
            bootstyle=style,
            takefocus=True,
            padding=(13, 8),
        ).pack(side="left", padx=(0, 8))

    def _render_gestures(self) -> None:
        self.page_body.columnconfigure(0, weight=1)
        header = ttk.Frame(self.page_body)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Your custom gestures", font=("Segoe UI", 17, "bold")).grid(
            row=0, column=0, sticky="w")
        actions = ttk.Frame(header)
        actions.grid(row=0, column=1, sticky="e")
        ttk.Button(actions, text="Train / test model", command=self._open_training_hub,
                   bootstyle="secondary outline", padding=(12, 8)).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Create Gesture", command=self._new_gesture,
                   bootstyle="primary", padding=(14, 8)).pack(side="left")
        collection = ttk.Labelframe(self.page_body, text="Saved datasets", padding=16,
                                    bootstyle="secondary")
        collection.grid(row=1, column=0, sticky="nsew")
        collection.columnconfigure(0, weight=1)
        try:
            datasets = self.gesture_store.list()
        except (OSError, ValueError) as exc:
            ttk.Label(collection, text=f"Could not read local gesture datasets: {exc}",
                      bootstyle="danger", wraplength=700).grid(row=0, column=0, sticky="w")
            return
        if not datasets:
            ttk.Label(collection, text="No saved gestures yet", font=("Segoe UI", 14, "bold")
                      ).grid(row=0, column=0, sticky="w", pady=(2, 6))
            ttk.Label(collection, text="Create a gesture, record several examples, then save its local feature dataset.",
                      wraplength=700).grid(row=1, column=0, sticky="w")
            ttk.Button(collection, text="Create your first gesture", command=self._new_gesture,
                       bootstyle="primary outline").grid(row=2, column=0, sticky="w", pady=(14, 0))
            return
        for row, dataset in enumerate(datasets):
            card = ttk.Frame(collection, padding=(8, 10))
            card.grid(row=row, column=0, sticky="ew", pady=(0, 6))
            card.columnconfigure(0, weight=1)
            ttk.Label(card, text=dataset.name, font=("Segoe UI", 12, "bold")).grid(
                row=0, column=0, sticky="w")
            hands = "one hand" if dataset.hand_count == 1 else "two hands"
            readiness = "Meets minimum example count" if dataset.is_trainable else "Needs at least 3 examples"
            ttk.Label(card, text=f"Static pose  |  {hands}  |  {len(dataset.examples)} examples  |  {readiness}",
                      bootstyle="secondary").grid(row=1, column=0, sticky="w", pady=(3, 0))
            ttk.Button(card, text="Review / add examples",
                       command=lambda item=dataset: self._manage_gesture(item),
                       bootstyle="primary outline").grid(row=0, column=1, rowspan=2, padx=(8, 4))
            ttk.Button(card, text="Rename", command=lambda item=dataset: self._rename_gesture(item),
                       style="GMReadable.TButton").grid(row=0, column=2, rowspan=2, padx=4)
            ttk.Button(card, text="Delete", command=lambda item=dataset: self._delete_gesture(item),
                       bootstyle="danger outline").grid(row=0, column=3, rowspan=2, padx=(4, 0))
            ttk.Separator(collection).grid(row=row + 1, column=0, sticky="ew")

    def _render_studio(self) -> None:
        if self._studio_stage == "dynamic_record":
            self._render_dynamic_record()
        elif self._studio_stage == "dynamic_review":
            self._render_dynamic_review()
        elif self._studio_stage == "dynamic_evaluation":
            self._render_dynamic_evaluation()
        elif self._studio_session is None:
            self._render_studio_details()
        elif self._studio_stage == "record":
            self._render_studio_record()
        elif self._studio_stage == "saved":
            self._render_studio_saved()
        else:
            self._render_studio_review()

    def _open_training_hub(self) -> None:
        try:
            datasets = self.gesture_store.list()
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not load gestures", str(exc), parent=self.root)
            return
        if not datasets:
            messagebox.showinfo("No saved gestures", "Create and save at least two custom gestures before training.", parent=self.root)
            return
        self._studio_session = GestureRecordingSession(datasets[0])
        self._studio_stage = "saved"
        self.show_page("Gesture Studio")

    def _studio_steps(self, active: int) -> None:
        steps = ttk.Frame(self.page_body)
        steps.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        for index, title in enumerate(("1  |  Name", "2  |  Record", "3  |  Review", "4  |  Save", "5  |  Training")):
            ttk.Label(steps, text=title, padding=(9, 6),
                      bootstyle="primary" if index == active else "secondary").pack(side="left", padx=(0, 6))

    def _render_studio_details(self) -> None:
        self._studio_steps(0)
        panel = ttk.Labelframe(self.page_body, text="Create a custom gesture", padding=22,
                               bootstyle="secondary")
        panel.grid(row=1, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="Give your gesture a name", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, sticky="w")
        ttk.Label(panel, text="Create a custom static pose or try the experimental dynamic hand-sequence workflow. Both store numerical landmarks locally.",
                  wraplength=700).grid(row=1, column=0, sticky="w", pady=(5, 18))
        self._studio_name_var = tk.StringVar(self.root)
        self._studio_type_var = tk.StringVar(self.root, value="Static hand pose")
        self._studio_hand_count_var = tk.StringVar(self.root, value="1")
        ttk.Label(panel, text="Name").grid(row=2, column=0, sticky="w")
        name = ttk.Entry(panel, textvariable=self._studio_name_var, width=42)
        name.grid(row=3, column=0, sticky="w", pady=(4, 14))
        ttk.Label(panel, text="Gesture type").grid(row=4, column=0, sticky="w")
        ttk.Combobox(panel, textvariable=self._studio_type_var,
                     values=("Static hand pose", "Dynamic hand sequence (experimental)"),
                     state="readonly", width=38).grid(row=5, column=0, sticky="w", pady=(4, 14))
        ttk.Label(panel, text="Add examples to an existing dynamic label (optional)").grid(row=6, column=0, sticky="w")
        try:
            dynamic_datasets = self.dynamic_store.list()
            dynamic_names = tuple(item.name for item in dynamic_datasets)
        except (OSError, ValueError):
            dynamic_names = ()
        self._dynamic_label_var = tk.StringVar(self.root, value="")
        ttk.Combobox(panel, textvariable=self._dynamic_label_var, values=dynamic_names,
                     state="readonly", width=38).grid(row=7, column=0, sticky="w", pady=(4, 14))
        ttk.Label(panel, text="Static pose hand configuration").grid(row=8, column=0, sticky="w")
        choices = ttk.Frame(panel)
        choices.grid(row=9, column=0, sticky="w", pady=(5, 8))
        for label, value in (("One hand", "1"), ("Two hands", "2")):
            ttk.Radiobutton(choices, text=label, value=value, variable=self._studio_hand_count_var,
                            bootstyle="primary").pack(side="left", padx=(0, 16))
        self._studio_error_var = tk.StringVar(self.root, value="")
        ttk.Label(panel, textvariable=self._studio_error_var, bootstyle="danger").grid(
            row=10, column=0, sticky="w", pady=(0, 8))
        buttons = ttk.Frame(panel)
        buttons.grid(row=11, column=0, sticky="w")
        ttk.Button(buttons, text="Start recording", command=self._begin_gesture,
                   bootstyle="primary", padding=(14, 8)).pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="My Gestures", command=lambda: self.show_page("My Gestures"),
                   bootstyle="secondary outline").pack(side="left")
        name.focus_set()

    def _render_dynamic_record(self) -> None:
        session = self._dynamic_session
        recorder = self._dynamic_recorder
        if session is None or recorder is None:
            self._studio_stage = "details"
            self._render_studio_details()
            return
        self._studio_steps(1)
        panel = ttk.Labelframe(self.page_body,
            text=f"Experimental dynamic sequence · {session.dataset.name}", padding=16,
            bootstyle="warning")
        panel.grid(row=1, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(3, weight=1)
        ttk.Label(panel, text=("Keep exactly one hand visible. Move through the gesture naturally; "
                               "each take saves 8–60 fresh landmark samples. Only numeric features are stored."),
                  wraplength=760, justify="left").grid(row=0, column=0, sticky="w", pady=(0, 9))
        controls = ttk.Frame(panel)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(controls, text="Start local camera", command=self._start_studio_detection,
                   bootstyle="primary").pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Start take", command=self._start_dynamic_take,
                   bootstyle="success").pack(side="left", padx=6)
        ttk.Button(controls, text="Stop and save take", command=self._finish_dynamic_take,
                   bootstyle="primary outline").pack(side="left", padx=6)
        ttk.Button(controls, text="Cancel current take", command=self._cancel_dynamic_take,
                   bootstyle="secondary outline").pack(side="left", padx=6)
        self._studio_preview_var = tk.BooleanVar(self.root, value=True)
        ttk.Checkbutton(controls, text="Show preview", variable=self._studio_preview_var,
                        command=self._schedule_camera_poll, bootstyle="primary").pack(side="right")
        self._dynamic_status_label = ttk.Label(panel, text="Camera stopped; no take is recording.",
                                                wraplength=760)
        self._dynamic_status_label.grid(row=2, column=0, sticky="w", pady=(0, 8))
        self._studio_preview = CameraPreview(panel, bootstyle="secondary")
        self._studio_preview.grid(row=3, column=0, sticky="nsew")
        self._recording_progress_label = ttk.Label(
            panel, text=f"{len(session.dataset.examples)} saved sequences · take {recorder.frame_count}/60 frames")
        self._recording_progress_label.grid(row=4, column=0, sticky="w", pady=(8, 0))
        footer = ttk.Frame(panel)
        footer.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(footer, text="Review sequences", command=self._open_dynamic_review,
                   bootstyle="primary outline").pack(side="left", padx=(0, 8))
        ttk.Button(footer, text="Cancel workflow", command=self._cancel_dynamic_workflow,
                   bootstyle="secondary outline").pack(side="left")
        self._schedule_camera_poll()

    def _render_dynamic_review(self) -> None:
        dataset = self._dynamic_session.dataset if self._dynamic_session else None
        if dataset is None:
            self._studio_stage = "details"
            self._render_studio_details()
            return
        self._studio_steps(2)
        panel = ttk.Labelframe(self.page_body, text=f"Review dynamic sequences · {dataset.name}",
                               padding=18, bootstyle="warning")
        panel.grid(row=1, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        table = ttk.Treeview(panel, columns=("number", "time", "frames"), show="headings", height=9)
        for column, heading, width in (("number", "Take", 90), ("time", "Captured", 230),
                                       ("frames", "Landmark frames", 140)):
            table.heading(column, text=heading)
            table.column(column, width=width, anchor="center")
        table.grid(row=1, column=0, sticky="nsew", pady=(10, 8))
        ttk.Label(panel, text=(f"{len(dataset.examples)} numeric sequences · dataset v{dataset.format_version} · "
                               f"features {dataset.feature_version}"), bootstyle="secondary").grid(
                                   row=0, column=0, sticky="w")
        try:
            counts = " · ".join(f"{item.name}: {len(item.examples)}" for item in self.dynamic_store.list())
        except (OSError, ValueError) as exc:
            counts = f"Could not read label counts: {exc}"
        ttk.Label(panel, text=f"Saved sequences per dynamic label: {counts or 'none yet'}",
                  wraplength=780).grid(row=4, column=0, sticky="w", pady=(10, 0))
        for index, example in enumerate(dataset.examples, 1):
            table.insert("", "end", iid=example.id,
                         values=(index, example.captured_at.replace("T", " ")[:19], len(example.frames)))
        controls = ttk.Frame(panel)
        controls.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(controls, text="Remove selected take",
                   command=lambda: self._remove_dynamic_example(table),
                   bootstyle="danger outline").pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Record more", command=self._return_dynamic_record,
                   bootstyle="secondary outline").pack(side="left", padx=4)
        ttk.Button(controls, text="Train and evaluate labels", command=self._start_dynamic_training,
                   state="disabled" if self._dynamic_training_running else "normal",
                   bootstyle="primary").pack(side="right")
        ttk.Label(panel, text="Evaluation uses the classifier’s grouped sequence holdout. It warns when a safe split cannot be formed; results are not webcam calibrated and may not generalize across recording sessions.",
                  wraplength=780, bootstyle="warning").grid(row=3, column=0, sticky="w", pady=(12, 0))

    def _render_dynamic_evaluation(self) -> None:
        self._studio_steps(4)
        panel = ttk.Labelframe(self.page_body, text="Dynamic sequence evaluation · experimental",
                               padding=18, bootstyle="warning")
        panel.grid(row=1, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        self._dynamic_report_label = ttk.Label(panel, text=(
            self._format_dynamic_evaluation(self._dynamic_saved_model)
            if self._dynamic_saved_model else "Training and grouped holdout evaluation are running locally…"),
            justify="left", wraplength=800)
        self._dynamic_report_label.grid(row=0, column=0, sticky="w")
        ttk.Button(panel, text="Back to sequence review", command=self._open_dynamic_review,
                   bootstyle="secondary outline").grid(row=1, column=0, sticky="w", pady=(12, 0))

    @staticmethod
    def _format_dynamic_evaluation(model: DynamicGestureModel | None) -> str:
        if model is None:
            return "No dynamic evaluation report is available."
        report = model.evaluation
        if not report.evaluated or report.confusion is None:
            return ("Evaluation unavailable: the grouped holdout could not include every label. "
                    "No accuracy score was produced. Add varied independent takes for each label.\n" +
                    "\n".join(report.warnings))
        matrix = report.confusion
        lines = [f"Grouped holdout accuracy: {report.accuracy:.1%} · "
                 f"{'reliability warning' if not report.reliable else 'meets data-count guidance'}",
                 "Per-label results (precision / recall / support):"]
        for index, name in enumerate(report.labels):
            support = sum(matrix[index])
            predicted = sum(row[index] for row in matrix)
            correct = matrix[index][index]
            precision = correct / predicted if predicted else 0.0
            recall = correct / support if support else 0.0
            lines.append(f"  {name}: {precision:.0%} / {recall:.0%} / {support}")
        lines.append("Confusion matrix (actual rows, predicted columns):")
        lines.extend("  " + "  ".join(map(str, row)) for row in matrix)
        lines.append("Not webcam calibrated; recording-session generalization is not guaranteed.")
        lines.extend(report.warnings)
        return "\n".join(lines)

    def _render_studio_record(self) -> None:
        session = self._studio_session
        assert session is not None
        self._studio_steps(1)
        panel = ttk.Labelframe(self.page_body, text=f"Record examples  |  {session.dataset.name}",
                               padding=16, bootstyle="secondary")
        panel.grid(row=1, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(3, weight=1)
        instruction = ("Center one hand in view. Hold the pose steady, then click Record example. Vary the pose slightly across takes."
                       if session.dataset.hand_count == 1 else
                       "Center both hands in view. Hold their pose steady, then click Record example. Vary the pose slightly across takes.")
        ttk.Label(panel, text=instruction, wraplength=750, justify="left").grid(
            row=0, column=0, sticky="w", pady=(0, 9))
        controls = ttk.Frame(panel)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(controls, text="Start local camera", command=self._start_studio_detection,
                   bootstyle="primary").pack(side="left", padx=(0, 6))
        ttk.Button(controls, text="Stop camera", command=self.hand_engine.stop,
                   bootstyle="secondary outline").pack(side="left", padx=6)
        self._studio_preview_var = tk.BooleanVar(self.root, value=True)
        ttk.Checkbutton(controls, text="Show preview", variable=self._studio_preview_var,
                        command=self._schedule_camera_poll, bootstyle="primary").pack(side="left", padx=8)
        ttk.Button(controls, text="Record example", command=self._record_gesture_example,
                   bootstyle="success").pack(side="right")
        self._studio_status_label = ttk.Label(panel, text="Waiting for local camera and landmark model...",
                                              wraplength=760)
        self._studio_status_label.grid(row=2, column=0, sticky="w", pady=(0, 8))
        self._studio_preview = CameraPreview(panel, bootstyle="secondary")
        self._studio_preview.grid(row=3, column=0, sticky="nsew")
        self._recording_progress_label = ttk.Label(
            panel, text=f"{len(session.dataset.examples)} examples  |  collect at least 3 valid examples",
            bootstyle="secondary")
        self._recording_progress_label.grid(row=4, column=0, sticky="w", pady=(8, 0))
        footer = ttk.Frame(panel)
        footer.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(footer, text="Review examples", command=self._open_studio_review,
                   bootstyle="primary outline").pack(side="left", padx=(0, 8))
        ttk.Button(footer, text="Cancel draft", command=self._cancel_studio,
                   bootstyle="secondary outline").pack(side="left")
        self._schedule_camera_poll()

    def _render_studio_review(self) -> None:
        session = self._studio_session
        assert session is not None
        dataset = session.dataset
        self._studio_steps(2)
        panel = ttk.Labelframe(self.page_body, text=f"Review examples  |  {dataset.name}",
                               padding=18, bootstyle="secondary")
        panel.grid(row=1, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        ttk.Label(panel, text=(f"{len(dataset.examples)} numerical examples  |  {dataset.hand_count} hand(s)  |  "
                               f"dataset v{dataset.format_version}  |  features {dataset.feature_version}"),
                  bootstyle="secondary").grid(row=0, column=0, sticky="w", pady=(0, 10))
        table = ttk.Treeview(panel, columns=("number", "time", "size"), show="headings", height=9)
        for column, heading, width in (("number", "Example", 90), ("time", "Captured", 230), ("size", "Numeric features", 150)):
            table.heading(column, text=heading)
            table.column(column, width=width, anchor="center" if column != "time" else "w")
        table.grid(row=1, column=0, sticky="nsew")
        for index, example in enumerate(dataset.examples, 1):
            table.insert("", "end", iid=example.id,
                         values=(index, example.captured_at.replace("T", " ")[:19], len(example.features)))
        detail = ttk.Label(panel, text="Select an example to inspect its normalized feature values.",
                           wraplength=730)
        detail.grid(row=2, column=0, sticky="w", pady=(8, 0))
        def inspect(_event: object = None) -> None:
            selection = table.selection()
            if not selection:
                return
            example = next((item for item in dataset.examples if item.id == selection[0]), None)
            if example is not None:
                preview = ", ".join(f"{value:.3f}" for value in example.features[:9])
                detail.configure(text=f"{len(example.features)} normalized values  |  first values: {preview}")
        table.bind("<<TreeviewSelect>>", inspect)
        actions = ttk.Frame(panel)
        actions.grid(row=3, column=0, sticky="ew", pady=(14, 0))
        ttk.Button(actions, text="Remove selected", command=lambda: self._remove_review_example(table),
                   bootstyle="danger outline").pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Add more examples", command=self._open_studio_record,
                   bootstyle="secondary outline").pack(side="left", padx=4)
        ttk.Button(actions, text="Save dataset", command=self._save_studio_dataset,
                   bootstyle="primary").pack(side="right")
        ttk.Button(actions, text="Cancel changes", command=self._cancel_studio,
                   bootstyle="secondary outline").pack(side="right", padx=8)

    def _render_studio_saved(self) -> None:
        session = self._studio_session
        assert session is not None
        self._studio_steps(4)
        saved = ttk.Labelframe(self.page_body, text="Dataset saved", padding=15,
                               bootstyle="success")
        saved.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(saved, text=f"{session.dataset.name} is saved locally",
                  font=("Segoe UI", 14, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(saved, text=(f"{len(session.dataset.examples)} feature-only examples · dataset "
                               f"v{session.dataset.format_version} · features {session.dataset.feature_version}"),
                  wraplength=720).grid(row=1, column=0, sticky="w", pady=(5, 0))

        training = ttk.Labelframe(self.page_body, text="Train and evaluate custom gestures",
                                  padding=16, bootstyle="secondary")
        training.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        training.columnconfigure(0, weight=1)
        try:
            datasets = self.gesture_store.list()
            ready, readiness_message = self._training_readiness(datasets)
        except (OSError, ValueError) as exc:
            datasets, ready, readiness_message = (), False, f"Could not read gesture datasets: {exc}"
        if self._saved_model is not None and self._model_current:
            model_message = f"Saved model loaded · {len(self._saved_model.labels)} custom labels · trained {self._saved_model.trained_at[:19].replace('T', ' ')} UTC"
        else:
            model_message = self._training_message
        ttk.Label(training, text=model_message, wraplength=760,
                  bootstyle="success" if self._model_current else "secondary").grid(
                      row=0, column=0, sticky="w")
        ttk.Label(training, text=readiness_message, wraplength=760).grid(
            row=1, column=0, sticky="w", pady=(5, 8))
        buttons = ttk.Frame(training)
        buttons.grid(row=2, column=0, sticky="w")
        ttk.Button(buttons, text="Train and evaluate", command=self._start_training,
                   state="normal" if ready and not self._training_running else "disabled",
                   bootstyle="primary").pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Review dataset", command=self._open_studio_review,
                   bootstyle="secondary outline").pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="My Gestures", command=self._finish_studio,
                   bootstyle="secondary outline").pack(side="left")
        if self._training_outcome is not None and self._model_current:
            self._render_evaluation(training, self._training_outcome.report)
        elif self._saved_model is not None and not self._model_current:
            ttk.Label(training, text="Saved model is stale because a gesture dataset changed. Train again to update it.",
                      bootstyle="warning", wraplength=760).grid(row=3, column=0, sticky="w", pady=(9, 0))

        live = ttk.Labelframe(self.page_body, text="Live test · fresh camera samples",
                              padding=14, bootstyle="info")
        live.grid(row=3, column=0, sticky="ew")
        live.columnconfigure(0, weight=1)
        ttk.Label(live, text="Predictions use new local camera landmarks. They are separate from the grouped holdout metrics.",
                  wraplength=760).grid(row=0, column=0, sticky="w")
        config = ttk.Labelframe(live, text="Recognition gate", padding=9, bootstyle="secondary")
        config.grid(row=1, column=0, sticky="ew", pady=(7, 0))
        config.columnconfigure(0, weight=1)
        config.columnconfigure(1, weight=0)
        ttk.Label(config, text="Gesture", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=(8, 0), pady=(0, 4))
        ttk.Label(config, text="Use in live test", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=1, sticky="w", padx=8, pady=(0, 4))
        ttk.Label(config, text="Minimum score", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=2, sticky="w", padx=(12, 8), pady=(0, 4))
        ttk.Label(config, text="Cooldown (s)", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=3, sticky="w", padx=(8, 4), pady=(0, 4))
        self._recognition_enabled_vars = {}
        self._recognition_threshold_vars = {}
        self._recognition_cooldown_vars = {}
        self._recognition_row_frames = {}
        for row_index, label in enumerate(self._saved_model.labels if self._saved_model else (), start=1):
            stored_timing = self._active_profile.get("thresholds", {}).get(label.dataset_id, {})
            enabled_default, threshold_default = self._recognition_preferences.get(
                label.dataset_id, (True, float(stored_timing.get("minimum_margin", 0.0))))
            if label.dataset_id == self._stop_gesture_id:
                enabled_default = True
            enabled_var = tk.BooleanVar(self.root, value=enabled_default)
            threshold_var = tk.StringVar(self.root, value=str(threshold_default))
            cooldown_default = self._recognition_cooldown_preferences.get(
                label.dataset_id, float(stored_timing.get("cooldown_seconds", self._recognition_cooldown_seconds)))
            cooldown_var = tk.StringVar(self.root, value=str(cooldown_default))
            self._recognition_enabled_vars[label.dataset_id] = enabled_var
            self._recognition_threshold_vars[label.dataset_id] = threshold_var
            self._recognition_cooldown_vars[label.dataset_id] = cooldown_var
            row = ttk.Frame(config, padding=(6, 5), bootstyle="light")
            row.grid(row=row_index, column=0, columnspan=3, sticky="ew", pady=2)
            row.columnconfigure(0, weight=1)
            ttk.Label(row, text=label.name, font=("Segoe UI", 10, "bold")).grid(
                row=0, column=0, sticky="w", padx=(5, 10))
            ttk.Checkbutton(row,
                            text="Enabled · required for Stop Gesture" if label.dataset_id == self._stop_gesture_id else "Enabled",
                            variable=enabled_var,
                            command=lambda key=label.dataset_id: self._update_live_gesture_setting(key),
                            bootstyle="success").grid(
                                row=0, column=1, sticky="w", padx=8)
            entry = ttk.Entry(row, textvariable=threshold_var, width=9)
            entry.grid(row=0, column=2, sticky="e", padx=(12, 8))
            entry.bind("<FocusOut>", lambda _event, key=label.dataset_id: self._update_live_gesture_setting(key))
            entry.bind("<Return>", lambda _event, key=label.dataset_id: self._update_live_gesture_setting(key))
            cooldown_entry = ttk.Entry(row, textvariable=cooldown_var, width=7)
            cooldown_entry.grid(row=0, column=3, sticky="e", padx=(8, 5))
            cooldown_entry.bind("<FocusOut>", lambda _event, key=label.dataset_id: self._update_live_gesture_setting(key))
            cooldown_entry.bind("<Return>", lambda _event, key=label.dataset_id: self._update_live_gesture_setting(key))
        explanation_row = len(self._recognition_enabled_vars) + 1
        enabled_summary = ttk.Label(config, text="Disabled gestures are excluded from live classification.",
                                    bootstyle="info", wraplength=700)
        enabled_summary.grid(row=explanation_row, column=0, columnspan=3,
                             sticky="w", pady=(7, 2))
        self._recognition_summary_label = enabled_summary
        ttk.Label(config, text="Minimum score is an uncalibrated SVC decision score, not a probability. Raise it to reject weaker matches.",
                  bootstyle="secondary", wraplength=700).grid(row=explanation_row + 1, column=0, columnspan=3,
                                                              sticky="w", pady=(2, 5))
        self._refresh_recognition_summary()
        controls_row = explanation_row + 2
        ttk.Label(config, text="Consecutive fresh frames").grid(row=controls_row, column=0, sticky="w")
        self._activation_frames_var = tk.StringVar(self.root, value=str(self._recognition_activation_frames))
        activation_entry = ttk.Entry(config, textvariable=self._activation_frames_var, width=5)
        activation_entry.grid(row=controls_row, column=1, sticky="w", padx=(8, 0))
        activation_entry.bind("<FocusOut>", self._update_recognition_timing)
        activation_entry.bind("<Return>", self._update_recognition_timing)
        self._cooldown_var = None
        live_buttons = ttk.Frame(live)
        live_buttons.grid(row=2, column=0, sticky="w", pady=(7, 5))
        ttk.Button(live_buttons, text="Start live test", command=self._start_live_test,
                   state="normal" if self._saved_model is not None and self._model_current else "disabled",
                   bootstyle="primary").pack(side="left", padx=(0, 8))
        ttk.Button(live_buttons, text="Stop runtime", command=lambda: self.request_command(RuntimeCommand.STOP),
                   bootstyle="secondary outline").pack(side="left")
        ttk.Button(live_buttons, text="Pause recognition", command=self._toggle_recognition_pause,
                   bootstyle="warning outline").pack(side="left", padx=(8, 0))
        ttk.Button(live_buttons, text="Emergency stop", command=self._emergency_stop_recognition,
                   bootstyle="danger").pack(side="left", padx=(8, 0))
        self._studio_preview_var = tk.BooleanVar(self.root, value=False)
        ttk.Checkbutton(live, text="Show live camera preview", variable=self._studio_preview_var,
                        command=self._schedule_camera_poll, bootstyle="primary").grid(
            row=3, column=0, sticky="w", pady=(2, 5))
        self._studio_preview = CameraPreview(live, bootstyle="secondary", height=180)
        self._studio_preview.grid(row=4, column=0, sticky="ew")
        self._live_status_label = ttk.Label(live, text="Live test is stopped.", wraplength=760)
        self._live_status_label.grid(row=5, column=0, sticky="w", pady=(7, 0))
        self._adaptive_status_label = ttk.Label(live, text="Adaptive scheduler: stopped · 0 FPS",
                                                bootstyle="secondary", font=("Segoe UI", 9),
                                                wraplength=760, justify="left")
        self._adaptive_status_label.grid(row=6, column=0, sticky="w", pady=(3, 0))

    @staticmethod
    def _training_readiness(datasets: tuple[GestureDataset, ...]) -> tuple[bool, str]:
        if len(datasets) < 2:
            return False, "Save at least two custom gestures to train a multi-class model."
        counts = [len(item.examples) for item in datasets]
        if any(count < 3 for count in counts):
            return False, "Every gesture needs at least 3 valid examples."
        if len({item.hand_count for item in datasets}) != 1:
            return False, "All gestures in one model must use the same hand configuration."
        try:
            for dataset in datasets:
                from gesture_manager.gesture_data import validate_dataset
                validate_dataset(dataset)
        except ValueError as exc:
            return False, f"A saved dataset is invalid: {exc}"
        return True, f"{len(datasets)} custom gestures are ready. At least 5 independent examples per gesture are recommended for useful validation."

    def _render_evaluation(self, parent: ttk.Labelframe, report: EvaluationReport) -> None:
        row = 3
        heading = "Grouped holdout evaluation" if report.evaluated else "Validation unavailable"
        ttk.Label(parent, text=heading, font=("Segoe UI", 11, "bold")).grid(
            row=row, column=0, sticky="w", pady=(12, 4))
        row += 1
        if report.evaluated:
            metric_text = f"Validation accuracy {report.accuracy:.1%} · macro F1 {report.macro_f1:.3f} · " + (
                "holdout sample thresholds met" if report.reliable else "PRELIMINARY · estimate is unreliable")
            ttk.Label(parent, text=metric_text,
                      bootstyle="success" if report.reliable else "warning").grid(
                row=row, column=0, sticky="w")
            row += 1
            table = ttk.Treeview(parent, columns=("support", "correct", "precision", "recall", "f1"),
                                 show="tree headings", height=min(6, max(2, len(report.classes))))
            table.heading("#0", text="Custom gesture")
            table.column("#0", width=180, anchor="w")
            for column, title in (("support", "Validation samples"), ("correct", "Correct"),
                                  ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
                table.heading(column, text=title)
                table.column(column, width=115, anchor="center")
            table.grid(row=row, column=0, sticky="ew", pady=(6, 0))
            for result in report.classes:
                table.insert("", "end", text=result.label.name, values=(
                    result.support, result.correct, f"{result.precision:.3f}",
                    f"{result.recall:.3f}", f"{result.f1:.3f}"))
            row += 1
            if report.confusion is not None:
                matrix_text = "Confusion matrix · rows are actual, columns are predicted\n" + "Actual ↓ / Predicted →  " + " | ".join(report.confusion_labels)
                for name, values in zip(report.confusion_labels, report.confusion, strict=True):
                    matrix_text += f"\n{name}: " + " | ".join(str(value) for value in values)
                ttk.Label(parent, text=matrix_text, justify="left", wraplength=760,
                          font=("Consolas", 9)).grid(row=row, column=0, sticky="w", pady=(7, 0))
                row += 1
        for warning in report.warnings:
            ttk.Label(parent, text=f"• {warning}", bootstyle="warning",
                      wraplength=760).grid(row=row, column=0, sticky="w", pady=(3, 0))
            row += 1

    def _start_training(self) -> None:
        if self._training_running:
            return
        if self.controller.state in (RuntimeState.ACTIVE, RuntimeState.PAUSED):
            self.controller.stop()
        else:
            self._stop_live_test()
        self._training_running = True
        self._training_message = "Training and grouped evaluation are running locally…"
        if self.current_page == "Gesture Studio" and self._studio_stage == "saved":
            self.show_page("Gesture Studio")

        def train() -> None:
            try:
                outcome = self.trainer.train(self.gesture_store.list())
            except Exception as exc:
                self._service_events.put(("training_error", str(exc)))
            else:
                self._service_events.put(("training_done", outcome))

        self._training_thread = Thread(target=train, name="GestureManagerTraining", daemon=False)
        self._training_thread.start()
        self._schedule_service_poll()

    def start_runtime(self) -> None:
        """Start camera inference, custom classification, and mapped actions."""
        self._start_recognition_runtime()

    def _start_live_test(self) -> None:
        """Button callback for the Studio live test, sharing the runtime path."""
        try:
            self._start_recognition_runtime()
        except (RuntimeError, ValueError) as exc:
            if self._live_status_label is not None and self._live_status_label.winfo_exists():
                self._live_status_label.configure(text=f"Could not start recognition: {exc}")

    def _start_recognition_runtime(self) -> None:
        model = self._saved_model
        if model is None or not self._model_current:
            if self._training_running or any(thread.is_alive() for thread in self._model_load_threads):
                raise RuntimeError("The active profile model is still loading. Try Start again in a moment.")
            raise RuntimeError("The active profile needs a current trained model. Train its custom gestures first.")
        if self._live_classifier is not None and self._live_classifier.is_alive:
            if self._live_test_active:
                return
            raise RuntimeError("The previous recognition worker is stopping. Try Start again shortly.")
        model_ids = {label.dataset_id for label in model.labels}
        control_ids = {key for key in (self._stop_gesture_id, self._pause_gesture_id,
                                       self._resume_gesture_id) if key is not None}
        missing_controls = control_ids - model_ids
        if missing_controls:
            raise RuntimeError("A configured Pause, Resume, or Stop Gesture is missing from this profile's current model. Update the control gesture settings before starting.")
        try:
            enabled = {key: variable.get() for key, variable in self._recognition_enabled_vars.items()
                       if key in model_ids}
            if self._stop_gesture_id is not None:
                enabled[self._stop_gesture_id] = True
            for control_id in (self._pause_gesture_id, self._resume_gesture_id):
                if control_id is not None:
                    enabled[control_id] = True
            thresholds = {key: float(variable.get())
                          for key, variable in self._recognition_threshold_vars.items()
                          if key in model_ids}
            activation_frames = int(self._activation_frames_var.get()) if self._activation_frames_var else 3
            cooldown = float(self._cooldown_var.get()) if self._cooldown_var else 1.0
            if activation_frames < 1 or not math.isfinite(cooldown) or not 0 <= cooldown <= 60:
                raise ValueError("Use at least 1 consecutive frame and a cooldown from 0 to 60 seconds.")
            if any(not math.isfinite(value) or not 0 <= value <= 1_000_000 for value in thresholds.values()):
                raise ValueError("SVC margin thresholds must be finite and non-negative.")
            cooldowns = {key: float(variable.get())
                         for key, variable in self._recognition_cooldown_vars.items()
                         if key in model_ids}
            if any(not math.isfinite(value) or not 0 <= value <= 3600 for value in cooldowns.values()):
                raise ValueError("Use per-gesture cooldowns from 0 to 3600 seconds.")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Recognition settings are invalid: {exc}") from exc
        stored_thresholds = self._active_profile.get("thresholds", {})
        for label in model.labels:
            stored = stored_thresholds.get(label.dataset_id, {})
            enabled.setdefault(label.dataset_id, self._recognition_preferences.get(label.dataset_id, (True, 0.0))[0])
            thresholds.setdefault(label.dataset_id, float(stored.get("minimum_margin", 0.0)))
            cooldowns.setdefault(label.dataset_id, float(stored.get("cooldown_seconds", cooldown)))
        self._recognition_activation_frames = activation_frames
        self._recognition_cooldown_seconds = cooldown
        self._recognition_preferences = {
            key: (enabled.get(key, True), thresholds.get(key, 0.0))
            for key in {label.dataset_id for label in model.labels}
        }
        self._recognition_cooldown_preferences.update(cooldowns)
        try:
            # Restore the profile's saved choice only after explicit consent.
            # A stop/failure clears this session's consent, so restarting the
            # service asks again when this profile prefers real input.
            if not self._real_input_confirmed:
                self.action_dispatcher.set_dry_run(True)
                self._dry_run_var.set(True)
            index = int(self._camera_index_var.get().strip()) if self._camera_index_var is not None else self._camera_index
        except ValueError:
            index = self._camera_index
        self._camera_index = index
        state = self.hand_engine.status.state
        if state not in (EngineState.RUNNING, EngineState.STARTING, EngineState.PAUSED):
            started = self.hand_engine.start(EngineConfig(
                camera=CameraConfig(index=index, width=640, height=480),
                processing_rate_hz=AdaptiveScheduler.ACTIVE_FOCUSED_FPS,
                model_dir=model_assets_directory()))
            if not started:
                raise RuntimeError("The camera or landmark engine is still stopping. Try Start again shortly.")
        elif state is EngineState.PAUSED:
            self.hand_engine.resume()
        if not self._real_input_confirmed and not self._profile_mode_preference:
            # The camera is ready, but recognition/action dispatch has not
            # started yet, so consent is requested before any event can fire.
            self._confirm_real_input()
        self._live_classifier = LiveRecognitionWorker(
            model, lambda: self.hand_engine.status, self._live_results,
            enabled=enabled, minimum_margins=thresholds,
            activation_frames=activation_frames, cooldown_seconds=cooldown,
            cooldowns=cooldowns,
            release_frames=2)
        self._live_classifier.start()
        self._recognition_paused = False
        self._gesture_controls.configure(
            pause_gesture_id=self._pause_gesture_id,
            resume_gesture_id=self._resume_gesture_id,
            stop_gesture_id=self._stop_gesture_id,
        )
        self._gesture_actions_paused = False
        self._gesture_actions_paused_since = None
        self._recognition_emergency_stopped = False
        self._live_test_active = True
        self._inactivity_tracker.reset()
        self._inactivity_last_frame_sequence = None
        self._stop_gesture_trigger = StopGestureTrigger()
        self._dry_run_var.set(True)
        if self._live_status_label is not None and self._live_status_label.winfo_exists():
            self._live_status_label.configure(
                text="Runtime started. Enabled gesture mappings run in dry-run mode unless real input was confirmed.")
        self._schedule_camera_poll()
        self._schedule_service_poll()
        self._apply_adaptive_schedule()

    def pause_runtime(self) -> None:
        worker = self._live_classifier
        if worker is None or not worker.is_alive:
            return
        worker.pause()
        self.hand_engine.pause()
        self._recognition_paused = True
        self._strict_runtime_pause = True
        self._clear_live_results()
        # Runtime pause is strict: release the camera so tray pause always has
        # the same predictable privacy and resource behavior.
        if self._wake_worker is not None:
            self._wake_worker.stop()
            self._wake_worker = None
        if self._live_status_label is not None and self._live_status_label.winfo_exists():
            self._live_status_label.configure(text="Runtime paused; camera release requested.")
        self._apply_adaptive_schedule()

    def resume_runtime(self) -> None:
        worker = self._live_classifier
        if worker is None or not worker.is_alive:
            raise RuntimeError("Recognition worker is not available. Stop and start the runtime again.")
        if self.hand_engine.status.state is EngineState.PAUSED:
            self.hand_engine.resume()
        worker.resume()
        self._recognition_paused = False
        self._strict_runtime_pause = False
        self._inactivity_tracker.reset()
        self._inactivity_last_frame_sequence = None
        self._clear_live_results()
        if self._live_status_label is not None and self._live_status_label.winfo_exists():
            self._live_status_label.configure(text="Runtime resumed; hold a pose after a neutral release.")
        self._schedule_service_poll()
        self._apply_adaptive_schedule()

    def stop_runtime(self) -> None:
        self._stop_live_test()

    def _update_live_gesture_setting(self, gesture_id: str) -> None:
        enabled_var = self._recognition_enabled_vars.get(gesture_id)
        threshold_var = self._recognition_threshold_vars.get(gesture_id)
        required_control_ids = {item for item in (self._stop_gesture_id,
                                                    self._pause_gesture_id,
                                                    self._resume_gesture_id)
                                if item is not None}
        if gesture_id in required_control_ids and enabled_var is not None and not enabled_var.get():
            enabled_var.set(True)
            if self._live_status_label is not None and self._live_status_label.winfo_exists():
                self._live_status_label.configure(
                    text="Configured control gestures must stay enabled for safe recognition.")
            return
        try:
            threshold = float(threshold_var.get()) if threshold_var is not None else 0.0
            if not math.isfinite(threshold) or not 0 <= threshold <= 1_000_000:
                raise ValueError("Margin threshold must be finite and non-negative")
            enabled = enabled_var.get() if enabled_var is not None else True
            cooldown_var = self._recognition_cooldown_vars.get(gesture_id)
            cooldown = float(cooldown_var.get()) if cooldown_var is not None else 1.0
            if not math.isfinite(cooldown) or not 0 <= cooldown <= 3600:
                raise ValueError("Cooldown must be from 0 to 3600 seconds")
            self._recognition_preferences[gesture_id] = (enabled, threshold)
            self._recognition_cooldown_preferences[gesture_id] = cooldown
            self._clear_live_results()
            worker = self._live_classifier
            if worker is not None and worker.is_alive:
                worker.gate.update_gesture(gesture_id, enabled=enabled,
                                           minimum_margin=threshold,
                                           cooldown_seconds=cooldown)
            self._sync_active_profile()
            self._refresh_recognition_summary()
        except (ValueError, TypeError) as exc:
            if self._live_status_label is not None:
                self._live_status_label.configure(text=f"Gesture setting was not applied: {exc}")

    def _refresh_recognition_summary(self) -> None:
        label = getattr(self, "_recognition_summary_label", None)
        if label is None or not label.winfo_exists():
            return
        active = sum(variable.get() for variable in self._recognition_enabled_vars.values())
        total = len(self._recognition_enabled_vars)
        label.configure(
            text=(f"{active} of {total} gestures enabled. Disabled gestures are excluded from live classification."
                  if active >= 2 else
                  f"{active} of {total} gestures enabled. Enable at least two for live classification."),
            bootstyle="info" if active >= 2 else "warning",
        )

    def _update_recognition_timing(self, _event: object = None) -> None:
        try:
            frames = int(self._activation_frames_var.get()) if self._activation_frames_var else 3
            if frames < 1:
                raise ValueError("Use at least 1 consecutive frame")
            self._recognition_activation_frames = frames
            if self._live_test_active and self._live_status_label is not None:
                self._live_status_label.configure(text="Activation frame changes apply to the next live test.")
            self._sync_active_profile()
        except (ValueError, TypeError) as exc:
            if self._live_status_label is not None:
                self._live_status_label.configure(text=f"Recognition timing setting was not applied: {exc}")

    def _toggle_recognition_pause(self) -> None:
        worker = self._live_classifier
        if worker is None or not worker.is_alive:
            if self._live_status_label is not None:
                self._live_status_label.configure(text="Start the live test before pausing recognition.")
            return
        if self._recognition_paused:
            self._request_wake_resume("Recognition resumed; release all hands to re-arm the neutral gate.")
            text = "Stopping optional wake camera before resuming recognition…"
        else:
            worker.pause()
            self.hand_engine.pause()
            self._recognition_paused = True
            self._clear_live_results()
            if self._wake_detection_var.get():
                self._start_wake_when_camera_released()
                text = "Recognition paused; optional 2.5 FPS motion wake detection is starting."
            else:
                text = "Strict pause requested; the camera is being released."
        if self._live_status_label is not None:
            self._live_status_label.configure(text=text)
        self._apply_adaptive_schedule()

    def _emergency_stop_recognition(self) -> None:
        worker = self._live_classifier
        if worker is not None:
            worker.emergency_stop()
        self.hand_engine.stop()
        if self._wake_worker is not None:
            self._wake_worker.stop()
        self._recognition_paused = False
        self._recognition_emergency_stopped = True
        self._live_test_active = False
        self._adaptive_mode = "stopped"
        self._clear_live_results()
        self._refresh_execution_mode_label()
        if self._studio_preview is not None:
            self._studio_preview.clear("Emergency stopped")
        if self._live_status_label is not None:
            self._live_status_label.configure(
                text="EMERGENCY STOP · recognition is stopped and the camera is being released. Start a new live test to resume.")
        self._apply_adaptive_schedule()
        if self.controller.state in (RuntimeState.ACTIVE, RuntimeState.PAUSED):
            self.controller.stop()

    def _stop_live_test(self) -> None:
        # Requiring confirmation again after a stop or failure prevents a
        # later manual restart from silently restoring real keyboard input.
        self.action_dispatcher.set_dry_run(True)
        self._real_input_confirmed = False
        self._dry_run_var.set(True)
        if self._live_classifier is not None:
            self._live_classifier.stop()
        self.hand_engine.stop()
        if self._wake_worker is not None:
            self._wake_worker.stop()
        self._recognition_paused = False
        self._strict_runtime_pause = False
        self._live_test_active = False
        self._gesture_actions_paused = False
        self._gesture_actions_paused_since = None
        self._gesture_controls.reset()
        self._adaptive_mode = "stopped"
        self._clear_live_results()
        if self._live_status_label is not None and self._live_status_label.winfo_exists():
            self._live_status_label.configure(text="Live test stopped; camera release requested.")
        self._refresh_gesture_control_state()
        self._apply_adaptive_schedule()

    def _schedule_adaptive_poll(self) -> None:
        if self._live_test_active and not self._adaptive_poll_scheduled:
            self._adaptive_poll_scheduled = True
            delay = 5000 if self._recognition_paused else 1000
            self.root.after(delay, self._poll_adaptive_schedule)

    def _poll_adaptive_schedule(self) -> None:
        self._adaptive_poll_scheduled = False
        if not self._live_test_active:
            return
        if self._check_runtime_inactivity():
            return
        self._apply_adaptive_schedule()
        self._schedule_adaptive_poll()

    def _check_runtime_inactivity(self) -> bool:
        """Strictly pause after five minutes without meaningful camera motion."""
        if (self.controller.state is not RuntimeState.ACTIVE
                or not self._live_test_active or self._recognition_paused):
            return False
        newest = self.camera_manager.get_latest_frame()
        if newest is None:
            return False
        sequence, frame = newest
        if sequence == self._inactivity_last_frame_sequence:
            return False
        self._inactivity_last_frame_sequence = sequence
        try:
            idle = self._inactivity_tracker.update(frame, now=monotonic())
        except (TypeError, ValueError) as exc:
            logger.warning("Could not inspect camera movement for automatic pause: %s", exc)
            self._inactivity_tracker.reset()
            return False
        if not idle:
            return False
        self._last_runtime_action = (
            "Automatically paused after 5 minutes without camera movement; "
            "the camera was released to reduce resource use.")
        self.controller.pause()
        return True

    def _apply_adaptive_schedule(self) -> None:
        if not self._live_test_active or self._recognition_emergency_stopped:
            schedule = AdaptiveScheduler.schedule(RuntimeState.STOPPED, focused=False)
        elif self._recognition_paused:
            schedule = AdaptiveScheduler.schedule(RuntimeState.PAUSED, focused=False,
                                                  wake_detection=(self._wake_detection_var.get()
                                                                  and not self._strict_runtime_pause),
                                                  strict_pause=(self._strict_runtime_pause
                                                                or not self._wake_detection_var.get()))
        else:
            try:
                hwnd = int(self.root.winfo_id())
            except tk.TclError:
                hwnd = 0
            focused = self._focus_detector.is_focused(hwnd)
            # Unknown platform state uses the lower-power cadence.
            schedule = AdaptiveScheduler.schedule(
                RuntimeState.ACTIVE, focused=focused is True,
                shortcut_paused=self._gesture_actions_paused)
            self.hand_engine.set_processing_rate(schedule.target_fps)
        self._adaptive_mode = schedule.mode
        if self._adaptive_status_label is not None and self._adaptive_status_label.winfo_exists():
            target = f"{schedule.target_fps:g} FPS"
            now = monotonic()
            engine_status = self.hand_engine.status
            elapsed = max(0.001, now - self._fps_sample_time)
            self._observed_fps = max(0, engine_status.processed_frames - self._fps_sample_frames) / elapsed
            self._fps_sample_time = now
            self._fps_sample_frames = engine_status.processed_frames
            metrics = self._performance_sampler.sample()
            cpu = (f"{metrics.cpu_percent:.0f}% process CPU (one core)"
                   if metrics.cpu_percent is not None else "process CPU n/a")
            memory = f"{metrics.working_set_mb:.0f} MB" if metrics.working_set_mb is not None else "memory n/a"
            latency = (f"{engine_status.inference_latency_ms:.0f} ms inference"
                       if engine_status.inference_latency_ms is not None else "inference n/a")
            self._adaptive_status_label.configure(
                text=(f"Adaptive: {schedule.mode.replace('_', ' ')} · target {target}, "
                      f"observed {self._observed_fps:.1f} FPS · {latency} · {cpu} · {memory}"))
        if schedule.release_camera and self._live_test_active and not self._recognition_paused:
            self.hand_engine.stop()
        self._schedule_adaptive_poll()

    def _start_wake_when_camera_released(self) -> None:
        if not self._live_test_active or not self._recognition_paused or not self._wake_detection_var.get():
            return
        if self.camera_manager.status.state is not CameraState.STOPPED:
            self.root.after(75, self._start_wake_when_camera_released)
            return
        try:
            index = int(self._camera_index_var.get().strip()) if self._camera_index_var else 0
        except ValueError:
            index = 0
        self._wake_worker = WakeMotionWorker(
            self.camera_manager, index,
            on_motion=lambda: self._service_events.put(("wake_motion",)),
        )
        self._wake_worker.start()
        self.root.after(250, self._poll_wake_status)

    def _update_wake_detection_setting(self) -> None:
        if not self._live_test_active or not self._recognition_paused:
            return
        if self._wake_detection_var.get():
            self._start_wake_when_camera_released()
        elif self._wake_worker is not None:
            self._wake_worker.stop()
            self.root.after(50, self._finish_strict_pause)
        self._apply_adaptive_schedule()

    def _finish_strict_pause(self) -> None:
        if not self._recognition_paused or self._wake_detection_var.get():
            return
        if self.camera_manager.status.state is not CameraState.STOPPED:
            self.root.after(50, self._finish_strict_pause)
            return
        self._wake_worker = None
        if self._live_status_label is not None and self._live_status_label.winfo_exists():
            self._live_status_label.configure(text="Strict pause active; camera released and recognition idle.")
        self._apply_adaptive_schedule()

    def _poll_wake_status(self) -> None:
        worker = self._wake_worker
        if worker is None or not self._recognition_paused:
            return
        if worker.error and self._live_status_label is not None and self._live_status_label.winfo_exists():
            self._live_status_label.configure(text=f"Wake detection unavailable: {worker.error}")
            return
        if worker.is_alive:
            self.root.after(500, self._poll_wake_status)

    def _request_wake_resume(self, status_text: str) -> None:
        self._wake_resume_pending = True
        self._wake_resume_text = status_text
        if self._wake_worker is not None:
            self._wake_worker.stop()
        else:
            self.camera_manager.stop()
        self.root.after(50, self._finish_wake_resume)

    def _finish_wake_resume(self) -> None:
        if not self._wake_resume_pending:
            return
        if self.camera_manager.status.state is not CameraState.STOPPED:
            self.root.after(50, self._finish_wake_resume)
            return
        worker = self._live_classifier
        if not self._live_test_active or worker is None or not worker.is_alive:
            self._wake_resume_pending = False
            return
        self.hand_engine.resume()
        worker.resume()
        self._recognition_paused = False
        self._wake_resume_pending = False
        self._wake_worker = None
        self._clear_live_results()
        if self._live_status_label is not None and self._live_status_label.winfo_exists():
            self._live_status_label.configure(text=self._wake_resume_text)
        self._apply_adaptive_schedule()

    def _refresh_live_status(self) -> None:
        label = self._live_status_label
        if label is None or not label.winfo_exists():
            return
        status = self.hand_engine.status
        if status.state is EngineState.ERROR:
            label.configure(text=status.error or "Camera or landmark engine failed.")
        elif status.state in (EngineState.STARTING, EngineState.STOPPING):
            label.configure(text="Starting local camera/model…" if status.state is EngineState.STARTING else "Stopping camera…")

    def _load_saved_model_async(self) -> None:
        self._profile_generation += 1
        generation = self._profile_generation
        trainer = self.trainer
        gesture_store = self.gesture_store
        def load() -> None:
            try:
                model = trainer.load_saved()
                datasets = gesture_store.list() if model is not None else ()
                current = model is not None and model_matches_datasets(model, datasets)
                self._service_events.put(("model_loaded", generation, model, current, None))
            except Exception as exc:
                self._service_events.put(("model_loaded", generation, None, False, str(exc)))
        self._model_load_thread = Thread(target=load, name="GestureManagerModelLoad", daemon=False)
        self._model_load_threads = [thread for thread in self._model_load_threads if thread.is_alive()]
        self._model_load_threads.append(self._model_load_thread)
        self._model_load_thread.start()

    def _schedule_service_poll(self) -> None:
        if not self._service_poll_scheduled:
            self._service_poll_scheduled = True
            delay = 1000 if self._recognition_paused else 150
            self.root.after(delay, self._poll_service_events)

    def _poll_service_events(self) -> None:
        self._service_poll_scheduled = False
        changed = False
        engine_status = self.hand_engine.status
        if self._live_test_active and engine_status.state is EngineState.ERROR:
            self._stop_live_test()
            if self.controller.state in (RuntimeState.ACTIVE, RuntimeState.PAUSED):
                self.controller.report_error(engine_status.error or "Camera or landmark engine failed.")
            changed = True
        if self.current_page == "Dashboard":
            detail_label = getattr(self, "_runtime_overview_detail", None)
            if detail_label is not None and detail_label.winfo_exists():
                if self.controller.error:
                    detail = self.controller.error
                elif self.controller.state is RuntimeState.ACTIVE:
                    activity = ("Ordinary shortcuts are paused. " if self._gesture_actions_paused
                                else "Eligible shortcut mappings may run. ")
                    mode = ("Configured actions: dry run." if self.action_dispatcher.dry_run
                            else "Real keyboard input is enabled.")
                    detail = "Camera and local gesture recognition are running. " + activity + mode
                    if engine_status.state is EngineState.STARTING:
                        detail = "Starting the local camera and hand landmark model…"
                    elif engine_status.state is EngineState.ERROR:
                        detail = engine_status.error or "Camera or landmark model failed."
                elif self.controller.state is RuntimeState.PAUSED:
                    detail = "Runtime paused; the camera is being released."
                elif self.controller.state is RuntimeState.STOPPING:
                    detail = "Stopping recognition and releasing the camera."
                elif self.controller.state is RuntimeState.STOPPED:
                    detail = "Runtime stopped. No camera is active."
                else:
                    detail = "Starting the local runtime."
                if detail_label.cget("text") != detail:
                    detail_label.configure(text=detail)
            activity_label = getattr(self, "_dashboard_activity", None)
            if activity_label is not None and activity_label.winfo_exists():
                if activity_label.cget("text") != self._last_runtime_action:
                    activity_label.configure(text=self._last_runtime_action)
        while True:
            try:
                event = self._service_events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "wake_motion":
                self._request_wake_resume("Motion detected; recognition resumed. Release all hands to re-arm the neutral gate.")
            elif kind == "profile_export_done":
                if self._profile_status is not None and self._profile_status.winfo_exists():
                    self._profile_status.configure(text=f"Profile exported: {event[1]}", bootstyle="success")
            elif kind == "profile_import_done":
                self.root.after(50, lambda selected=event[1]: self._activate_imported_profile(selected))
            elif kind == "profile_validate_done":
                if self._profile_status is not None and self._profile_status.winfo_exists():
                    self._profile_status.configure(text=str(event[1]),
                                                   bootstyle="success" if event[2] else "danger")
            elif kind == "profile_io_error":
                if self._profile_status is not None and self._profile_status.winfo_exists():
                    self._profile_status.configure(text=str(event[1]), bootstyle="danger")
            elif kind == "model_loaded":
                if self._training_running or self._training_outcome is not None:
                    continue
                _, generation, model, current, error = event
                if generation != self._profile_generation:
                    continue
                self._saved_model = model
                self._model_current = bool(current)
                if error:
                    self._training_message = f"Saved model could not be loaded safely: {error}"
                elif model is None:
                    self._training_message = "No saved model yet. Train after creating at least two custom gestures."
                elif current:
                    self._training_message = "Saved model loaded without retraining."
                else:
                    self._training_message = "Saved model is stale because local examples changed. Retrain to update it."
                changed = True
            elif kind == "training_done":
                self._training_outcome = event[1]  # type: ignore[assignment]
                self._saved_model = self._training_outcome.model
                self._model_current = True
                self._training_running = False
                self._training_message = "Training and evaluation completed locally."
                self._sync_active_profile(sync_assets=True)
                changed = True
            elif kind == "training_error":
                self._training_running = False
                self._training_message = f"Training failed: {event[1]}"
                changed = True
            elif kind == "dynamic_training_done":
                model = event[1]
                self._dynamic_training_running = False
                self._dynamic_saved_model = model
                if self._dynamic_report_label is not None and self._dynamic_report_label.winfo_exists():
                    self._dynamic_report_label.configure(
                        text=self._format_dynamic_evaluation(model),
                        bootstyle="success" if model.evaluation.reliable else "warning")
                changed = True
            elif kind == "dynamic_training_error":
                self._dynamic_training_running = False
                if self._dynamic_report_label is not None and self._dynamic_report_label.winfo_exists():
                    self._dynamic_report_label.configure(text=f"Training/evaluation failed: {event[1]}",
                                                         bootstyle="danger")
                changed = True
        while True:
            try:
                event = self._live_results.get_nowait()
            except queue.Empty:
                break
            if (not self._live_test_active or self._recognition_paused
                    or self._recognition_emergency_stopped):
                continue
            score = (f" · raw SVC margin {event.decision_score:.4f} (uncalibrated)"
                     if event.decision_score is not None else "")
            control_action = self._gesture_controls.route(event)
            is_pause_resume = event.gesture_id in {
                item for item in (self._pause_gesture_id, self._resume_gesture_id)
                if item is not None
            }
            if is_pause_resume:
                if control_action in ("pause", "resume"):
                    self._gesture_actions_paused = self._gesture_controls.actions_paused
                    self._gesture_actions_paused_since = (
                        monotonic() if self._gesture_actions_paused else None)
                    self._last_runtime_action = (
                        "Pause Gesture recognized · ordinary shortcuts blocked; recognition remains active."
                        if control_action == "pause" else
                        "Resume Gesture recognized · ordinary shortcut eligibility restored."
                    )
                else:
                    self._last_runtime_action = (
                        "Pause/Resume Gesture candidate ignored · "
                        f"{self._gesture_controls.last_rejection_reason or 'already in that state'}; no action dispatched."
                    )
                self._refresh_gesture_control_state()
                if self._live_status_label is not None and self._live_status_label.winfo_exists():
                    self._live_status_label.configure(
                        text=self._last_runtime_action,
                        bootstyle="warning" if self._gesture_actions_paused else "info")
                continue
            is_configured_stop = (self._stop_gesture_id is not None
                                  and event.gesture_id == self._stop_gesture_id)
            if is_configured_stop:
                # Stop control is independent of shortcut mappings and gets
                # first refusal. Even rejected/stale Stop Gesture events are
                # consumed so they cannot fall through to their normal mapping.
                if not event.trigger_eligible:
                    self._last_runtime_action = describe_stop_candidate(
                        event, self._recognition_activation_frames)
                    if self._live_status_label is not None and self._live_status_label.winfo_exists():
                        pending = event.reason == "activation_pending"
                        self._live_status_label.configure(
                            text=self._last_runtime_action,
                            bootstyle="info" if pending else "warning")
                elif (control_action == "stop"
                      and self._stop_gesture_trigger.consume(event, self._stop_gesture_id)):
                    self._last_runtime_action = "Stop Gesture recognized · service stopped; manual restart required."
                    self._clear_live_results()
                    if self.controller.state in (RuntimeState.ACTIVE, RuntimeState.PAUSED):
                        self.controller.stop()
                    else:
                        self._stop_live_test()
                    if self._live_status_label is not None and self._live_status_label.winfo_exists():
                        self._live_status_label.configure(
                            text="Stop Gesture recognized · service stopped; camera release requested. Manually start the service to resume.",
                            bootstyle="warning")
                    self._refresh_stop_gesture_status()
                else:
                    self._last_runtime_action = (
                        "Stop Gesture candidate rejected · "
                        f"{self._stop_gesture_trigger.last_rejection_reason or 'event not accepted'}; no action dispatched.")
                    if self._live_status_label is not None and self._live_status_label.winfo_exists():
                        self._live_status_label.configure(text=self._last_runtime_action,
                                                          bootstyle="warning")
                continue
            if event.trigger_eligible and self._gesture_actions_paused:
                action_text = "Blocked · shortcut execution is paused"
                self._last_runtime_action = action_text
            elif event.trigger_eligible:
                # Dispatch is deliberately independent of the current page or
                # whether Gesture Studio's diagnostic label exists.
                result = self._gesture_controls.dispatch_ordinary(
                    event, self.action_dispatcher, self._action_mappings)
                action_text = result.message if result is not None else "Blocked · control gesture"
                self._last_runtime_action = action_text
            else:
                action_text = f"Not eligible · {event.reason.replace('_', ' ')}"
            if self._live_status_label is not None and self._live_status_label.winfo_exists():
                self._live_status_label.configure(
                    text=f"{event.prediction} · {action_text}{score}")
        if changed and (self.current_page == "Dashboard"
                        or (self.current_page == "Gesture Studio" and self._studio_stage == "saved")
                        or self.current_page == "Actions"):
            self.show_page(self.current_page)
        active = (self._training_running or
                  self._dynamic_training_running or
                  any(thread.is_alive() for thread in self._model_load_threads) or
                  (self._profile_io_thread is not None and self._profile_io_thread.is_alive()) or
                  (self._live_classifier is not None and self._live_classifier.is_alive))
        if active:
            self._schedule_service_poll()

    def _clear_live_results(self) -> None:
        while True:
            try:
                self._live_results.get_nowait()
            except queue.Empty:
                return

    def stop_background_tasks(self) -> None:
        """Request worker shutdown before the Tk root is destroyed."""
        self.action_dispatcher.emergency_stop()
        self._stop_shortcut_listener()
        self._stop_live_test()
        if self._wake_worker is not None:
            self._wake_worker.stop()

    def background_tasks_stopped(self) -> bool:
        listener_stopped = (self._action_record_listener is None
                            or not self._action_record_listener.is_alive())
        wake_stopped = self._wake_worker is None or self._wake_worker.wait(0)
        return listener_stopped and wake_stopped and not any(thread is not None and thread.is_alive()
                       for thread in (*self._model_load_threads, self._training_thread,
                                      self._profile_io_thread, self._dynamic_training_thread)) and (
            self._live_classifier is None or self._live_classifier.wait(0))

    def _new_gesture(self) -> None:
        self._studio_session = None
        self._studio_stage = "details"
        self.show_page("Gesture Studio")

    def _manage_gesture(self, dataset: GestureDataset) -> None:
        self._studio_session = GestureRecordingSession(dataset)
        self._studio_stage = "review"
        self.show_page("Gesture Studio")

    def _begin_gesture(self) -> None:
        if (self._studio_type_var is not None and
                self._studio_type_var.get() == "Dynamic hand sequence (experimental)"):
            try:
                selected = self._dynamic_label_var.get().strip() if self._dynamic_label_var else ""
                if selected:
                    dataset = next(item for item in self.dynamic_store.list()
                                   if item.name == selected)
                else:
                    dataset = create_dynamic_draft(self._studio_name_var.get())
                    if any(item.name.casefold() == dataset.name.casefold()
                           for item in self.dynamic_store.list()):
                        raise ValueError("A dynamic gesture with this name already exists; select it to add examples")
                self._dynamic_session = DynamicRecordingSession(dataset)
                self._dynamic_recorder = DynamicStudioRecorder(
                    self._dynamic_session, self.hand_engine.status.processed_frames)
                self._studio_session = None
                self._studio_stage = "dynamic_record"
                self.show_page("Gesture Studio")
            except (ValueError, OSError, StopIteration) as exc:
                if self._studio_error_var is not None:
                    self._studio_error_var.set(str(exc) or "Selected dynamic label is unavailable")
            return
        try:
            if self._studio_type_var is None or self._studio_type_var.get() != "Static hand pose":
                raise ValueError("Choose the supported static hand pose type")
            dataset = create_draft(self._studio_name_var.get(), int(self._studio_hand_count_var.get()))
            if any(item.name.casefold() == dataset.name.casefold() for item in self.gesture_store.list()):
                raise ValueError("A gesture with this name already exists")
        except (ValueError, OSError) as exc:
            self._studio_error_var.set(str(exc))
            return
        self._studio_session = GestureRecordingSession(dataset)
        self._studio_stage = "record"
        self._last_recorded_processed_frames = self.hand_engine.status.processed_frames
        self.show_page("Gesture Studio")

    def _start_dynamic_take(self) -> None:
        if self._dynamic_recorder is None:
            return
        if self.hand_engine.status.state is not EngineState.RUNNING:
            if self._dynamic_status_label is not None:
                self._dynamic_status_label.configure(text="Start the local camera and wait until detection is running.")
            return
        try:
            self._dynamic_recorder.begin()
            self._dynamic_recorder.last_sequence = self.hand_engine.status.processed_frames
            if self._dynamic_status_label is not None:
                self._dynamic_status_label.configure(text="Recording active · perform one complete movement with one hand visible.")
            if self._recording_progress_label is not None:
                self._recording_progress_label.configure(text="Recording 0/60 fresh landmark frames · stop at 8 or more")
        except (RuntimeError, ValueError) as exc:
            if self._dynamic_status_label is not None:
                self._dynamic_status_label.configure(text=str(exc))

    def _finish_dynamic_take(self) -> None:
        recorder = self._dynamic_recorder
        if recorder is None or not recorder.recording:
            if self._dynamic_status_label is not None:
                self._dynamic_status_label.configure(text="No take is currently recording.")
            return
        try:
            recorder.finish()
            self.dynamic_store.save(recorder.session.dataset)
            self._dynamic_session = recorder.session
            self._dynamic_status_label.configure(text="Valid numeric sequence saved locally; no image or video was stored.")
            self._refresh_dynamic_recording_progress()
        except (ValueError, OSError, RuntimeError) as exc:
            if self._dynamic_status_label is not None:
                self._dynamic_status_label.configure(text=f"Take not saved: {exc}")

    def _cancel_dynamic_take(self) -> None:
        if self._dynamic_recorder is not None:
            self._dynamic_recorder.cancel()
        if self._dynamic_status_label is not None:
            self._dynamic_status_label.configure(text="Current take cancelled; saved sequences were preserved.")
        self._refresh_dynamic_recording_progress()

    def _refresh_dynamic_recording_progress(self) -> None:
        if self._recording_progress_label is None or self._dynamic_recorder is None:
            return
        self._recording_progress_label.configure(
            text=f"{len(self._dynamic_recorder.session.dataset.examples)} saved sequences · take "
                 f"{self._dynamic_recorder.frame_count}/60 frames")

    def _open_dynamic_review(self) -> None:
        if self._dynamic_recorder is not None and self._dynamic_recorder.recording:
            self._dynamic_recorder.cancel()
        if self._dynamic_session is not None and self._dynamic_session.dataset.examples:
            try:
                self.dynamic_store.save(self._dynamic_session.dataset)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Could not save dynamic sequences", str(exc), parent=self.root)
                return
        self.hand_engine.stop()
        self._studio_stage = "dynamic_review"
        self.show_page("Gesture Studio")

    def _return_dynamic_record(self) -> None:
        self._studio_stage = "dynamic_record"
        self.show_page("Gesture Studio")

    def _remove_dynamic_example(self, table: ttk.Treeview) -> None:
        if self._dynamic_session is None:
            return
        selection = table.selection()
        if not selection:
            return
        try:
            self._dynamic_session.remove_example(selection[0])
            self.dynamic_store.save(self._dynamic_session.dataset)
            self._render_dynamic_review_page()
        except (KeyError, OSError, ValueError) as exc:
            messagebox.showerror("Could not remove sequence", str(exc), parent=self.root)

    def _render_dynamic_review_page(self) -> None:
        self._studio_stage = "dynamic_review"
        self.show_page("Gesture Studio")

    def _cancel_dynamic_workflow(self) -> None:
        if self._dynamic_recorder is not None:
            self._dynamic_recorder.cancel()
        self.hand_engine.stop()
        self._dynamic_recorder = None
        self._dynamic_session = None
        self._studio_stage = "details"
        self.show_page("My Gestures")

    def _start_dynamic_training(self) -> None:
        if self._dynamic_training_running:
            return
        try:
            datasets = self.dynamic_store.list()
            if len(datasets) < 2:
                raise ValueError("Record at least two dynamic gesture labels before training.")
            for dataset in datasets:
                if len(dataset.examples) < 3:
                    raise ValueError(f"{dataset.name!r} needs at least 3 completed sequences.")
        except (OSError, ValueError) as exc:
            messagebox.showinfo("Dynamic evaluation not ready", str(exc), parent=self.root)
            return
        self._studio_stage = "dynamic_evaluation"
        self._dynamic_training_running = True
        self._dynamic_saved_model = None
        self.show_page("Gesture Studio")

        def train() -> None:
            try:
                model = DynamicGestureTrainer().train(datasets)
                self.dynamic_model_store.save(model)
                self._service_events.put(("dynamic_training_done", model))
            except Exception as exc:
                self._service_events.put(("dynamic_training_error", str(exc)))
        self._dynamic_training_thread = Thread(target=train, name="GestureManagerDynamicTraining", daemon=False)
        self._dynamic_training_thread.start()

    def _start_studio_detection(self) -> None:
        index = 0
        if self._camera_index_var is not None:
            try:
                index = int(self._camera_index_var.get().strip())
            except ValueError:
                pass
        started = self.hand_engine.start(EngineConfig(
            camera=CameraConfig(index=index, width=640, height=480), processing_rate_hz=8.0,
            model_dir=model_assets_directory()))
        if not started and self.hand_engine.status.state not in (EngineState.RUNNING, EngineState.STARTING):
            if self._studio_status_label is not None:
                self._studio_status_label.configure(text="Detection could not start; stop the existing camera session and retry.")
        self._schedule_camera_poll()

    def _record_gesture_example(self) -> None:
        session = self._studio_session
        status = self.hand_engine.status
        if session is None:
            return
        if status.state is not EngineState.RUNNING or status.hands_detected is not True:
            message = "Waiting for the required hand pose. Check the instructions and camera preview."
            if self._studio_status_label is not None:
                self._studio_status_label.configure(text=message)
            return
        if status.processed_frames == self._last_recorded_processed_frames:
            if self._studio_status_label is not None:
                self._studio_status_label.configure(text="Wait for the next fresh landmark sample before recording again.")
            return
        try:
            session.add_hands(status.hands)
        except ValueError as exc:
            if self._studio_status_label is not None:
                self._studio_status_label.configure(text=f"Example not recorded: {exc}")
            return
        self._last_recorded_processed_frames = status.processed_frames
        if self._recording_progress_label is not None:
            count = len(session.dataset.examples)
            self._recording_progress_label.configure(text=f"{count} examples  |  " +
                ("dataset meets the minimum example count" if count >= 3 else f"{3-count} more needed before training"))
        if self._studio_status_label is not None:
            self._studio_status_label.configure(text="Example added as normalized numeric landmarks. No image was saved.")

    def _open_studio_review(self) -> None:
        self.hand_engine.stop()
        self._studio_stage = "review"
        self.show_page("Gesture Studio")

    def _open_studio_record(self) -> None:
        self._studio_stage = "record"
        self._last_recorded_processed_frames = self.hand_engine.status.processed_frames
        self.show_page("Gesture Studio")

    def _remove_review_example(self, table: ttk.Treeview) -> None:
        session = self._studio_session
        selected = table.selection()
        if session is None or not selected:
            return
        try:
            session.remove_example(selected[0])
            self._studio_stage = "review"
            self.show_page("Gesture Studio")
        except KeyError as exc:
            messagebox.showerror("Example not found", str(exc), parent=self.root)

    def _save_studio_dataset(self) -> None:
        session = self._studio_session
        if session is None:
            return
        if not session.dataset.examples:
            messagebox.showerror("No examples", "Record at least one valid example before saving this dataset.", parent=self.root)
            return
        try:
            self.gesture_store.save(session.dataset)
            self._sync_active_profile(sync_assets=True)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not save dataset", str(exc), parent=self.root)
            return
        self._invalidate_trained_model()
        self.hand_engine.stop()
        self._studio_stage = "saved"
        self.show_page("Gesture Studio")

    def _cancel_studio(self) -> None:
        self.hand_engine.stop()
        if self._studio_session is not None:
            self._studio_session.cancel()
        self._studio_session = None
        self._studio_stage = "details"
        self.show_page("My Gestures")

    def _finish_studio(self) -> None:
        self._studio_session = None
        self._studio_stage = "details"
        self.show_page("My Gestures")

    def _rename_gesture(self, dataset: GestureDataset) -> None:
        name = simpledialog.askstring("Rename gesture", "New gesture name:", initialvalue=dataset.name, parent=self.root)
        if name is None:
            return
        try:
            self.gesture_store.rename(dataset.id, name)
            self._sync_active_profile(sync_assets=True)
            self._invalidate_trained_model()
            self.show_page("My Gestures")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not rename gesture", str(exc), parent=self.root)

    def _delete_gesture(self, dataset: GestureDataset) -> None:
        confirmed = messagebox.askyesno("Delete gesture", f"Delete {dataset.name!r} and its local feature dataset? This cannot be undone.", parent=self.root)
        if not confirmed:
            return
        try:
            self.gesture_store.delete(dataset.id)
            self._sync_active_profile(sync_assets=True)
            self._invalidate_trained_model()
            if self._studio_session is not None and self._studio_session.dataset.id == dataset.id:
                self._studio_session = None
            self.show_page("My Gestures")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not delete gesture", str(exc), parent=self.root)

    def _invalidate_trained_model(self) -> None:
        self._model_current = False
        self._training_outcome = None
        if self._saved_model is not None:
            self._training_message = "Saved model is stale because local examples changed. Retrain to update it."

    def _render_actions(self) -> None:
        panel = ttk.Labelframe(self.page_body, text="Gesture actions", padding=18,
                               bootstyle="secondary")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        action_toolbar = ttk.Frame(panel)
        action_toolbar.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(action_toolbar, text="Clear gesture actions…",
                   command=self._clear_gesture_actions,
                   style="GMReadable.TButton").pack(side="left")
        ttk.Label(action_toolbar, text="Removes saved shortcut mappings only; control gestures and training data stay intact.",
                  bootstyle="secondary", wraplength=650).pack(side="left", padx=(10, 0))
        if self._action_store_error:
            ttk.Label(panel, text=f"Action settings could not be loaded: {self._action_store_error}",
                      bootstyle="danger", wraplength=720).grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(panel, text=("Assign one supported keyboard shortcut to each trained custom gesture. "
                               "Actions are independent of recognition enablement and classifier training."),
                  wraplength=720).grid(row=0, column=0, sticky="w")
        try:
            available_gestures = self.gesture_store.list()
        except (OSError, ValueError):
            available_gestures = ()
        labels = available_gestures or (self._saved_model.labels if self._saved_model is not None else ())
        self._stop_gesture_choices = {"Disabled": None}
        for label in labels:
            gesture_id = getattr(label, "dataset_id", getattr(label, "id", None))
            if gesture_id is not None:
                self._stop_gesture_choices[label.name] = gesture_id
        selected_name = next((name for name, key in self._stop_gesture_choices.items()
                              if key == self._stop_gesture_id), "Disabled")
        self._stop_gesture_choice = tk.StringVar(self.root, value=selected_name)
        stop_panel = ttk.Labelframe(panel, text="Stop Gesture · static custom poses", padding=12,
                                    bootstyle="warning")
        stop_panel.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        stop_panel.columnconfigure(0, weight=1)
        ttk.Label(stop_panel, text=("Select a trained static gesture to stop recognition and release the camera. "
                                    "This works independently of its shortcut mapping and real-input mode."),
                  wraplength=720).grid(row=0, column=0, sticky="w")
        stop_row = ttk.Frame(stop_panel)
        stop_row.grid(row=1, column=0, sticky="ew", pady=(7, 0))
        stop_selector = ttk.Combobox(stop_row, textvariable=self._stop_gesture_choice,
                                     values=tuple(self._stop_gesture_choices),
                                     state="readonly", width=32)
        stop_selector.pack(side="left")
        stop_selector.bind("<<ComboboxSelected>>",
                           lambda _event: self._save_stop_gesture())
        ttk.Button(stop_row, text="Save Stop Gesture", command=self._save_stop_gesture,
                   style="GMReadable.TButton").pack(side="left", padx=(8, 0))
        self._stop_gesture_status_label = ttk.Label(
            stop_panel, text=self._stop_gesture_status_text(), wraplength=720,
            bootstyle="danger" if self._stop_gesture_error else "secondary")
        self._stop_gesture_status_label.grid(row=2, column=0, sticky="w", pady=(6, 0))
        pause_name = next((name for name, key in self._stop_gesture_choices.items()
                           if key == self._pause_gesture_id), "Disabled")
        resume_name = next((name for name, key in self._stop_gesture_choices.items()
                            if key == self._resume_gesture_id), "Disabled")
        self._pause_gesture_choice = tk.StringVar(self.root, value=pause_name)
        self._resume_gesture_choice = tk.StringVar(self.root, value=resume_name)
        controls = ttk.Labelframe(panel, text="Shortcut Pause / Resume Gestures",
                                  padding=12, bootstyle="info")
        controls.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        controls.columnconfigure(0, weight=1)
        ttk.Label(
            controls,
            text=("Pause blocks ordinary shortcut actions while camera recognition stays active at a lower 2.5 FPS target. "
                  "Resume restores eligible shortcut actions. Pause and Resume may share one label; "
                  "it pauses while Active and resumes while Paused. Stop must use a different label."),
            wraplength=720,
        ).grid(row=0, column=0, sticky="w")
        control_row = ttk.Frame(controls)
        control_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(control_row, text="Pause:").pack(side="left", padx=(0, 5))
        ttk.Combobox(control_row, textvariable=self._pause_gesture_choice,
                     values=tuple(self._stop_gesture_choices), state="readonly",
                     width=22).pack(side="left", padx=(0, 12))
        ttk.Label(control_row, text="Resume:").pack(side="left", padx=(0, 5))
        ttk.Combobox(control_row, textvariable=self._resume_gesture_choice,
                     values=tuple(self._stop_gesture_choices), state="readonly",
                     width=22).pack(side="left", padx=(0, 8))
        ttk.Button(control_row, text="Save controls", command=self._save_pause_resume_gestures,
                   style="GMReadable.TButton").pack(side="left")
        state_row = ttk.Frame(controls)
        state_row.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self._gesture_control_status_label = ttk.Label(state_row, text="")
        self._gesture_control_status_label.pack(side="left", anchor="w")
        self._gesture_control_toggle_button = ttk.Button(
            state_row, text="", command=self._toggle_gesture_actions_manually,
            style="GMPauseControl.TButton")
        self._gesture_control_toggle_button.pack(side="right")
        self._refresh_gesture_control_state()
        if self._saved_model is None:
            ttk.Label(panel, text="Train a custom gesture model before assigning mappings.",
                      bootstyle="warning").grid(row=5, column=0, sticky="w", pady=(10, 0))
            no_model_mode = ttk.Labelframe(panel, text="Execution mode", padding=12,
                                           bootstyle="info")
            no_model_mode.grid(row=6, column=0, sticky="ew", pady=(10, 0))
            current_mode = ("Emergency stop active · actions blocked" if self.action_dispatcher.emergency_stopped
                            else "Dry-run · no keystrokes are sent" if self.action_dispatcher.dry_run
                            else "Real input · shortcuts can be sent")
            ttk.Label(no_model_mode, text=f"Execution mode: {current_mode}").grid(
                row=0, column=0, sticky="w")
            self._dry_run_var.set(self.action_dispatcher.dry_run)
            ttk.Checkbutton(
                no_model_mode, text="Dry-run mode (On = never send keyboard input)",
                variable=self._dry_run_var, command=self._execution_mode_toggled,
                bootstyle="success",
                state="disabled" if self.action_dispatcher.emergency_stopped else "normal",
            ).grid(row=1, column=0, sticky="w", pady=(5, 0))
            ttk.Button(no_model_mode, text="Enable real keyboard input…",
                       command=self._confirm_real_input, bootstyle="warning outline",
                       state="disabled" if self.action_dispatcher.emergency_stopped else "normal").grid(
                           row=2, column=0, sticky="w", pady=(6, 0))
            preference = "Dry-run" if self._profile_mode_preference else "Real input (confirmation required each session)"
            ttk.Label(no_model_mode,
                      text=f"Active profile preference: {preference}. Session startup always uses Dry-run.",
                      wraplength=700, bootstyle="secondary").grid(row=3, column=0, sticky="w", pady=(5, 0))
            return
        if not self._model_current:
            ttk.Label(panel, text="The saved model is stale. Mappings can be edited, but live actions remain unavailable until retraining.",
                      bootstyle="warning", wraplength=720).grid(row=5, column=0, sticky="w", pady=(8, 0))
        self._action_shortcut_vars = {}
        self._action_enabled_vars = {}
        for row_index, gesture in enumerate(self._saved_model.labels, start=6):
            existing = self._action_mappings.get(gesture.dataset_id)
            row = ttk.Labelframe(panel, padding=10, bootstyle="light")
            row.grid(row=row_index, column=0, sticky="ew", pady=(8, 0))
            row.columnconfigure(1, weight=1)
            enabled_var = tk.BooleanVar(self.root, value=existing.enabled if existing else False)
            shortcut_var = tk.StringVar(self.root, value=existing.shortcut if existing else "")
            self._action_enabled_vars[gesture.dataset_id] = enabled_var
            self._action_shortcut_vars[gesture.dataset_id] = shortcut_var
            ttk.Label(row, text=gesture.name, font=("Segoe UI", 11, "bold"),
                      bootstyle="primary").grid(row=0, column=0, columnspan=3,
                                                 sticky="w", pady=(0, 7))
            ttk.Checkbutton(row, text="Action enabled", variable=enabled_var,
                            bootstyle="success").grid(row=1, column=0, sticky="w", padx=(0, 10))
            entry = ttk.Entry(row, textvariable=shortcut_var)
            entry.grid(row=1, column=1, sticky="ew", padx=(0, 8))
            ttk.Button(row, text="Record", command=lambda key=gesture.dataset_id: self._record_action_shortcut(key),
                       style="GMReadable.TButton").grid(row=1, column=2, sticky="e")
            ttk.Label(row, text="Example: Ctrl+C, Ctrl+Shift+Z, Alt+Tab. Modifiers plus one key are required.",
                      bootstyle="secondary", font=("Segoe UI", 9), wraplength=700).grid(
                          row=2, column=0, columnspan=3, sticky="w", pady=(5, 0))

        dry_run_card = ttk.Labelframe(panel, text="Safe execution controls", padding=12,
                                      bootstyle="info")
        dry_run_card.grid(row=100, column=0, sticky="ew", pady=(12, 0))
        dry_run_card.columnconfigure(0, weight=1)
        if self.action_dispatcher.emergency_stopped:
            mode = "Emergency stop active · actions blocked; recognition may continue"
        elif self.action_dispatcher.dry_run:
            mode = "Dry-run · no keystrokes are sent"
        else:
            mode = "Real input · configured shortcuts can be sent"
        self._execution_mode_label = ttk.Label(
            dry_run_card, text=f"Execution mode: {mode}",
            bootstyle="success" if self.action_dispatcher.dry_run else "warning")
        self._execution_mode_label.grid(row=0, column=0, sticky="w")
        self._dry_run_var.set(self.action_dispatcher.dry_run)
        ttk.Checkbutton(
            dry_run_card, text="Dry-run mode (On = never send keyboard input)",
            variable=self._dry_run_var, command=self._execution_mode_toggled,
            bootstyle="success",
            state="disabled" if self.action_dispatcher.emergency_stopped else "normal",
        ).grid(row=1, column=0, sticky="w", pady=(5, 0))
        preference = "Dry-run" if self._profile_mode_preference else "Real input (confirmation required each session)"
        pref_label = (f"Active profile preference: {preference}. The current session always starts in Dry-run."
                      if not self._profile_mode_error else
                      f"Profile preference unavailable; using Dry-run: {self._profile_mode_error}")
        ttk.Label(dry_run_card, text=pref_label, wraplength=720,
                  bootstyle="secondary" if not self._profile_mode_error else "danger").grid(
                      row=2, column=0, sticky="w", pady=(4, 0))
        buttons = ttk.Frame(dry_run_card)
        buttons.grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Button(buttons, text="Save mappings", command=self._save_action_mappings,
                   bootstyle="primary").pack(side="left", padx=(0, 7))
        ttk.Button(buttons, text="Enable real keyboard input…", command=self._confirm_real_input,
                   bootstyle="warning outline", state="disabled" if self.action_dispatcher.emergency_stopped else "normal").pack(side="left", padx=3)
        ttk.Button(buttons, text="Return to dry run", command=self._return_to_dry_run,
                   bootstyle="secondary outline").pack(side="left", padx=3)
        ttk.Button(buttons, text="Rearm actions", command=self.rearm_actions,
                   bootstyle="danger outline", state="normal" if self.action_dispatcher.emergency_stopped else "disabled").pack(side="left", padx=3)
        ttk.Button(buttons, text="GLOBAL EMERGENCY STOP", command=self.emergency_stop_global,
                   bootstyle="danger").pack(side="left", padx=(8, 0))
        ttk.Label(dry_run_card, text=("Only the allowlisted keyboard_shortcut action is supported. No shell commands, scripts, "
                                     "or arbitrary profile content are executed. Real shortcuts go to whichever app currently has focus."),
                  wraplength=720, bootstyle="secondary").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self._action_status_label = ttk.Label(panel, text="Mappings are stored locally, separately from gesture data and models.",
                                              wraplength=720)
        self._action_status_label.grid(row=101, column=0, sticky="w", pady=(8, 0))

    def _stop_gesture_status_text(self) -> str:
        if self._stop_gesture_error:
            return f"Settings error; Stop Gesture is disabled until corrected: {self._stop_gesture_error}"
        if self._stop_gesture_id is None:
            return "No Stop Gesture configured. Manual Stop Service remains available."
        try:
            datasets = self.gesture_store.list()
        except (OSError, ValueError):
            datasets = ()
        name = next((item.name for item in datasets if item.id == self._stop_gesture_id), None)
        if name is None and self._saved_model is not None:
            name = next((label.name for label in self._saved_model.labels
                         if label.dataset_id == self._stop_gesture_id), None)
        return (f"Configured: {name}. Requires the normal stable, confident recognition gate."
                if name else "Configured gesture is not available in the current model; runtime start will be blocked.")

    def _refresh_stop_gesture_status(self) -> None:
        label = getattr(self, "_stop_gesture_status_label", None)
        if label is not None and label.winfo_exists():
            label.configure(text=self._stop_gesture_status_text(),
                            bootstyle="danger" if self._stop_gesture_error else "secondary")

    def _save_stop_gesture(self) -> None:
        if self._stop_gesture_error:
            messagebox.showerror("Stop Gesture settings unavailable", self._stop_gesture_error,
                                 parent=self.root)
            return
        selected = self._stop_gesture_choice.get() if self._stop_gesture_choice else "Disabled"
        gesture_id = self._stop_gesture_choices.get(selected)
        if selected not in self._stop_gesture_choices:
            messagebox.showerror("Invalid Stop Gesture", "Select a gesture from the current static model.",
                                 parent=self.root)
            return
        try:
            self._active_profile = self.profile_store.update_execution_settings(
                self._active_profile["id"],
                dry_run=self._profile_mode_preference,
                stop_gesture_id=gesture_id)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not save Stop Gesture", str(exc), parent=self.root)
            return
        self._stop_gesture_id = gesture_id
        self._stop_gesture_error = None
        self._stop_gesture_trigger = StopGestureTrigger()
        if gesture_id is not None:
            self._recognition_preferences[gesture_id] = (
                True, self._recognition_preferences.get(gesture_id, (True, 0.0))[1])
            enabled_var = self._recognition_enabled_vars.get(gesture_id)
            if enabled_var is not None:
                enabled_var.set(True)
            worker = self._live_classifier
            if worker is not None and worker.is_alive:
                worker.gate.update_gesture(gesture_id, enabled=True)
        was_paused = self._gesture_actions_paused
        self._gesture_controls.configure(
            pause_gesture_id=self._pause_gesture_id,
            resume_gesture_id=self._resume_gesture_id,
            stop_gesture_id=self._stop_gesture_id,
        )
        self._gesture_actions_paused = was_paused
        self._gesture_controls.set_actions_paused(was_paused)
        self._sync_active_profile()
        self._refresh_stop_gesture_status()

    def _save_action_mappings(self) -> None:
        model = self._saved_model
        if model is None:
            return
        updated = {key: value for key, value in self._action_mappings.items()
                   if key not in {label.dataset_id for label in model.labels}}
        try:
            for label in model.labels:
                shortcut = self._action_shortcut_vars[label.dataset_id].get().strip()
                enabled = self._action_enabled_vars[label.dataset_id].get()
                if not shortcut:
                    if enabled:
                        raise ValueError(f"Enter a shortcut for enabled action '{label.name}'")
                    continue
                updated[label.dataset_id] = ActionMapping(
                    gesture_id=label.dataset_id, gesture_name=label.name,
                    action_type="keyboard_shortcut", shortcut=shortcut, enabled=enabled,
                )
            self.action_store.save(updated)
        except (OSError, ValueError) as exc:
            if self._action_status_label is not None:
                self._action_status_label.configure(text=f"Could not save mappings: {exc}", bootstyle="danger")
            return
        self._action_mappings = updated
        self._action_store_error = None
        self._sync_active_profile()
        if self._action_status_label is not None:
            self._action_status_label.configure(text="Action mappings saved locally. Classifier training was not changed.", bootstyle="success")

    def _clear_gesture_actions(self) -> None:
        """Clear shortcut mappings without touching gesture data or control gestures."""
        if not messagebox.askyesno(
                "Clear gesture actions",
                "Remove all saved shortcut mappings from the active profile? This does not delete gesture data or Pause, Resume, and Stop settings.",
                parent=self.root):
            return
        try:
            self.action_store.save({})
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not clear gesture actions", str(exc), parent=self.root)
            return
        self._action_mappings = {}
        self._action_store_error = None
        self._sync_active_profile()
        for variable in getattr(self, "_action_shortcut_vars", {}).values():
            variable.set("")
        for variable in getattr(self, "_action_enabled_vars", {}).values():
            variable.set(False)
        if self._action_status_label is not None:
            self._action_status_label.configure(
                text="Shortcut mappings cleared from the active profile.", bootstyle="success")

    def _record_action_shortcut(self, gesture_id: str) -> None:
        variable = self._action_shortcut_vars.get(gesture_id)
        if variable is None:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Record shortcut")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()
        self._action_record_dialog = dialog
        self._action_record_target = variable
        self._action_recorded_shortcut = None
        body = ttk.Frame(dialog, padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Capture a keyboard shortcut",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(body, text=("Press Start capture, then press one keyboard key by itself or hold a modifier and "
                             "press a key. The global listener is active only during this recording. On Windows, "
                             "the captured key or chord is suppressed so it is recorded without being performed. "
                             "F1–F12 and standard volume/media keys are supported. The laptop Fn key itself is "
                             "handled by keyboard firmware and cannot be captured as a modifier. Fn combos work "
                             "only if Windows reports a supported key; brightness and vendor-specific keys may "
                             "not be available to apps."),
                  wraplength=460).pack(anchor="w", pady=(5, 12))
        self._action_record_status = ttk.Label(body, text="Not listening. No keys are being monitored.",
                                               bootstyle="secondary", wraplength=460)
        self._action_record_status.pack(anchor="w", pady=(0, 12))
        buttons = ttk.Frame(body)
        buttons.pack(fill="x")
        start_button = ttk.Button(buttons, text="Start capture",
                                  command=lambda: self._start_shortcut_capture(dialog, start_button),
                                  style="GMReadable.TButton")
        start_button.pack(side="left")
        self._action_record_button = ttk.Button(buttons, text="Use captured shortcut",
                                                command=self._use_captured_shortcut,
                                                bootstyle="success", state="disabled")
        self._action_record_button.pack(side="left", padx=7)
        ttk.Button(buttons, text="Cancel", command=self._close_shortcut_recorder,
                   bootstyle="secondary outline").pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", self._close_shortcut_recorder)
        dialog.bind("<Escape>", lambda _event: self._close_shortcut_recorder())

    def _start_shortcut_capture(self, dialog: tk.Toplevel, start_button: ttk.Button) -> None:
        if self._action_record_listener is not None and self._action_record_listener.is_alive():
            return
        try:
            from pynput.keyboard import Key, Listener
        except ImportError:
            if self._action_record_status is not None:
                self._action_record_status.configure(
                    text="Global recording needs pynput. Install the updated project dependencies, then restart.",
                    bootstyle="danger")
            return
        self._action_recorded_shortcut = None
        if self._action_record_button is not None:
            self._action_record_button.configure(state="disabled")
        self._action_record_queue = queue.Queue(maxsize=2)
        held_modifiers: set[str] = set()
        listener_ref: list[object] = []
        captured = False

        def complete_capture(shortcut: str) -> None:
            nonlocal captured
            if captured:
                return
            captured = True
            try:
                self._action_record_queue.put_nowait(("captured", shortcut))
            except queue.Full:
                pass
            if listener_ref:
                listener_ref[0].stop()

        def windows_event_filter(message: int, data: object) -> bool:
            """Capture and suppress only a supported chord on Windows."""
            if captured:
                return True
            vk = getattr(data, "vkCode", -1)
            flags = getattr(data, "flags", 0)
            if flags & 0x2000:  # Do not react to injected key events.
                return True
            is_press = message in (0x0100, 0x0104)  # WM_KEYDOWN / WM_SYSKEYDOWN
            is_release = message in (0x0101, 0x0105)  # WM_KEYUP / WM_SYSKEYUP
            if not (is_press or is_release):
                return True
            token = windows_vk_token(vk)
            if token in ("Ctrl", "Alt", "Shift", "Win"):
                if is_press:
                    held_modifiers.add(token)
                else:
                    held_modifiers.discard(token)
                if listener_ref:
                    listener_ref[0].suppress_event()
                return True
            if not is_press or token is None:
                return True
            try:
                shortcut = normalize_shortcut("+".join((*held_modifiers, token)))
            except ValueError as exc:
                try:
                    self._action_record_queue.put_nowait(("error", str(exc)))
                except queue.Full:
                    pass
                return True
            complete_capture(shortcut)
            if listener_ref:
                listener_ref[0].suppress_event()
            return True

        def on_press(key: object) -> None:
            token = pynput_key_token(key, Key)
            if token in ("Ctrl", "Alt", "Shift", "Win"):
                held_modifiers.add(token)
                return
            if token is None:
                return
            try:
                shortcut = normalize_shortcut("+".join((*held_modifiers, token)))
            except ValueError as exc:
                try:
                    self._action_record_queue.put_nowait(("error", str(exc)))
                except queue.Full:
                    pass
                return
            complete_capture(shortcut)

        def on_release(key: object) -> None:
            token = pynput_key_token(key, Key)
            if token in ("Ctrl", "Alt", "Shift", "Win"):
                held_modifiers.discard(token)

        try:
            listener_options = {"suppress": False}
            if platform.system() == "Windows":
                listener_options["win32_event_filter"] = windows_event_filter
            listener = Listener(on_press=on_press, on_release=on_release, **listener_options)
            listener_ref.append(listener)
            self._action_record_listener = listener
            listener.start()
        except Exception as exc:
            if self._action_record_status is not None:
                self._action_record_status.configure(text=f"Could not start shortcut capture: {exc}",
                                                     bootstyle="danger")
            return
        if self._action_record_status is not None:
            self._action_record_status.configure(
                text="Listening globally. On Windows, supported modifier keys are temporarily blocked from other apps; they are released when capture ends.",
                bootstyle="warning")
        start_button.configure(state="disabled")
        self._poll_shortcut_capture(dialog, start_button)

    def _poll_shortcut_capture(self, dialog: tk.Toplevel, start_button: ttk.Button) -> None:
        if not dialog.winfo_exists():
            self._stop_shortcut_listener()
            return
        while True:
            try:
                kind, value = self._action_record_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "captured":
                self._action_recorded_shortcut = value
                if self._action_record_status is not None:
                    self._action_record_status.configure(text=f"Captured: {value}", bootstyle="success")
                if self._action_record_button is not None:
                    self._action_record_button.configure(state="normal")
            elif self._action_record_status is not None:
                self._action_record_status.configure(text=value, bootstyle="danger")
        listener = self._action_record_listener
        if listener is not None and listener.is_alive():
            dialog.after(35, lambda: self._poll_shortcut_capture(dialog, start_button))
        else:
            start_button.configure(state="normal")

    def _use_captured_shortcut(self) -> None:
        if self._action_record_target is None or self._action_recorded_shortcut is None:
            return
        self._action_record_target.set(self._action_recorded_shortcut)
        self._close_shortcut_recorder()

    def _stop_shortcut_listener(self) -> None:
        listener = self._action_record_listener
        if listener is not None and listener.is_alive():
            listener.stop()

    def _close_shortcut_recorder(self) -> None:
        self._stop_shortcut_listener()
        dialog = self._action_record_dialog
        self._action_record_dialog = None
        self._action_record_target = None
        self._action_record_status = None
        self._action_record_button = None
        if dialog is not None and dialog.winfo_exists():
            dialog.destroy()

    def _confirm_real_input(self) -> None:
        if self.action_dispatcher.emergency_stopped:
            messagebox.showwarning("Emergency stop active", "Rearm actions explicitly before enabling real input.", parent=self.root)
            return
        confirmed = messagebox.askyesno(
            "Enable real keyboard input?",
            "Gesture Manager will send configured shortcuts to the currently focused application when an enabled gesture becomes stable. This may change documents or trigger actions in that application. Continue?",
            icon="warning", parent=self.root,
        )
        if not confirmed:
            self._dry_run_var.set(True)
            return
        if platform.system() != "Windows":
            self._dry_run_var.set(True)
            messagebox.showerror("Unsupported operating system", "Real keyboard dispatch is supported only on Windows.", parent=self.root)
            return
        try:
            backend = PynputKeyboardBackend()
        except RuntimeError as exc:
            self._dry_run_var.set(True)
            messagebox.showerror("Keyboard backend unavailable", str(exc), parent=self.root)
            return
        self.action_dispatcher.set_backend(backend)
        self.action_dispatcher.confirm_real_input(confirmed=True)
        self._real_input_confirmed = True
        self._dry_run_var.set(False)
        self._save_profile_mode_preference(False)
        self.show_page("Actions")

    def _apply_saved_execution_preference(self) -> None:
        """Offer to restore a profile's Real-input choice with explicit consent.

        The profile stores the user's choice. The running session still starts
        in Dry-run, satisfying the confirmation gate after restart/profile
        changes. Declining this prompt does not overwrite the saved choice.
        """
        if self._profile_mode_error or self._profile_mode_preference:
            return
        if self.action_dispatcher.emergency_stopped or not self.root.winfo_exists():
            return
        self._confirm_real_input()

    def _execution_mode_toggled(self) -> None:
        """Turn Dry-run off only by routing through explicit confirmation."""
        if not self._dry_run_var.get():
            self._dry_run_var.set(True)
            self._confirm_real_input()
        else:
            self._return_to_dry_run()

    def _save_profile_mode_preference(self, dry_run: bool) -> None:
        try:
            self._active_profile = self.profile_store.update_execution_settings(
                self._active_profile["id"],
                dry_run=dry_run,
                stop_gesture_id=self._stop_gesture_id)
            self._profile_mode_preference = dry_run
            self._profile_mode_error = None
        except (OSError, ValueError) as exc:
            self._profile_mode_preference = True
            self._profile_mode_error = str(exc)

    def _refresh_execution_mode_label(self) -> None:
        label = getattr(self, "_execution_mode_label", None)
        if label is None or not label.winfo_exists():
            return
        if self.action_dispatcher.emergency_stopped:
            mode = "Emergency stop active · actions blocked; recognition may continue"
        elif self.action_dispatcher.dry_run:
            mode = "Dry-run · no keystrokes are sent"
        else:
            mode = "Real input · configured shortcuts can be sent"
        label.configure(text=f"Execution mode: {mode}",
                        bootstyle="success" if self.action_dispatcher.dry_run else "warning")

    def _return_to_dry_run(self) -> None:
        self.action_dispatcher.set_dry_run(True)
        self._real_input_confirmed = False
        self._dry_run_var.set(True)
        self._save_profile_mode_preference(True)
        if self.current_page == "Actions":
            self.show_page("Actions")

    def emergency_stop_global(self) -> None:
        self.action_dispatcher.emergency_stop()
        if self._action_status_label is not None and self._action_status_label.winfo_exists():
            self._action_status_label.configure(text="GLOBAL EMERGENCY STOP · actions are blocked; recognition may continue. Stop Service releases the camera.",
                                                bootstyle="danger")
        if self.current_page == "Actions":
            self.show_page("Actions")

    def rearm_actions(self) -> None:
        if not self.action_dispatcher.emergency_stopped:
            return
        if not messagebox.askyesno("Rearm action dispatcher?",
                                   "Clear the emergency stop? Action dispatch returns in dry-run mode.",
                                   icon="warning", parent=self.root):
            return
        self.action_dispatcher.rearm()
        self._real_input_confirmed = False
        if self.current_page == "Actions":
            self.show_page("Actions")

    def _render_profiles(self) -> None:
        panel = ttk.Labelframe(self.page_body, text="Gesture profiles", padding=18, bootstyle="secondary")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="Profiles keep custom datasets, the matching local model, action mappings, and per-gesture settings together.",
                  wraplength=760).grid(row=0, column=0, sticky="w")
        profiles = self.profile_store.list()
        self._profile_name_var = tk.StringVar(self.root, value=self._active_profile["name"])
        selector = ttk.Combobox(panel, textvariable=self._profile_name_var,
                                values=[item["name"] for item in profiles], state="readonly", width=32)
        selector.grid(row=1, column=0, sticky="w", pady=(12, 8))
        selector.bind("<<ComboboxSelected>>", lambda _event: self._select_profile_name())
        ttk.Label(panel, text=f"Active profile: {self._active_profile['name']} · {len(self._active_profile['gesture_definitions'])} gestures",
                  bootstyle="info").grid(row=2, column=0, sticky="w")
        buttons = ttk.Frame(panel)
        buttons.grid(row=3, column=0, sticky="w", pady=(10, 0))
        for title, command, style in (
            ("Create", self._create_profile, "primary"), ("Rename", self._rename_profile, "secondary outline"),
            ("Validate", self._validate_active_profile, "secondary outline"),
            ("Export…", self._export_active_profile, "secondary outline"),
            ("Import…", self._import_profile, "secondary outline"),
            ("Delete", self._delete_active_profile, "danger outline"),
        ):
            ttk.Button(buttons, text=title, command=command, bootstyle=style).pack(side="left", padx=(0, 6))
        self._profile_status = ttk.Label(panel, text="Profile data remains local unless you export an archive.", wraplength=760)
        self._profile_status.grid(row=4, column=0, sticky="w", pady=(10, 0))
        info = ttk.Labelframe(self.page_body, text="Portable format", padding=14, bootstyle="info")
        info.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(info, text="A .gmp ZIP contains profile.json, feature-only dataset JSON files, and the validated data-only model JSON when present. Import validates paths, schemas, datasets, model, shortcuts, thresholds, and archive size before installation. Profiles cannot contain executable content.",
                  wraplength=760, justify="left").pack(anchor="w")

    def _select_profile_name(self) -> None:
        name = self._profile_name_var.get() if self._profile_name_var is not None else ""
        profile = next((item for item in self.profile_store.list() if item["name"] == name), None)
        if profile is not None and profile["id"] != self._active_profile["id"]:
            self._activate_profile(profile)

    def _activate_profile(self, profile: dict[str, object]) -> None:
        if self._profile_io_thread is not None and self._profile_io_thread.is_alive():
            if self._profile_status is not None and self._profile_status.winfo_exists():
                self._profile_status.configure(text="Wait for profile import/export to finish.", bootstyle="warning")
            return
        if self._training_running:
            messagebox.showinfo("Training in progress", "Wait for profile training to finish before switching profiles.",
                                parent=self.root)
            return
        if self._live_test_active:
            self.controller.stop()
        # A newly selected or imported profile can contain enabled mappings.
        # Require explicit opt-in again before any of them can send input.
        self.action_dispatcher.set_dry_run(True)
        self._real_input_confirmed = False
        self._dry_run_var.set(True)
        selected = self.profile_store.load(str(profile["id"]), validate_assets=False)
        self.profile_store.set_active(selected["id"])
        self._active_profile = selected
        self._configure_profile_storage()
        self._saved_model = None
        self._model_current = False
        self._training_outcome = None
        self._recognition_preferences = {}
        self._recognition_cooldown_preferences = {}
        self._training_message = f"Loading profile '{selected['name']}' model…"
        self._load_saved_model_async()
        self.show_page("Profiles")
        self.root.after(150, self._apply_saved_execution_preference)

    def _create_profile(self) -> None:
        if self._profile_io_thread is not None and self._profile_io_thread.is_alive():
            return
        name = simpledialog.askstring("Create profile", "Profile name:", parent=self.root)
        if name is None:
            return
        try:
            profile = self.profile_store.create(name)
            self._activate_profile(profile)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not create profile", str(exc), parent=self.root)

    def _rename_profile(self) -> None:
        if self._profile_io_thread is not None and self._profile_io_thread.is_alive():
            return
        name = simpledialog.askstring("Rename profile", "New profile name:",
                                      initialvalue=self._active_profile["name"], parent=self.root)
        if name is None:
            return
        try:
            self._active_profile = self.profile_store.rename(self._active_profile["id"], name)
            self.show_page("Profiles")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not rename profile", str(exc), parent=self.root)

    def _validate_active_profile(self) -> None:
        if self._profile_io_thread is not None and self._profile_io_thread.is_alive():
            return
        profile_id = self._active_profile["id"]
        if self._profile_status is not None:
            self._profile_status.configure(text="Validating profile assets in the background…", bootstyle="info")
        def validate() -> None:
            try:
                profile = self.profile_store.load(profile_id)
                datasets = self.profile_store.validate_assets(profile)
                self._service_events.put(("profile_validate_done",
                    f"Valid: {len(datasets)} datasets and profile model/settings passed validation.", True))
            except Exception as exc:
                self._service_events.put(("profile_validate_done", f"Validation failed: {exc}", False))
        self._profile_io_thread = Thread(target=validate, name="GestureManagerProfileValidation", daemon=False)
        self._profile_io_thread.start()
        self._schedule_service_poll()

    def _export_active_profile(self) -> None:
        destination = filedialog.asksaveasfilename(parent=self.root, title="Export gesture profile",
            defaultextension=".gmp", filetypes=(("Gesture Manager profile", "*.gmp"), ("ZIP archive", "*.zip")))
        if not destination:
            return
        if self._profile_io_thread is not None and self._profile_io_thread.is_alive():
            return
        profile_id = self._active_profile["id"]
        if self._profile_status is not None:
            self._profile_status.configure(text="Validating and exporting profile in the background…", bootstyle="info")
        def export() -> None:
            try:
                path = self.profile_store.export(profile_id, Path(destination))
                self._service_events.put(("profile_export_done", str(path)))
            except Exception as exc:
                self._service_events.put(("profile_io_error", f"Could not export profile: {exc}"))
        self._profile_io_thread = Thread(target=export, name="GestureManagerProfileExport", daemon=False)
        self._profile_io_thread.start()
        self._schedule_service_poll()

    def _import_profile(self) -> None:
        source = filedialog.askopenfilename(parent=self.root, title="Import gesture profile",
            filetypes=(("Gesture Manager profile", "*.gmp *.zip"), ("All files", "*.*")))
        if not source:
            return
        if self._profile_io_thread is not None and self._profile_io_thread.is_alive():
            return
        if self._profile_status is not None:
            self._profile_status.configure(text="Validating and importing profile in the background…", bootstyle="info")
        def import_profile() -> None:
            try:
                profile = self.profile_store.import_archive(Path(source))
                self._service_events.put(("profile_import_done", profile))
            except Exception as exc:
                self._service_events.put(("profile_io_error", f"Could not import profile: {exc}"))
        self._profile_io_thread = Thread(target=import_profile, name="GestureManagerProfileImport", daemon=False)
        self._profile_io_thread.start()
        self._schedule_service_poll()

    def _activate_imported_profile(self, profile: object) -> None:
        if self._profile_io_thread is not None and self._profile_io_thread.is_alive():
            self.root.after(50, lambda: self._activate_imported_profile(profile))
            return
        if isinstance(profile, dict):
            self._activate_profile(profile)

    def _delete_active_profile(self) -> None:
        if self._profile_io_thread is not None and self._profile_io_thread.is_alive():
            return
        profiles = self.profile_store.list()
        if len(profiles) <= 1:
            messagebox.showinfo("Keep one profile", "At least one profile must remain.", parent=self.root)
            return
        if not messagebox.askyesno("Delete profile", f"Delete '{self._active_profile['name']}' and its local files?",
                                   icon="warning", parent=self.root):
            return
        alternate = next(item for item in profiles if item["id"] != self._active_profile["id"])
        try:
            if self._live_test_active:
                self.controller.stop()
            self.action_dispatcher.set_dry_run(True)
            self._real_input_confirmed = False
            self.profile_store.set_active(alternate["id"])
            self.profile_store.delete(self._active_profile["id"])
            self._active_profile = alternate
            self._configure_profile_storage()
            self._saved_model = None
            self._model_current = False
            self._load_saved_model_async()
            self.show_page("Profiles")
            self.root.after(150, self._apply_saved_execution_preference)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not delete profile", str(exc), parent=self.root)

    def _sync_active_profile(self, *, sync_assets: bool = False) -> None:
        if not hasattr(self, "_saved_model"):
            return
        try:
            timings = {key: dict(value) for key, value in self._active_profile.get("thresholds", {}).items()
                       if key in self._active_profile["dataset_refs"]}
            for gesture_id, (_enabled, margin) in self._recognition_preferences.items():
                previous = timings.get(gesture_id, {})
                timings[gesture_id] = {
                    "minimum_margin": margin,
                    "cooldown_seconds": self._recognition_cooldown_preferences.get(
                        gesture_id, previous.get("cooldown_seconds", self._recognition_cooldown_seconds)),
                }
            for gesture_id, cooldown in self._recognition_cooldown_preferences.items():
                previous = timings.get(gesture_id, {})
                timings[gesture_id] = {
                    "minimum_margin": previous.get("minimum_margin", 0.0),
                    "cooldown_seconds": cooldown,
                }
            if sync_assets:
                datasets = self.gesture_store.list()
                model = self._saved_model
                ids = {dataset.id for dataset in datasets}
                if model is not None and not {label.dataset_id for label in model.labels}.issubset(ids):
                    model = None
                self._active_profile = self.profile_store.update_configuration(
                    self._active_profile["id"], datasets=datasets, model=model,
                    mappings=self._action_mappings, thresholds=timings, copy_assets=False)
            else:
                self._active_profile = self.profile_store.update_settings(
                    self._active_profile["id"], mappings=self._action_mappings, thresholds=timings)
        except (OSError, ValueError) as exc:
            if self._profile_status is not None and self._profile_status.winfo_exists():
                self._profile_status.configure(text=f"Profile update failed: {exc}", bootstyle="danger")

    def _render_camera(self) -> None:
        panel = ttk.Labelframe(self.page_body, text="Local camera", padding=18,
                               bootstyle="secondary")
        panel.grid(row=0, column=0, sticky="ew")
        panel.columnconfigure(0, weight=1)
        controls = ttk.Labelframe(panel, text="Capture", padding=12, bootstyle="light")
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Camera index").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self._camera_index_var = tk.StringVar(self.root, value=str(self._camera_index))
        self._camera_device_combo = ttk.Combobox(
            controls, textvariable=self._camera_index_var, state="normal", width=12, values=(),
        )
        self._camera_device_combo.grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(controls, text="Scan cameras", command=self._scan_cameras,
                   bootstyle="secondary outline").grid(row=0, column=2, sticky="w", padx=(8, 0), pady=4)
        ttk.Label(controls, text="Resolution").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        self._camera_resolution_var = tk.StringVar(self.root, value="640  |  480")
        ttk.Combobox(controls, textvariable=self._camera_resolution_var,
                     values=("640  |  480", "1280  |  720"), state="readonly", width=12
                     ).grid(row=1, column=1, sticky="w", pady=4)
        capture_buttons = ttk.Frame(controls)
        capture_buttons.grid(row=1, column=2, sticky="e", padx=(8, 0), pady=4)
        ttk.Button(capture_buttons, text="Start capture", command=self._start_camera,
                   bootstyle="primary").pack(side="left", padx=(0, 6))
        ttk.Button(capture_buttons, text="Stop", command=self.camera_manager.stop,
                   bootstyle="secondary outline").pack(side="left")

        self._camera_status_label = ttk.Label(panel, text="Camera stopped",
                                               bootstyle="secondary", wraplength=760)
        self._camera_status_label.grid(row=1, column=0, sticky="w", pady=(8, 5))
        preview_tools = ttk.Frame(panel)
        preview_tools.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        self._camera_preview_var = tk.BooleanVar(self.root, value=False)
        ttk.Checkbutton(preview_tools, text="Show live preview", variable=self._camera_preview_var,
                        command=self._toggle_camera_preview, bootstyle="primary").pack(side="left")
        ttk.Label(preview_tools, text="Preview frames stay in memory and are never saved.",
                  bootstyle="secondary").pack(side="left", padx=(12, 0))
        self._camera_preview = CameraPreview(panel, bootstyle="secondary", height=240)
        self._camera_preview.grid(row=3, column=0, sticky="ew")

        landmark_controls = ttk.Labelframe(panel, text="Local hand landmarks", padding=12,
                                           bootstyle="secondary")
        landmark_controls.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        landmark_controls.columnconfigure(1, weight=1)
        ttk.Label(landmark_controls, text="Processing rate").grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        self._processing_rate_var = tk.StringVar(self.root, value="8 Hz")
        ttk.Combobox(landmark_controls, textvariable=self._processing_rate_var,
                     values=("5 Hz", "8 Hz", "10 Hz", "15 Hz"), state="readonly", width=8
                     ).grid(row=0, column=1, sticky="w", pady=4)
        detect_buttons = ttk.Frame(landmark_controls)
        detect_buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Button(detect_buttons, text="Start detection", command=self._start_detection,
                   bootstyle="primary").pack(side="left", padx=(0, 6))
        ttk.Button(detect_buttons, text="Pause", command=self.hand_engine.pause,
                   bootstyle="secondary outline").pack(side="left", padx=3)
        ttk.Button(detect_buttons, text="Resume", command=self.hand_engine.resume,
                   bootstyle="secondary outline").pack(side="left", padx=3)
        ttk.Button(detect_buttons, text="Stop detection", command=self.hand_engine.stop,
                   bootstyle="secondary outline").pack(side="left", padx=3)
        self._engine_status_label = ttk.Label(
            landmark_controls, text="Detection stopped. The local model has not been loaded.",
            wraplength=760)
        self._engine_status_label.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(panel, text="Camera preview and hand detection use the same camera. Stop capture before starting detection, or stop detection before starting capture.",
                  bootstyle="secondary", wraplength=760).grid(row=5, column=0, sticky="w", pady=(8, 0))
        self._refresh_camera_widgets()

    def _scan_cameras(self) -> None:
        if getattr(self, "_camera_scan_running", False):
            return
        if self.camera_manager.status.state in (
            CameraState.STARTING, CameraState.RUNNING, CameraState.STOPPING
        ):
            if self._camera_message is not None:
                self._camera_message.configure(
                    text="Stop this app's camera capture before scanning devices."
                )
            return
        self._camera_scan_running = True
        if self._camera_message is not None:
            self._camera_message.configure(text="Scanning camera indices…")

        def scan() -> None:
            devices = CameraManager.enumerate_cameras()
            try:
                self._scan_results.put_nowait(devices)
            except queue.Full:
                pass

        Thread(target=scan, name="GestureManagerCameraScan", daemon=True).start()
        self._read_scan_results()

    def _read_scan_results(self) -> None:
        try:
            devices = self._scan_results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._read_scan_results)
            return
        self._camera_scan_running = False
        self._camera_devices = devices
        if hasattr(self, "_camera_device_combo"):
            self._camera_device_combo.configure(values=tuple(str(item.index) for item in devices))
        if self._camera_message is not None:
            self._camera_message.configure(
                text=(f"Found {len(devices)} openable camera(s)." if devices else
                      "No openable camera index found. A device may be absent, busy, or outside the scan range.")
            )

    def _start_camera(self) -> None:
        try:
            index = int(self._camera_index_var.get().strip())
            width, height = ((1280, 720) if self._camera_resolution_var.get() == "1280 × 720"
                             else (640, 480))
            started = self.camera_manager.start(CameraConfig(index=index, width=width, height=height))
            if not started and self._camera_message is not None:
                self._camera_message.configure(text="Camera worker is already running or stopping.")
        except (ValueError, TypeError) as exc:
            if self._camera_message is not None:
                self._camera_message.configure(text=f"Invalid camera selection: {exc}")
        self._refresh_camera_widgets()
        self._schedule_camera_poll()

    def _start_detection(self) -> None:
        try:
            index = int(self._camera_index_var.get().strip())
            width, height = ((1280, 720) if self._camera_resolution_var.get() == "1280 × 720"
                             else (640, 480))
            rate = float(self._processing_rate_var.get().split()[0])
            started = self.hand_engine.start(EngineConfig(
                camera=CameraConfig(index=index, width=width, height=height),
                processing_rate_hz=rate,
                model_dir=model_assets_directory(),
            ))
            if not started and self._camera_message is not None:
                self._camera_message.configure(text="Detection is already running or stopping.")
        except (ValueError, TypeError) as exc:
            if self._camera_message is not None:
                self._camera_message.configure(text=f"Invalid detection settings: {exc}")
        self._schedule_camera_poll()

    def _toggle_camera_preview(self) -> None:
        if not self._camera_preview_var.get() and self._camera_preview is not None:
            self._camera_preview.clear()
        self._camera_last_sequence = -1
        self._schedule_camera_poll()

    def _schedule_camera_poll(self) -> None:
        should_poll = self.current_page == "Camera" or (
            self.current_page == "Gesture Studio" and self._studio_preview is not None
            and (self._studio_stage in ("record", "dynamic_record") or
                 (self._live_classifier is not None and self._live_classifier.is_alive))
        )
        if should_poll and not self._camera_poll_scheduled:
            self._camera_poll_scheduled = True
            self.root.after(200, self._poll_camera)

    def _poll_camera(self) -> None:
        self._camera_poll_scheduled = False
        if self.current_page == "Camera":
            self._refresh_camera_widgets()
            self._refresh_engine_widgets()
            preview_var = self._camera_preview_var
            preview = self._camera_preview
        elif self.current_page == "Gesture Studio" and self._studio_preview is not None:
            if self._studio_stage == "record":
                self._refresh_studio_status()
            elif self._studio_stage == "dynamic_record":
                self._poll_dynamic_recording()
            elif self._live_classifier is not None and self._live_classifier.is_alive:
                self._refresh_live_status()
            preview_var = self._studio_preview_var
            preview = self._studio_preview
        else:
            return
        if preview_var is not None and preview_var.get():
            newest = self.camera_manager.get_latest_frame()
            if newest is not None and newest[0] != self._camera_last_sequence:
                self._camera_last_sequence = newest[0]
                if preview is not None:
                    engine_status = self.hand_engine.status
                    overlay_hands = engine_status.hands if engine_status.hands_detected else ()
                    preview.show_frame(newest[1], overlay_hands)
        self._schedule_camera_poll()

    def _poll_dynamic_recording(self) -> None:
        recorder = self._dynamic_recorder
        label = self._dynamic_status_label
        if recorder is None or label is None or not label.winfo_exists():
            return
        status = self.hand_engine.status
        if status.state is EngineState.ERROR:
            if recorder.recording:
                recorder.cancel()
            label.configure(text=status.error or "Camera or landmark engine failed; take cancelled.")
            self._refresh_dynamic_recording_progress()
            return
        if not recorder.recording:
            if status.state is EngineState.STOPPED:
                label.configure(text="Camera stopped; saved sequences are preserved.")
            return
        update = recorder.consume(status)
        self._refresh_dynamic_recording_progress()
        if update.example is not None:
            try:
                self.dynamic_store.save(recorder.session.dataset)
                label.configure(text=update.message + " Numeric landmarks saved; no video retained.")
            except (OSError, ValueError) as exc:
                label.configure(text=f"Could not save completed take: {exc}", bootstyle="danger")
                return
        elif update.message:
            label.configure(text=f"Recording active · {update.frame_count}/60 frames · {update.message}")

    def _refresh_studio_status(self) -> None:
        label = self._studio_status_label
        if label is None or not label.winfo_exists():
            return
        status = self.hand_engine.status
        expected = self._studio_session.dataset.hand_count if self._studio_session else 1
        if status.state is EngineState.ERROR:
            text = status.error or "Camera or landmark engine failed."
        elif status.state is EngineState.PAUSED:
            text = "Camera paused. Resume detection to record another example."
        elif status.state is EngineState.STOPPED:
            text = "Camera stopped. Start local camera to continue recording."
        elif status.state in (EngineState.STARTING, EngineState.STOPPING):
            text = "Loading local hand model and camera…" if not status.model_loaded else "Starting or stopping local camera…"
        elif status.hands_detected:
            text = f"{status.hand_count} hand(s) detected · {status.inference_latency_ms:.1f} ms inference · " + (
                "ready to record" if status.hand_count == expected else f"this gesture needs {expected} hand(s)")
        else:
            text = "No hand detected yet. Place the required hand configuration in the preview."
        label.configure(text=text)

    def _refresh_camera_widgets(self) -> None:
        label = self._camera_status_label
        if label is None or not label.winfo_exists():
            return
        status = self.camera_manager.status
        text = f"Camera {status.index}: {status.state.value.title()}" if status.index is not None else "Camera stopped"
        if status.frame_resolution:
            text += f" · {status.frame_resolution[0]} × {status.frame_resolution[1]} · {status.frames_captured} frames"
        if status.startup_ms is not None:
            text += f" · opened in {status.startup_ms:.0f} ms"
        if status.error:
            text += f" · {status.error}"
        label.configure(text=text)

    def _refresh_engine_widgets(self) -> None:
        label = self._engine_status_label
        if label is None or not label.winfo_exists():
            return
        status = self.hand_engine.status
        if status.state is EngineState.ERROR:
            text = status.error or "Hand landmark engine failed."
        elif status.state is EngineState.STARTING:
            text = "Loading local model or starting camera…" if not status.model_loaded else "Starting camera…"
        elif status.state is EngineState.PAUSED:
            text = f"Detection paused · {status.processed_frames} frames processed"
        elif status.state is EngineState.STOPPING:
            text = "Stopping detection and releasing local resources…"
        elif status.state is EngineState.STOPPED:
            text = "Detection stopped. The local model has not been loaded."
        elif status.hands_detected is None:
            text = "Waiting for the first local inference…"
        elif status.hands_detected:
            latency = status.inference_latency_ms
            text = f"{status.hand_count} hand(s) detected · {latency:.1f} ms inference · {status.processed_frames} frames processed"
        else:
            text = f"No hand detected · {status.inference_latency_ms:.1f} ms inference · {status.processed_frames} frames processed"
        label.configure(text=text)

    def _render_general_settings(self) -> None:
        appearance = ttk.Labelframe(
            self.page_body,
            text="Appearance",
            padding=20,
            bootstyle="secondary",
        )
        appearance.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        ttk.Label(
            appearance,
            text="Theme",
            font=("Segoe UI", 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            appearance,
            text="Choose a light or dark Nord theme. The choice is saved to the active profile and restored on startup.",
            wraplength=650,
        ).grid(row=1, column=0, sticky="w", pady=(4, 11))
        choices = ttk.Frame(appearance)
        choices.grid(row=2, column=0, sticky="w")
        for label, value in (("Light", "light"), ("Dark", "dark")):
            ttk.Radiobutton(
                choices,
                text=label,
                value=value,
                variable=self._theme_mode,
                command=self._apply_theme_choice,
                bootstyle="primary",
                takefocus=True,
            ).pack(side="left", padx=(0, 18))
        ttk.Label(
            appearance,
            textvariable=self._theme_notice,
            bootstyle="success",
        ).grid(row=3, column=0, sticky="w", pady=(9, 0))

        behavior = ttk.Labelframe(
            self.page_body,
            text="Window behavior",
            padding=20,
            bootstyle="secondary",
        )
        behavior.grid(row=1, column=0, sticky="ew")
        ttk.Label(
            behavior,
            text="Closing the window hides Gesture Manager to the notification area.",
            wraplength=650,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            behavior,
            text="Theme choice is saved with this profile and restored when it is active.",
            bootstyle="secondary",
            font=("Segoe UI", 9),
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))
        runtime = ttk.Labelframe(self.page_body, text="Pause behavior", padding=20,
                                 bootstyle="secondary")
        runtime.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        ttk.Label(runtime, text=("Automatic inactivity pause: after 5 minutes without meaningful camera-frame movement, "
                                 "the runtime pauses and releases the camera. Resume manually from the Dashboard or tray."),
                  bootstyle="info", wraplength=700).grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Checkbutton(
            runtime, text="Allow motion-only wake detection while recognition is paused",
            variable=self._wake_detection_var, command=self._update_wake_detection_setting,
            bootstyle="success",
        ).grid(row=1, column=0, sticky="w")
        ttk.Label(runtime, text=("Off by default: strict pause releases the camera. When enabled, the camera runs at "
                                 "320×240 and 2.5 FPS using frame-difference motion only; motion resumes local "
                                 "recognition. No microphone or hand classifier runs while paused."),
                  bootstyle="secondary", wraplength=700).grid(row=2, column=0, sticky="w", pady=(5, 0))

    def _apply_theme_choice(self) -> None:
        mode = self._validated_theme_mode(self._theme_mode.get())
        try:
            self._active_profile = self.profile_store.update_theme_mode(
                self._active_profile["id"], theme_mode=mode)
        except (OSError, ValueError) as exc:
            self._theme_mode.set(self.root.theme_mode)
            self._theme_notice.set(f"Could not save theme to this profile: {exc}")
            return
        self.root.theme_mode = mode
        apply_readable_text_styles(self.root.style, mode)
        apply_windows_titlebar_theme(self.root, mode)
        self._theme_notice.set(f"Theme saved to profile '{self._active_profile['name']}'.")
        self._refresh_theme_button()

    def _render_about(self) -> None:
        details = ttk.Labelframe(
            self.page_body,
            text="Application information",
            padding=20,
            bootstyle="secondary",
        )
        details.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self._diagnostic_row(details, 0, "Application", "Gesture Manager")
        self._diagnostic_row(details, 1, "Version", __version__)
        self._diagnostic_row(details, 2, "Python", platform.python_version())
        self._diagnostic_row(details, 3, "Operating system", platform.platform())
        self._diagnostic_row(details, 4, "Tray integration", self.tray_status())
        self._diagnostic_row(details, 5, "Runtime", self.controller.state.value.title())
        try:
            ui_version = version("ttkbootstrap")
        except PackageNotFoundError:
            ui_version = "Not installed"
        self._diagnostic_row(details, 6, "UI toolkit", f"ttkbootstrap {ui_version}")

        note = ttk.Labelframe(
            self.page_body,
            text="Diagnostics scope",
            padding=20,
            bootstyle="info",
        )
        note.grid(row=1, column=0, sticky="ew")
        ttk.Label(
            note,
            text=(
                "These diagnostics report local application and interpreter details. "
                "Camera availability, recognition metrics, and gesture results are "
                "not queried or fabricated."
            ),
            wraplength=690,
            justify="left",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            self.page_body,
            text=f"Python executable: {sys.executable}",
            bootstyle="secondary",
            wraplength=750,
            font=("Segoe UI", 9),
        ).grid(row=2, column=0, sticky="w", pady=(12, 0))

    def _diagnostic_row(self, parent: ttk.Frame, row: int, label: str, value: str) -> None:
        ttk.Label(parent, text=label, bootstyle="secondary").grid(
            row=row, column=0, sticky="w", padx=(0, 28), pady=5
        )
        value_label = ttk.Label(parent, text=value, wraplength=500)
        value_label.grid(row=row, column=1, sticky="w", pady=5)
        if label == "Runtime":
            self._diagnostic_state = value_label

    def _empty_page(self, title: str, detail: str, glyph: str) -> None:
        panel = ttk.Labelframe(
            self.page_body,
            text="Profiles",
            padding=(28, 28),
            bootstyle="secondary",
        )
        panel.grid(row=0, column=0, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text=glyph, font=("Segoe UI Symbol", 32), bootstyle="primary").grid(
            row=0, column=0, pady=(35, 10)
        )
        ttk.Label(panel, text=title, font=("Segoe UI", 16, "bold")).grid(
            row=1, column=0
        )
        ttk.Label(
            panel,
            text=detail,
            wraplength=540,
            justify="center",
        ).grid(row=2, column=0, pady=(8, 35))

