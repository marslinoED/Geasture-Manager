"""Packaged visual assets used by the desktop window and notification tray."""

from __future__ import annotations

from importlib.resources import files
from io import BytesIO
from pathlib import Path
import sys

from PIL import Image


def load_app_icon() -> Image.Image:
    """Load the bundled app mark without relying on the current working directory."""
    icon_bytes = files("gesture_manager").joinpath("assets", "app_icon.png").read_bytes()
    with Image.open(BytesIO(icon_bytes)) as source:
        return source.convert("RGBA")


def app_icon_ico_path() -> Path:
    """Return the bundled Windows icon path for the native title bar/taskbar."""
    return Path(__file__).resolve().parent / "assets" / "app_icon.ico"


def model_assets_directory() -> Path:
    """Find bundled inference assets without depending on the working directory."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "models"
    return Path(__file__).resolve().parents[2] / "models"
