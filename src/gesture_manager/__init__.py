"""Gesture Manager desktop application and packaged version metadata."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def _application_version() -> str:
    # PyInstaller builds include a version file generated from APP_VERSION in
    # build_release.py. In source checkouts, use the installed project metadata.
    if getattr(sys, "frozen", False):
        version_file = Path(getattr(sys, "_MEIPASS")) / "release_version.json"
        try:
            version_data = json.loads(version_file.read_text(encoding="utf-8"))
            value = version_data["version"]
            if isinstance(value, str) and value:
                return value
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
            return "0+unknown"

    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("gesture-manager")
    except (ImportError, PackageNotFoundError):
        return "0+unknown"


__version__ = _application_version()

