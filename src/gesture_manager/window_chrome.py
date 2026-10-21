"""Native Windows title-bar colors that follow the selected app theme."""

from __future__ import annotations

import logging
import sys
import tkinter as tk

logger = logging.getLogger(__name__)


def _colorref(value: str) -> int:
    """Convert ``#RRGGBB`` to the Windows COLORREF layout (0x00BBGGRR)."""
    value = value.removeprefix("#")
    if len(value) != 6:
        raise ValueError("Expected a six-digit RGB color")
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return red | (green << 8) | (blue << 16)


def apply_windows_titlebar_theme(root: tk.Misc, mode: str) -> bool:
    """Apply dark/light and accent colors to a Windows 11 native title bar.

    Tk does not expose non-client title-bar colors. DWM attributes are used on
    Windows; other platforms keep their normal native window decorations.
    """
    if sys.platform != "win32" or mode not in ("light", "dark"):
        return False
    try:
        import ctypes

        dwmapi = ctypes.WinDLL("dwmapi")
        set_attribute = dwmapi.DwmSetWindowAttribute
        set_attribute.argtypes = (ctypes.c_void_p, ctypes.c_uint,
                                  ctypes.c_void_p, ctypes.c_uint)
        set_attribute.restype = ctypes.c_long
        hwnd = ctypes.c_void_p(root.winfo_id())

        def set_value(attribute: int, value: int) -> bool:
            stored = ctypes.c_uint(value)
            return set_attribute(hwnd, attribute, ctypes.byref(stored),
                                 ctypes.sizeof(stored)) == 0

        dark = int(mode == "dark")
        # Windows 11 uses 20; 19 is the earlier Windows 10 attribute ID.
        dark_applied = set_value(20, dark) or set_value(19, dark)
        if mode == "dark":
            caption, text, border = "#0D1726", "#E8EEF7", "#2B3B51"
        else:
            caption, text, border = "#F7F9FC", "#17243A", "#D9E2EE"
        colors_applied = (
            set_value(35, _colorref(caption))
            and set_value(36, _colorref(text))
            and set_value(34, _colorref(border))
        )
        return dark_applied or colors_applied
    except Exception:
        logger.debug("Native title-bar theme is unavailable", exc_info=True)
        return False
