import unittest
from unittest.mock import patch

from gesture_manager.theme import apply_readable_text_styles
from gesture_manager.window_chrome import _colorref, apply_windows_titlebar_theme


class FakeStyle:
    def __init__(self):
        self.configured = {}
        self.mapped = {}

    def configure(self, name, **values):
        self.configured[name] = values

    def map(self, name, **values):
        self.mapped[name] = values


class ThemeAndChromeTests(unittest.TestCase):
    def test_outline_buttons_have_readable_text_and_borders_in_both_themes(self):
        for mode, expected in (("light", "#334155"), ("dark", "#D8E1ED")):
            style = FakeStyle()
            apply_readable_text_styles(style, mode)
            secondary = style.configured["secondary.Outline.TButton"]
            self.assertEqual(secondary["foreground"], expected)
            self.assertEqual(secondary["borderwidth"], 1)
            self.assertEqual(secondary["relief"], "solid")
            self.assertIn("disabled", str(style.mapped["secondary.Outline.TButton"]))

    def test_colorref_uses_windows_bgr_order(self):
        self.assertEqual(_colorref("#123456"), 0x00563412)

    @patch("gesture_manager.window_chrome.sys.platform", "linux")
    def test_non_windows_titlebar_keeps_native_decoration(self):
        self.assertFalse(apply_windows_titlebar_theme(object(), "dark"))


if __name__ == "__main__":
    unittest.main()
