from __future__ import annotations

import unittest
from PIL import Image

from gesture_manager.ui import PAGES, dashboard_column_count
from gesture_manager.assets import app_icon_ico_path, load_app_icon
from gesture_manager.theme import (
    DARK_THEME, LIGHT_THEME, _THEMES, apply_readable_text_styles,
)


class InterfaceLayoutTests(unittest.TestCase):
    def test_dashboard_cards_adapt_to_window_width(self) -> None:
        self.assertEqual(dashboard_column_count(800), 1)
        self.assertEqual(dashboard_column_count(1024), 2)
        self.assertEqual(dashboard_column_count(1180), 3)

    def test_navigation_entries_have_unique_names_and_real_glyphs(self) -> None:
        names = [name for name, _glyph in PAGES]
        glyphs = [glyph for _name, glyph in PAGES]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(glyph and not glyph.startswith("\\u") for glyph in glyphs))

    def test_brand_icon_is_transparent_and_square(self) -> None:
        icon = load_app_icon()
        self.assertEqual(icon.mode, "RGBA")
        self.assertEqual(icon.width, icon.height)
        self.assertEqual(icon.getpixel((0, 0))[3], 0)

    def test_windows_icon_asset_exists(self) -> None:
        path = app_icon_ico_path()
        self.assertTrue(path.is_file())
        with Image.open(path) as icon:
            self.assertEqual(icon.format, "ICO")
            self.assertEqual(icon.size, (256, 256))

    def test_secondary_text_styles_have_contrast_in_both_modes(self) -> None:
        class StyleRecorder:
            def __init__(self):
                self.configured = {}

            def configure(self, name, **options):
                self.configured[name] = options

            def map(self, _name, **_options):
                pass

        style = StyleRecorder()
        apply_readable_text_styles(style, "light")
        self.assertEqual(style.configured["secondary.TLabel"]["foreground"], "#475569")
        self.assertEqual(style.configured["secondary.Inverse.TLabel"]["foreground"], "#334155")
        apply_readable_text_styles(style, "dark")
        self.assertEqual(style.configured["secondary.TLabel"]["foreground"], "#C7D2E0")
        self.assertEqual(style.configured["secondary.Inverse.TLabel"]["foreground"], "#E8EEF7")
        self.assertEqual(style.configured["GMNav.TButton"]["foreground"], "#E8EEF7")

    def test_light_and_dark_brand_themes_share_the_blue_accent(self) -> None:
        themes = {theme.name: theme for theme in _THEMES}
        self.assertEqual(set(themes), {LIGHT_THEME, DARK_THEME})
        self.assertEqual(themes[LIGHT_THEME].colors.primary, "#2563EB")
        self.assertEqual(themes[DARK_THEME].colors.primary, "#2563EB")
        self.assertNotEqual(themes[LIGHT_THEME].colors.bg, themes[DARK_THEME].colors.bg)


if __name__ == "__main__":
    unittest.main()
