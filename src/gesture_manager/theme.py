"""Gesture Manager's restrained blue-and-slate light and dark themes."""

from __future__ import annotations

from ttkbootstrap.style import Style, ThemeDefinition


LIGHT_THEME = "gesture-light"
DARK_THEME = "gesture-dark"

_THEMES = (
    ThemeDefinition(
        LIGHT_THEME,
        mode="light",
        colors={
            "primary": "#2563EB",
            "secondary": "#E8EEF6",
            "success": "#16834A",
            "info": "#0785B5",
            "warning": "#B96A08",
            "danger": "#C9363E",
            "light": "#F1F5F9",
            "dark": "#16243A",
            "bg": "#F7F9FC",
            "fg": "#17243A",
            "selectbg": "#2563EB",
            "selectfg": "#FFFFFF",
            "border": "#D9E2EE",
            "inputfg": "#17243A",
            "inputbg": "#FFFFFF",
            "active": "#DFEAF8",
        },
    ),
    ThemeDefinition(
        DARK_THEME,
        mode="dark",
        colors={
            "primary": "#2563EB",
            "secondary": "#1A2A40",
            "success": "#37A66A",
            "info": "#38BDF8",
            "warning": "#F0AE4C",
            "danger": "#F07878",
            "light": "#26374E",
            "dark": "#091321",
            "bg": "#0D1726",
            "fg": "#E8EEF7",
            "selectbg": "#2563EB",
            "selectfg": "#FFFFFF",
            "border": "#2B3B51",
            "inputfg": "#E8EEF7",
            "inputbg": "#142338",
            "active": "#243A56",
        },
    ),
)


def install_brand_themes(style: Style) -> None:
    """Register the light/dark brand themes and make the light theme initial."""
    for theme in _THEMES:
        style.register_theme(theme)
    style.set_theme_modes(light=LIGHT_THEME, dark=DARK_THEME)
    style.theme_use(LIGHT_THEME)


def apply_readable_text_styles(style: Style, mode: str) -> None:
    """Keep secondary text readable while preserving secondary panel surfaces."""
    if mode == "dark":
        secondary_text = "#C7D2E0"
        inverse_secondary_text = "#E8EEF7"
    else:
        secondary_text = "#475569"
        inverse_secondary_text = "#334155"
    style.configure("secondary.TLabel", foreground=secondary_text)
    style.configure("secondary.TLabelframe.Label", foreground=secondary_text)
    style.configure("secondary.Inverse.TLabel", foreground=inverse_secondary_text)
    if mode == "dark":
        nav_bg, nav_fg, nav_hover = "#142338", "#E8EEF7", "#243A56"
    else:
        nav_bg, nav_fg, nav_hover = "#FFFFFF", "#334155", "#E8EEF6"
    style.configure(
        "GMNav.TButton", background=nav_bg, foreground=nav_fg,
        borderwidth=0, relief="flat", anchor="w", padding=(12, 10),
    )
    style.map(
        "GMNav.TButton",
        background=[("pressed", "#2563EB"), ("active", nav_hover)],
        foreground=[("pressed", "#FFFFFF"), ("active", nav_fg)],
    )
    style.configure(
        "GMNavActive.TButton", background="#2563EB", foreground="#FFFFFF",
        borderwidth=0, relief="flat", anchor="w", padding=(12, 10),
    )
    style.map("GMNavActive.TButton", background=[("active", "#1D4ED8")])
    if mode == "dark":
        button_bg, button_fg, button_hover = "#243A56", "#F8FAFC", "#304B6D"
    else:
        button_bg, button_fg, button_hover = "#E8EEF6", "#17243A", "#D8E4F2"
    style.configure(
        "GMReadable.TButton", background=button_bg, foreground=button_fg,
        borderwidth=1, relief="solid", padding=(11, 7),
    )
    style.map(
        "GMReadable.TButton",
        background=[("disabled", button_bg), ("pressed", "#1D4ED8"), ("active", button_hover)],
        foreground=[("disabled", "#64748B" if mode == "light" else "#A9B6C8"),
                    ("pressed", "#FFFFFF"), ("active", button_fg)],
    )
    style.configure(
        "GMPauseControl.TButton", background="#F0AE4C", foreground="#17243A",
        borderwidth=1, relief="solid", padding=(10, 6),
    )
    style.map(
        "GMPauseControl.TButton",
        background=[("disabled", "#D6C7AE"), ("pressed", "#D9952F"), ("active", "#F5BF68")],
        foreground=[("disabled", "#475569"), ("pressed", "#17243A"), ("active", "#17243A")],
    )
    surface = "#0D1726" if mode == "dark" else "#F7F9FC"
    disabled_fg = "#A9B6C8" if mode == "dark" else "#64748B"
    disabled_border = "#64748B" if mode == "dark" else "#94A3B8"
    if mode == "dark":
        outlines = {
            "primary": ("#93C5FD", "#2563EB", "#FFFFFF"),
            "secondary": ("#D8E1ED", "#52657E", "#FFFFFF"),
            "success": ("#86EFAC", "#16834A", "#FFFFFF"),
            "info": ("#7DD3FC", "#0785B5", "#FFFFFF"),
            "warning": ("#F8C66D", "#D08718", "#17243A"),
            "danger": ("#FDA4AF", "#C9363E", "#FFFFFF"),
        }
    else:
        outlines = {
            "primary": ("#1D4ED8", "#2563EB", "#FFFFFF"),
            "secondary": ("#334155", "#64748B", "#FFFFFF"),
            "success": ("#126B3C", "#16834A", "#FFFFFF"),
            "info": ("#006B91", "#0785B5", "#FFFFFF"),
            "warning": ("#874B00", "#B96A08", "#FFFFFF"),
            "danger": ("#B4232C", "#C9363E", "#FFFFFF"),
        }
    for name, (foreground, accent, hover_foreground) in outlines.items():
        ttk_style = f"{name}.Outline.TButton"
        style.configure(
            ttk_style, foreground=foreground, background=surface,
            bordercolor=accent, lightcolor=surface, darkcolor=surface,
            borderwidth=1, relief="solid", focuscolor=accent,
        )
        style.map(
            ttk_style,
            foreground=[("disabled", disabled_fg), ("pressed !disabled", hover_foreground),
                        ("hover !disabled", hover_foreground)],
            background=[("pressed !disabled", accent), ("hover !disabled", accent)],
            bordercolor=[("disabled", disabled_border), ("pressed !disabled", accent),
                         ("hover !disabled", accent)],
            focuscolor=[("pressed !disabled", hover_foreground), ("hover !disabled", hover_foreground)],
            lightcolor=[("pressed !disabled", accent), ("hover !disabled", accent)],
            darkcolor=[("pressed !disabled", accent), ("hover !disabled", accent)],
        )
