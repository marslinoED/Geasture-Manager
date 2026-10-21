"""Small status-tinted variants of the supplied app mark for the system tray."""

from __future__ import annotations

from PIL import Image, ImageDraw


STATUS_COLORS: dict[str, tuple[int, int, int]] = {
    "active": (34, 197, 94),
    "paused": (245, 190, 45),
    "unpaused": (239, 68, 68),
    "stopped": (148, 163, 184),
}
_HUES = {"active": 85, "paused": 42, "unpaused": 0, "stopped": None}


def create_status_icons(mark: Image.Image, size: int = 64) -> dict[str, Image.Image]:
    """Hue-filter the logo's blue accents and add a clear state dot."""
    source = mark.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    hsv = source.convert("HSV")
    pixels = list(hsv.get_flattened_data())
    variants: dict[str, Image.Image] = {}
    for status, target_hue in _HUES.items():
        filtered = []
        for hue, saturation, value in pixels:
            # Recolor the logo's blue/cyan brand accents while preserving the
            # white hand, outlines, dark background, and antialiased alpha.
            if saturation > 55 and value > 50 and 110 <= hue <= 185:
                if target_hue is None:
                    filtered.append((hue, 0, value))
                else:
                    filtered.append((target_hue, saturation, value))
            else:
                filtered.append((hue, saturation, value))
        colored = Image.new("HSV", source.size)
        colored.putdata(filtered)
        icon = colored.convert("RGBA")
        alpha = source.getchannel("A")
        icon.putalpha(alpha)

        # A compact indicator remains visible at tray sizes even when Windows
        # displays a monochrome or low-resolution version of the image.
        draw = ImageDraw.Draw(icon)
        radius = max(5, size // 9)
        center = size - radius - 2
        bounds = (center - radius, center - radius, center + radius, center + radius)
        draw.ellipse(bounds, fill=(8, 16, 28, 255), outline=(255, 255, 255, 255), width=2)
        inner = (center - radius + 2, center - radius + 2,
                 center + radius - 2, center + radius - 2)
        draw.ellipse(inner, fill=(*STATUS_COLORS[status], 255))
        variants[status] = icon
    return variants
