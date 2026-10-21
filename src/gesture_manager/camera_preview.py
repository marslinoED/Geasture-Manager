"""Tk preview widget and pure OpenCV-frame conversion helpers."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageTk
import ttkbootstrap as ttk

from gesture_manager.hand_landmarks import DetectedHand


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def draw_hand_landmarks(frame: object, hands: tuple[DetectedHand, ...]) -> np.ndarray:
    """Return a copy of a BGR frame with normalized hand skeletons overlaid."""
    image = np.asarray(frame).copy()
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Landmark preview requires a BGR H×W×3 frame")
    height, width = image.shape[:2]
    for hand in hands:
        if len(hand.landmarks) != 21:
            continue
        points = [
            (round(point.x * width), round(point.y * height))
            for point in hand.landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(image, points[start], points[end], (55, 225, 125), 2, cv2.LINE_AA)
        for x, y in points:
            cv2.circle(image, (x, y), 4, (35, 245, 235), -1, cv2.LINE_AA)
    return image


def frame_to_pil(frame: object, max_size: tuple[int, int] = (640, 480)) -> Image.Image:
    """Convert an OpenCV BGR/BGRA/gray array to a bounded RGB Pillow image."""
    if len(max_size) != 2 or min(max_size) <= 0:
        raise ValueError("max_size must contain positive width and height")
    if not hasattr(frame, "shape") or len(frame.shape) not in (2, 3):
        raise ValueError("frame must be a grayscale, BGR, or BGRA image")
    if len(frame.shape) == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 3:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif frame.shape[2] == 4:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError("frame must have 1, 3, or 4 channels")
    image = Image.fromarray(rgb)
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


class CameraPreview(ttk.Frame):
    """Display frames supplied by the caller on Tk's main thread."""

    def __init__(self, master: object, **kwargs: object) -> None:
        super().__init__(master, **kwargs)
        self._photo: ImageTk.PhotoImage | None = None
        self._placeholder = ttk.Label(
            self,
            text="Preview is off",
            anchor="center",
            padding=(20, 70),
            bootstyle="secondary",
        )
        self._placeholder.pack(fill="both", expand=True)
        self._image_label: ttk.Label | None = None

    def show_frame(self, frame: object, hands: tuple[DetectedHand, ...] = ()) -> None:
        """Render one frame. Call only from the Tk main thread."""
        image = frame_to_pil(draw_hand_landmarks(frame, hands) if hands else frame)
        self._photo = ImageTk.PhotoImage(image=image, master=self)
        self._placeholder.pack_forget()
        if self._image_label is None or not self._image_label.winfo_exists():
            self._image_label = ttk.Label(self, image=self._photo, anchor="center")
            self._image_label.pack(fill="both", expand=True)
        else:
            self._image_label.configure(image=self._photo)

    def clear(self, message: str = "Preview is off") -> None:
        """Drop the displayed image and show a status message."""
        if self._image_label is not None:
            self._image_label.destroy()
            self._image_label = None
        self._photo = None
        self._placeholder.configure(text=message)
        self._placeholder.pack(fill="both", expand=True)
