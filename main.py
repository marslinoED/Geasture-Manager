"""Convenience entry point for running Gesture Manager from a checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gesture_manager.app import main  # noqa: E402


if __name__ == "__main__":
    main()

