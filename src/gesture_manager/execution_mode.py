"""Per-profile execution-mode preference storage.

The preference records the user's last choice. It never authorizes real input;
the application must still start in dry-run and obtain explicit confirmation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


class ExecutionModeStore:
    FORMAT = "gesture-manager-execution-mode"
    VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load_dry_run(self) -> bool:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return True
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Execution mode setting could not be read: {exc}") from exc
        if (not isinstance(data, dict)
                or set(data) != {"format", "version", "dry_run"}
                or data["format"] != self.FORMAT
                or type(data["version"]) is not int
                or data["version"] != self.VERSION
                or type(data["dry_run"]) is not bool):
            raise ValueError("Execution mode setting has an unsupported format")
        return data["dry_run"]

    def save_dry_run(self, dry_run: bool) -> None:
        if type(dry_run) is not bool:
            raise ValueError("dry_run must be a boolean")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"format": self.FORMAT, "version": self.VERSION,
                              "dry_run": dry_run}, indent=2) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=".execution-mode-", suffix=".tmp",
                                         dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
