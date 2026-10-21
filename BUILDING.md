# Building Gesture Manager for Windows

Gesture Manager is distributed as a portable, one-folder Windows application built by PyInstaller. The release contains `GestureManager.exe`, its runtime files, visual assets, and local hand landmark models. No installer builder is used.

The app stores profiles, settings, gesture datasets, and trained models under `%LOCALAPPDATA%\GestureManager`. They remain separate from the portable release and are not included in its ZIP.

## Prerequisites

- Windows 10 or later, 64-bit.
- Python 3.13 64-bit with Tkinter available.
- The project's runtime dependencies and local inference models. Model sources and licenses are documented in [models/README.md](models/README.md).

Create the virtual environment and install the runtime and pinned build dependency:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[build]"
```

The `build` extra pins PyInstaller 6.20.0. ZIP creation uses Python's standard library.

## Build a release

After changing application code:

1. Open `build_release.py`.
2. Change `APP_VERSION = "0.1.0"` to the desired semantic version, for example `APP_VERSION = "0.1.1"`.
3. Run `build_release.bat`.
4. Find the portable folder and ZIP in `releases/<version>/`.
5. Extract the ZIP and launch `GestureManager.exe`.

The batch file selects `.venv\Scripts\python.exe` when available, otherwise `py -3.13`, then `python`. The equivalent direct command is:

```powershell
python build_release.py
```

The script validates the version, files, and dependencies; synchronizes `pyproject.toml` from `APP_VERSION`; builds the PyInstaller one-folder app; copies it into a staging release; writes `README.txt` and `release.json`; creates the ZIP; and checks the archive for required files and corruption. It stops on the first error and returns a failure code.

Expected output:

```text
releases/
└── 0.1.0/
    ├── GestureManager.exe
    ├── _internal/
    │   ├── gesture_manager/
    │   ├── models/
    │   └── ...
    ├── README.txt
    ├── release.json
    └── GestureManager-Portable-0.1.0.zip
```

The ZIP contains the same portable files under a `GestureManager-<version>/` folder. It does not contain a recursive copy of itself.

## Rebuilding a version

Existing releases are never removed. If `releases/<version>/` already exists, the build moves that complete directory to `releases/_previous/<version>-<UTC timestamp>/` before placing the new release. Only named temporary build directories beneath `build/` are cleaned. Source code, local application data, and older release contents are preserved.

## Test the portable build

Test on the target Windows machine:

1. Extract the full ZIP to a writable folder. Keep `GestureManager.exe` beside `_internal`.
2. Launch `GestureManager.exe`; verify it starts in the notification area without opening a console.
3. Open the window from the tray and navigate the settings pages.
4. Verify camera detection and the bundled hand landmark model, then stop detection and confirm the camera releases.
5. Verify pause, resume, Stop Service, and application Exit behavior.
6. Confirm profiles and settings are saved under `%LOCALAPPDATA%\GestureManager` and remain separate from the extracted application.
7. In About & Diagnostics, confirm the displayed version matches `release.json`.

The build script verifies the ZIP member list and archive integrity. That check does not prove the app runs on a clean Windows installation. Test the portable copy on a machine with no project checkout or Python development environment before distributing it. Automated shortcut tests must continue to use a mock backend.

## Troubleshooting

- **PyInstaller missing or has the wrong version:** run the build dependency installation command above in the same virtual environment.
- **A runtime dependency is missing:** install the project with `python -m pip install -e ".[build]"` and rerun the build.
- **A model file is missing:** restore `models/hand_detector.tflite` and `models/hand_landmarks_detector.tflite`. The build does not download models.
- **Tkinter is unavailable:** install 64-bit Python 3.13 with Tcl/Tk support and recreate `.venv`.
- **An existing release path cannot be moved:** close Explorer windows or processes using files under that release, then retry. The build preserves the old release and will not overwrite it in place.
- **Camera model fails to load:** verify `_internal/models/` exists beside the executable and contains both `.tflite` files.

This workflow does not add code signing or automatic updates.
