## 1. Gesture Manager

Gesture Manager is a Windows app that lets you create your own hand gestures to control your PC with keyboard shortcuts.

## 2. Features

- Create custom hand gestures by recording examples with a camera.
- Train and test gesture recognition on your PC.
- Connect gestures to keyboard shortcuts, with Dry-run enabled by default.
- Pause, resume, or stop recognition from the app or system tray.
- Save gesture sets and settings in profiles.
- Use the camera preview and hand landmark display to check detection.

## 3. Download

1. Open the [latest GitHub release](https://github.com/marslinoED/Geasture-Manager/releases/latest).
2. In the release's **Assets** section, download `GestureManager-Portable-0.1.0.zip` (or the portable ZIP with the newest version number). Do not download **Source code**.
3. Extract the ZIP, open the extracted folder, and run `GestureManager.exe`.

If a release is listed, its portable ZIP will be in **Assets**. The project files here do not confirm whether a public release is currently available.

## 4. Development setup

The development workflow requires Windows 10 or later and Python 3.13 (64-bit). Python 3.14 and older Python versions are not supported by this project’s package metadata.

Open PowerShell in the repository folder and run:

```powershell
# Create and activate an isolated environment
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade packaging tools and install the app in editable mode
python -m pip install --upgrade pip setuptools
python -m pip install --editable .
```

The editable install downloads every runtime package declared by the project, including Pillow, OpenCV, ai-edge-litert, pynput, pystray, scikit-learn, and ttkbootstrap. It also installs the `gesture-manager` command and makes source changes immediately available.

To install the optional release/build tools as well, use the build extra:

```powershell
python -m pip install --editable ".[build]"
```

Run the application from the activated environment with either command:

```powershell
python main.py
# or
gesture-manager
```

Run the test suite with:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

To build a portable release after installing the build extra:

```powershell
python build_release.py
```

The hand-landmark model files in `models/` are part of the repository and are used by the app at runtime; no separate model download is needed.

### Publishing a GitHub release

Push a semantic-version tag to GitHub. The included GitHub Actions workflow builds the Windows portable ZIP and attaches it to a public GitHub Release automatically:

```powershell
git tag v0.1.0
git push origin v0.1.0
```

After the workflow finishes, anyone can download the ZIP from the repository’s **Releases** page. The workflow can also be started manually from the **Actions** tab by entering a version such as `0.1.1`.

## 5. How to Use

Run `GestureManager.exe`. The app starts in the system tray near the clock; select its icon to open Gesture Manager. Create gestures in **My Gestures** or **Gesture Studio**, then connect them to shortcuts on the **Actions** page. Use the Dashboard or tray menu to start, pause, resume, or stop recognition. Real keyboard input requires a separate confirmation in the app.

## 6. Requirements

- Windows 10 or later, 64-bit.
- A camera to record and recognize hand gestures.
- No separate Python installation is needed for the portable release.
