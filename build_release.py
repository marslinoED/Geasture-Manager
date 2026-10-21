"""Build a self-contained Windows portable release with PyInstaller."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Sequence
from zipfile import ZIP_DEFLATED, ZipFile


APP_VERSION = "0.1.0"
APP_NAME = "Gesture Manager"
ROOT = Path(__file__).resolve().parent
BUILD_DIR = ROOT / "build"
RELEASES_DIR = ROOT / "releases"
APP_ENTRY = ROOT / "main.py"
ICON = ROOT / "src" / "gesture_manager" / "assets" / "app_icon.ico"
PYPROJECT = ROOT / "pyproject.toml"
REQUIRED_MODEL_FILES = (
    ROOT / "models" / "hand_detector.tflite",
    ROOT / "models" / "hand_landmarks_detector.tflite",
)
GENERATED_BUILD_DIRS = frozenset({
    "pyinstaller-dist", "pyinstaller-work", "version", "release-staging",
})
REQUIRED_RUNTIME_MODULES = (
    "PIL", "ai_edge_litert", "cv2", "pynput", "pystray", "sklearn", "ttkbootstrap",
)
PYINSTALLER_VERSION = "6.20.0"
PORTABLE_ZIP_NAME = f"GestureManager-Portable-{APP_VERSION}.zip"
REQUIRED_PORTABLE_FILES = (
    "GestureManager.exe",
    "README.txt",
    "release.json",
    "_internal/release_version.json",
    "_internal/models/hand_detector.tflite",
    "_internal/models/hand_landmarks_detector.tflite",
    "_internal/gesture_manager/assets/app_icon.ico",
    "_internal/gesture_manager/assets/app_icon.png",
)


class ReleaseError(RuntimeError):
    """A release step failed with a user-actionable message."""


def parse_version(value: str) -> tuple[int, int, int, int]:
    """Validate semantic version syntax and return Windows' numeric tuple."""
    match = re.fullmatch(
        r"(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\."
        r"(?P<patch>0|[1-9]\d*)"
        r"(?:-(?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*)"
        r"(?:\.(?:0|[1-9]\d*|[0-9A-Za-z-]*[A-Za-z-][0-9A-Za-z-]*))*)?"
        r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?", value,
    )
    if match is None:
        raise ReleaseError(f"APP_VERSION is not a semantic version: {value!r}")
    parts = tuple(int(match.group(key)) for key in ("major", "minor", "patch"))
    if any(part > 65535 for part in parts):
        raise ReleaseError("Windows executable version components must not exceed 65535")
    return (*parts, 0)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def clean_generated_build_dir(name: str) -> Path:
    """Clear only a named disposable child of build/ and reject path escapes."""
    if name not in GENERATED_BUILD_DIRS:
        raise ReleaseError(f"Refusing to clean unrecognized build path: {name}")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    target = BUILD_DIR / name
    if not _inside(target, BUILD_DIR) or target.resolve() == BUILD_DIR.resolve():
        raise ReleaseError(f"Refusing unsafe cleanup path: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    return target


def _set_project_version(version: str) -> None:
    """Keep setuptools metadata synchronized from this script's one version."""
    original = PYPROJECT.read_text(encoding="utf-8")
    updated, count = re.subn(
        r'(?m)^(version\s*=\s*)"[^"]+"\s*$',
        lambda match: f'{match.group(1)}"{version}"', original, count=1,
    )
    if count != 1:
        raise ReleaseError("Could not locate exactly one [project] version in pyproject.toml")
    PYPROJECT.write_text(updated, encoding="utf-8", newline="\n")


def _write_build_metadata(version: str, version_tuple: tuple[int, int, int, int]) -> Path:
    version_dir = clean_generated_build_dir("version")
    (version_dir / "release_version.json").write_text(
        json.dumps({"name": APP_NAME, "version": version}, indent=2) + "\n",
        encoding="utf-8",
    )
    major, minor, patch, build = version_tuple
    version_text = f"{major}, {minor}, {patch}, {build}"
    (version_dir / "windows_version_info.txt").write_text(
        "VSVersionInfo(\n"
        f"  ffi=FixedFileInfo(filevers=({version_text}), prodvers=({version_text}), "
        "mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),\n"
        "  kids=[StringFileInfo([StringTable('040904B0', [\n"
        f"    StringStruct('CompanyName', '{APP_NAME}'),\n"
        f"    StringStruct('FileDescription', '{APP_NAME} desktop application'),\n"
        f"    StringStruct('FileVersion', '{version}'),\n"
        "    StringStruct('InternalName', 'GestureManager'),\n"
        "    StringStruct('OriginalFilename', 'GestureManager.exe'),\n"
        f"    StringStruct('ProductName', '{APP_NAME}'),\n"
        f"    StringStruct('ProductVersion', '{version}'),\n"
        "  ])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])]\n"
        ")\n",
        encoding="utf-8",
    )
    return version_dir


def _run(command: Sequence[str], step: str) -> None:
    print(f"\n[{step}] {' '.join(command)}", flush=True)
    try:
        subprocess.run(list(command), cwd=ROOT, check=True)
    except FileNotFoundError as exc:
        raise ReleaseError(f"{step} could not start: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise ReleaseError(f"{step} failed with exit code {exc.returncode}") from exc


def _pyinstaller_command(version_dir: Path, dist_dir: Path, work_dir: Path) -> list[str]:
    return [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir",
        "--windowed", "--name", "GestureManager", "--paths", str(ROOT / "src"),
        "--distpath", str(dist_dir), "--workpath", str(work_dir),
        "--specpath", str(work_dir), "--icon", str(ICON),
        "--version-file", str(version_dir / "windows_version_info.txt"),
        "--add-data", f"{ROOT / 'src' / 'gesture_manager' / 'assets'};gesture_manager/assets",
        "--add-data", f"{REQUIRED_MODEL_FILES[0]};models",
        "--add-data", f"{REQUIRED_MODEL_FILES[1]};models",
        "--add-data", f"{version_dir / 'release_version.json'};.",
        "--hidden-import", "pystray._win32",
        "--hidden-import", "pynput.keyboard._win32",
        # scipy's array-api compatibility package loads these modules with
        # __import__, which PyInstaller's static analysis can miss.
        "--hidden-import", "scipy._external.array_api_compat.numpy.linalg",
        "--hidden-import", "scipy._external.array_api_compat.numpy.fft",
        "--collect-all", "ai_edge_litert",
        "--collect-all", "ttkbootstrap",
        "--copy-metadata", "ttkbootstrap",
        "--collect-data", "PIL",
        str(APP_ENTRY),
    ]


def _validate_packaged_app(packaged_app: Path) -> None:
    required = (
        packaged_app / "GestureManager.exe",
        packaged_app / "_internal" / "release_version.json",
        packaged_app / "_internal" / "gesture_manager" / "assets" / "app_icon.ico",
        packaged_app / "_internal" / "gesture_manager" / "assets" / "app_icon.png",
        *(packaged_app / "_internal" / "models" / path.name for path in REQUIRED_MODEL_FILES),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ReleaseError("Packaged application is missing required files: " + ", ".join(missing))


def _write_portable_readme(destination: Path, version: str) -> None:
    destination.write_text(
        f"""{APP_NAME} {version} - Portable Windows build

START
1. Extract the complete ZIP archive to a folder you can write to.
2. Open the extracted GestureManager-{version} folder.
3. Double-click GestureManager.exe. No separate Python installation is needed.
4. The app starts in the Windows notification area. Open its tray icon to show
   the settings window, or use the tray menu to pause, resume, stop, or exit.

DATA AND SETTINGS
Gesture Manager keeps profiles, preferences, gesture datasets, and trained
models in %LOCALAPPDATA%\\GestureManager. Those files are separate from this
portable application folder and are not included in this release archive.

CAMERA AND SHORTCUTS
Camera processing and hand landmark recognition run locally. Camera access is
needed only while the camera or recognition service is running. Shortcut input
starts in Dry-run mode; real keyboard input requires explicit confirmation in
the app. The app does not require administrator privileges.

TROUBLESHOOTING
- If Windows shows a security prompt, verify that you trust the source before
  choosing to run the app.
- If the app cannot start, extract the entire archive again; do not move the
  executable away from its _internal folder.
- For camera problems, close other apps using the camera and retry from the
  Camera page.

Version: {version}
""",
        encoding="utf-8",
        newline="\n",
    )


def _write_release_metadata(stage: Path, version: str, built_at: str) -> dict[str, object]:
    portable_files = sorted(
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file()
    )
    portable_files.append("release.json")
    payload: dict[str, object] = {
        "application": APP_NAME,
        "version": version,
        "build_date_utc": built_at,
        "artifacts": ["GestureManager.exe", PORTABLE_ZIP_NAME],
        "portable_files": sorted(set(portable_files)),
    }
    (stage / "release.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8",
    )
    return payload


def _create_portable_zip(stage: Path, version: str) -> Path:
    archive = stage / f"GestureManager-Portable-{version}.zip"
    archive_prefix = f"GestureManager-{version}"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as output:
        for path in sorted(stage.rglob("*")):
            if not path.is_file() or path == archive:
                continue
            archive_name = (Path(archive_prefix) / path.relative_to(stage)).as_posix()
            output.write(path, archive_name)
    return archive


def _validate_portable_release(stage: Path, version: str, archive: Path) -> None:
    manifest_path = stage / "release.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"Portable release metadata could not be read: {exc}") from exc
    if (not isinstance(manifest, dict) or manifest.get("application") != APP_NAME
            or manifest.get("version") != version
            or archive.name not in manifest.get("artifacts", ())):
        raise ReleaseError("Portable release metadata is missing or inconsistent")

    expected_files = {
        path.relative_to(stage).as_posix()
        for path in stage.rglob("*")
        if path.is_file() and path != archive
    }
    missing = sorted(set(REQUIRED_PORTABLE_FILES) - expected_files)
    if missing:
        raise ReleaseError("Portable release is missing required files: " + ", ".join(missing))

    prefix = f"GestureManager-{version}/"
    try:
        with ZipFile(archive) as zipped:
            corrupt_member = zipped.testzip()
            if corrupt_member is not None:
                raise ReleaseError(f"Portable ZIP has a corrupt member: {corrupt_member}")
            names = set(zipped.namelist())
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"Portable ZIP could not be validated: {exc}") from exc
    actual_files = {name.removeprefix(prefix) for name in names if name.startswith(prefix)}
    if (actual_files != expected_files or len(names) != len(actual_files)
            or set(manifest.get("portable_files", ())) != expected_files):
        missing_from_zip = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ReleaseError(
            "ZIP contents do not match the portable folder. "
            f"Missing: {missing_from_zip[:5]}; unexpected: {unexpected[:5]}"
        )


def _preserve_existing_release(release_dir: Path) -> None:
    """Move the prior release into a retained archive directory, never delete it."""
    if not release_dir.exists():
        return
    backup_root = RELEASES_DIR / "_previous"
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = backup_root / f"{release_dir.name}-{stamp}"
    suffix = 1
    while backup.exists():
        backup = backup_root / f"{release_dir.name}-{stamp}-{suffix}"
        suffix += 1
    try:
        release_dir.rename(backup)
    except OSError as exc:
        raise ReleaseError(f"Could not preserve the previous release at {backup}: {exc}") from exc
    print(f"Preserved previous release at {backup}")


def build() -> Path:
    if os.name != "nt":
        raise ReleaseError("The Windows release build must run on Windows.")
    if sys.version_info[:2] != (3, 13) or sys.maxsize <= 2**32:
        raise ReleaseError("Use 64-bit Python 3.13 to build the Windows application.")
    version_tuple = parse_version(APP_VERSION)
    required_files = (APP_ENTRY, ICON, PYPROJECT, *REQUIRED_MODEL_FILES)
    missing = [path.relative_to(ROOT).as_posix() for path in required_files if not path.is_file()]
    if missing:
        raise ReleaseError("Required project files are missing: " + ", ".join(missing))
    missing_modules = [name for name in REQUIRED_RUNTIME_MODULES
                       if importlib.util.find_spec(name) is None]
    if missing_modules:
        raise ReleaseError("Runtime dependencies missing from the build interpreter: "
                           + ", ".join(missing_modules))
    if importlib.util.find_spec("PyInstaller") is None:
        raise ReleaseError(
            "PyInstaller is not installed in the selected interpreter. Run "
            "'.venv\\Scripts\\python.exe -m pip install -e \".[build]\"' first."
        )
    try:
        installed_pyinstaller = package_version("PyInstaller")
    except PackageNotFoundError as exc:
        raise ReleaseError("PyInstaller package metadata is missing; reinstall the build extra.") from exc
    if installed_pyinstaller != PYINSTALLER_VERSION:
        raise ReleaseError(
            f"PyInstaller {PYINSTALLER_VERSION} is required, but {installed_pyinstaller} is installed. "
            "Reinstall the project build extra."
        )

    _set_project_version(APP_VERSION)
    version_dir = _write_build_metadata(APP_VERSION, version_tuple)
    dist_dir = clean_generated_build_dir("pyinstaller-dist")
    work_dir = clean_generated_build_dir("pyinstaller-work")
    command = _pyinstaller_command(version_dir, dist_dir, work_dir)
    _run(command, "PyInstaller one-folder build")

    packaged_app = dist_dir / "GestureManager"
    _validate_packaged_app(packaged_app)

    staging_release = clean_generated_build_dir("release-staging")
    shutil.copytree(packaged_app, staging_release, dirs_exist_ok=True)
    _write_portable_readme(staging_release / "README.txt", APP_VERSION)
    built_at = datetime.now(timezone.utc).isoformat()
    _write_release_metadata(staging_release, APP_VERSION, built_at)
    portable_archive = _create_portable_zip(staging_release, APP_VERSION)
    _validate_portable_release(staging_release, APP_VERSION, portable_archive)

    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    release_dir = RELEASES_DIR / APP_VERSION
    _preserve_existing_release(release_dir)
    try:
        staging_release.rename(release_dir)
    except OSError as exc:
        raise ReleaseError(f"Could not move the completed portable release to {release_dir}: {exc}") from exc
    print(f"Portable release folder: {release_dir}")
    print(f"Portable ZIP archive: {release_dir / portable_archive.name}")
    return release_dir


def main() -> int:
    try:
        output = build()
    except ReleaseError as exc:
        print(f"RELEASE FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Portable release complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
