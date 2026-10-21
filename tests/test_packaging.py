import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import build_release
from gesture_manager.assets import model_assets_directory


class PackagingConfigurationTests(unittest.TestCase):
    def test_pyinstaller_includes_dynamically_imported_scipy_modules(self):
        command = build_release._pyinstaller_command(
            Path("version"), Path("dist"), Path("work"),
        )
        hidden_imports = {
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--hidden-import"
        }
        self.assertIn("scipy._external.array_api_compat.numpy.linalg", hidden_imports)
        self.assertIn("scipy._external.array_api_compat.numpy.fft", hidden_imports)

    def test_semantic_version_is_converted_to_windows_file_version(self):
        self.assertEqual(build_release.parse_version("0.1.0"), (0, 1, 0, 0))
        self.assertEqual(build_release.parse_version("2.4.6-rc.1+build.8"), (2, 4, 6, 0))

    def test_invalid_or_out_of_range_versions_are_rejected(self):
        for version in (
            "1.2", "01.2.3", "1.2.3.4", "65536.0.0", "1.2.3-", "1.2.3-01", "1.2.3+.",
        ):
            with self.subTest(version=version), self.assertRaises(build_release.ReleaseError):
                build_release.parse_version(version)

    def test_model_assets_default_to_checkout_independent_project_location(self):
        expected = Path(__file__).resolve().parents[1] / "models"
        self.assertEqual(model_assets_directory(), expected)

    def test_frozen_model_assets_use_pyinstaller_bundle_root(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(sys, "frozen", True, create=True), \
                    patch.object(sys, "_MEIPASS", temp, create=True):
                self.assertEqual(model_assets_directory(), Path(temp) / "models")

    def test_generated_version_resources_use_one_version_value(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(build_release, "BUILD_DIR", Path(temp)):
                version_dir = build_release._write_build_metadata("0.1.0", (0, 1, 0, 0))
            data = json.loads((version_dir / "release_version.json").read_text(encoding="utf-8"))
            windows_resource = (version_dir / "windows_version_info.txt").read_text(encoding="utf-8")
            self.assertEqual(data["version"], "0.1.0")
            self.assertIn("FileVersion', '0.1.0'", windows_resource)
            self.assertIn("filevers=(0, 1, 0, 0)", windows_resource)

    def test_generated_directory_cleanup_rejects_unknown_names(self):
        with self.assertRaises(build_release.ReleaseError):
            build_release.clean_generated_build_dir("../releases")

    def test_portable_zip_contains_every_application_file_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            for relative in build_release.REQUIRED_PORTABLE_FILES:
                path = stage / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"test payload")
            build_release._write_portable_readme(stage / "README.txt", "0.1.0")
            build_release._write_release_metadata(
                stage, "0.1.0", "2026-10-21T00:00:00+00:00",
            )
            archive = build_release._create_portable_zip(stage, "0.1.0")

            build_release._validate_portable_release(stage, "0.1.0", archive)
            with ZipFile(archive) as zipped:
                self.assertIsNone(zipped.testzip())
                names = set(zipped.namelist())
            expected = {
                f"GestureManager-0.1.0/{path.relative_to(stage).as_posix()}"
                for path in stage.rglob("*")
                if path.is_file() and path != archive
            }
            self.assertEqual(names, expected)

    def test_previous_release_is_retained_when_rebuilding_same_version(self):
        with tempfile.TemporaryDirectory() as temp:
            releases = Path(temp) / "releases"
            old_release = releases / "0.1.0"
            old_release.mkdir(parents=True)
            marker = old_release / "old-build.txt"
            marker.write_text("preserve this release", encoding="utf-8")
            with patch.object(build_release, "RELEASES_DIR", releases):
                build_release._preserve_existing_release(old_release)
            preserved = list((releases / "_previous").glob("0.1.0-*"))
            self.assertEqual(len(preserved), 1)
            self.assertEqual((preserved[0] / marker.name).read_text(encoding="utf-8"),
                             "preserve this release")
            self.assertFalse(old_release.exists())


if __name__ == "__main__":
    unittest.main()
