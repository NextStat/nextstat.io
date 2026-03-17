from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.validate_release_publish_artifacts import validate_release_publish_artifacts


def _write_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("nextstat/__init__.py", "__version__ = '0.10.1'\n")


def _write_sdist(path: Path) -> None:
    with tarfile.open(path, "w:gz") as archive:
        payload = path.parent / "PKG-INFO"
        payload.write_text("Metadata-Version: 2.1\nName: nextstat\n", encoding="utf-8")
        archive.add(payload, arcname="nextstat-0.10.1/PKG-INFO")


def test_validate_release_publish_artifacts_accepts_valid_wheels_and_sdist(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "nextstat-0.10.1-cp313-cp313-macosx_11_0_arm64.whl")
    _write_sdist(tmp_path / "nextstat-0.10.1.tar.gz")

    files = validate_release_publish_artifacts(tmp_path, "nextstat")

    assert [path.name for path in files] == [
        "nextstat-0.10.1-cp313-cp313-macosx_11_0_arm64.whl",
        "nextstat-0.10.1.tar.gz",
    ]


def test_validate_release_publish_artifacts_rejects_invalid_wheel(tmp_path: Path) -> None:
    bad = tmp_path / "nextstat-0.10.1-cp311-cp311-macosx_10_12_x86_64.whl"
    bad.write_text("not a zip file", encoding="utf-8")

    with pytest.raises(zipfile.BadZipFile):
        validate_release_publish_artifacts(tmp_path, "nextstat")


def test_validate_release_publish_artifacts_requires_matching_prefix(tmp_path: Path) -> None:
    _write_wheel(tmp_path / "nextstat_cli-0.10.1-py3-none-any.whl")

    with pytest.raises(SystemExit, match="no release publish artifacts found"):
        validate_release_publish_artifacts(tmp_path, "nextstat")
