from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_manifest_digest_script_writes_hex_and_binary_outputs(tmp_path: Path) -> None:
    manifest = tmp_path / "m15_bundle_manifest.json"
    manifest.write_text('{"schema_version":"m15_bundle_manifest_v1"}\n', encoding="utf-8")

    subprocess.check_call(
        [
            "python3",
            str(_repo_root() / "scripts" / "validation" / "write_manifest_digest.py"),
            "--manifest",
            str(manifest),
        ]
    )

    expected = hashlib.sha256(manifest.read_bytes()).digest()
    hex_out = tmp_path / "m15_bundle_manifest.sha256"
    bin_out = tmp_path / "m15_bundle_manifest.sha256.bin"

    assert hex_out.read_text(encoding="utf-8") == expected.hex() + "\n"
    assert bin_out.read_bytes() == expected
