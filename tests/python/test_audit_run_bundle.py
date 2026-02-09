from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


def _sha256_hex(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()


def test_audit_write_bundle_writes_hashes_and_manifest():
    from nextstat import audit

    repo = Path(__file__).resolve().parents[1]
    fixture = repo / "fixtures" / "simple_workspace.json"
    assert fixture.exists()

    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / "bundle"
        audit.write_bundle(
            bundle,
            command="test",
            args={"k": 1},
            input_path=fixture,
            output_value={"ok": True},
        )

        meta = json.loads((bundle / "meta.json").read_text())
        manifest = json.loads((bundle / "manifest.json").read_text())

        input_bytes = fixture.read_bytes()
        assert meta["input"]["input_sha256"] == _sha256_hex(input_bytes)

        # If we split pyhf workspace, hashes must exist.
        assert (bundle / "inputs" / "model_spec.json").exists()
        assert (bundle / "inputs" / "data.json").exists()
        spec_bytes = (bundle / "inputs" / "model_spec.json").read_bytes()
        data_bytes = (bundle / "inputs" / "data.json").read_bytes()
        assert meta["input"]["model_spec_sha256"] == _sha256_hex(spec_bytes)
        assert meta["input"]["data_sha256"] == _sha256_hex(data_bytes)

        files = manifest["files"]
        assert files and isinstance(files, list)
        for f in files:
            rel = f["path"]
            want = f["sha256"]
            p = bundle / rel
            got = _sha256_hex(p.read_bytes())
            assert want == got


def test_audit_write_bundle_errors_on_non_empty_dir():
    from nextstat import audit

    repo = Path(__file__).resolve().parents[1]
    fixture = repo / "fixtures" / "simple_workspace.json"
    assert fixture.exists()

    with tempfile.TemporaryDirectory() as td:
        bundle = Path(td) / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "junk.txt").write_text("x")

        try:
            audit.write_bundle(
                bundle,
                command="test",
                args={},
                input_path=fixture,
                output_value={"ok": True},
            )
        except ValueError as e:
            assert "bundle directory must be empty" in str(e)
        else:
            raise AssertionError("expected ValueError")

