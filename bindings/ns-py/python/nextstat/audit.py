"""Audit trail baseline helpers (Phase 9.R.2).

This is the OSS baseline: local, explicit run metadata logging and run bundles.

It is intentionally minimal and does not implement enterprise audit trail features
like append-only storage, e-signatures, approvals, or model registry integration.
Those belong to Pro modules (see docs/legal/open-core-boundaries.md).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_hex(path.read_bytes())


def _file_size(path: Path) -> int:
    return int(path.stat().st_size)


def _ensure_empty_dir(dir_path: Path) -> None:
    if dir_path.exists():
        if not dir_path.is_dir():
            raise ValueError(f"bundle path exists but is not a directory: {dir_path}")
        if any(dir_path.iterdir()):
            raise ValueError(f"bundle directory must be empty: {dir_path}")
    else:
        dir_path.mkdir(parents=True, exist_ok=True)


def _json_pretty_bytes(obj: Any) -> bytes:
    # Stable within Python by deterministic dict construction + sorted lists where needed.
    # We do not sort keys globally to preserve semantic ordering (and to mirror CLI bundles).
    s = json.dumps(obj, indent=2)
    return (s + "\n").encode("utf-8")


def _try_split_pyhf_workspace(input_bytes: bytes) -> Optional[Tuple[bytes, bytes]]:
    try:
        ws = json.loads(input_bytes)
    except Exception:
        return None

    if not isinstance(ws, dict):
        return None

    if not ("channels" in ws and "measurements" in ws and "observations" in ws):
        return None

    channels = ws.get("channels")
    measurements = ws.get("measurements")
    observations = ws.get("observations")
    version = ws.get("version")

    if not isinstance(channels, list) or not isinstance(measurements, list) or not isinstance(observations, list):
        return None

    # Spec: keep only semantic model description (no observations).
    spec: Dict[str, Any] = {"channels": channels, "measurements": measurements}
    if version is not None:
        spec["version"] = version

    # Data: observations, sorted by name for determinism (matches CLI behavior).
    obs_sorted = list(observations)
    try:
        obs_sorted.sort(key=lambda o: str(o.get("name")) if isinstance(o, dict) else "")
    except Exception:
        # If sorting fails, keep as-is; still a valid split.
        pass

    data: Dict[str, Any] = {"observations": obs_sorted}

    return (_json_pretty_bytes(spec), _json_pretty_bytes(data))


def environment_fingerprint() -> Dict[str, Any]:
    """Small, privacy-preserving environment fingerprint for reproducibility."""
    try:
        import nextstat as ns  # type: ignore
    except Exception:
        ns = None  # type: ignore

    out: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }

    if ns is not None:
        out["nextstat_version"] = getattr(ns, "__version__", None)
        core = getattr(ns, "_core", None)
        if core is not None:
            out["nextstat_core_version"] = getattr(core, "__version__", None)

    # Best-effort git metadata (optional).
    repo = os.getenv("NS_REPO_ROOT")
    if repo:
        out["repo_root"] = repo
    return out


@dataclass(frozen=True)
class BundleInputMeta:
    original_path: str
    input_sha256: str
    data_sha256: Optional[str] = None
    model_spec_sha256: Optional[str] = None


@dataclass(frozen=True)
class BundleMeta:
    tool: str
    tool_version: str
    created_unix_ms: int
    command: str
    args: Any
    input: BundleInputMeta
    env: Dict[str, Any]


def write_bundle(
    bundle_dir: str | Path,
    *,
    command: str,
    args: Any,
    input_path: str | Path,
    output_value: Any,
    tool_version: Optional[str] = None,
) -> None:
    """Write a reproducible run bundle (Python mirror of `ns-cli --bundle`).

    Layout:
    - meta.json
    - inputs/input.json (+ optional inputs/model_spec.json, inputs/data.json)
    - outputs/result.json
    - manifest.json (sha256 + size per file)
    """

    bdir = Path(bundle_dir)
    in_path = Path(input_path)

    _ensure_empty_dir(bdir)
    inputs_dir = bdir / "inputs"
    outputs_dir = bdir / "outputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    input_bytes = in_path.read_bytes()
    input_sha256 = _sha256_hex(input_bytes)

    input_copy = inputs_dir / "input.json"
    input_copy.write_bytes(input_bytes)

    data_sha256: Optional[str] = None
    model_spec_sha256: Optional[str] = None

    split = _try_split_pyhf_workspace(input_bytes)
    if split is not None:
        spec_bytes, data_bytes = split
        spec_path = inputs_dir / "model_spec.json"
        data_path = inputs_dir / "data.json"
        spec_path.write_bytes(spec_bytes)
        data_path.write_bytes(data_bytes)
        model_spec_sha256 = _sha256_hex(spec_bytes)
        data_sha256 = _sha256_hex(data_bytes)

    try:
        import nextstat as ns  # type: ignore

        ns_version = getattr(ns, "__version__", "0.0.0")
    except Exception:
        ns_version = "0.0.0"

    created_unix_ms = int(time.time() * 1000.0)
    meta = BundleMeta(
        tool="nextstat",
        tool_version=(str(tool_version) if tool_version is not None else str(ns_version)),
        created_unix_ms=created_unix_ms,
        command=str(command),
        args=args,
        input=BundleInputMeta(
            original_path=str(in_path),
            input_sha256=input_sha256,
            data_sha256=data_sha256,
            model_spec_sha256=model_spec_sha256,
        ),
        env=environment_fingerprint(),
    )

    meta_path = bdir / "meta.json"
    meta_path.write_bytes(_json_pretty_bytes(_as_json(meta)))

    out_path = outputs_dir / "result.json"
    out_path.write_bytes(_json_pretty_bytes(output_value))

    files: List[Dict[str, Any]] = []
    for rel in ["meta.json", "inputs/input.json", "outputs/result.json"]:
        p = bdir / rel
        files.append({"path": rel, "bytes": _file_size(p), "sha256": _sha256_file(p)})
    for rel in ["inputs/model_spec.json", "inputs/data.json"]:
        p = bdir / rel
        if p.exists():
            files.append({"path": rel, "bytes": _file_size(p), "sha256": _sha256_file(p)})

    manifest = {"bundle_version": 1, "files": files}
    (bdir / "manifest.json").write_bytes(_json_pretty_bytes(manifest))


def _as_json(meta: BundleMeta) -> Dict[str, Any]:
    # Keep key ordering close to Rust CLI bundle schema.
    d: Dict[str, Any] = {
        "tool": meta.tool,
        "tool_version": meta.tool_version,
        "created_unix_ms": meta.created_unix_ms,
        "command": meta.command,
        "args": meta.args,
        "input": {
            "original_path": meta.input.original_path,
            "input_sha256": meta.input.input_sha256,
        },
        "env": meta.env,
    }
    if meta.input.data_sha256 is not None:
        d["input"]["data_sha256"] = meta.input.data_sha256
    if meta.input.model_spec_sha256 is not None:
        d["input"]["model_spec_sha256"] = meta.input.model_spec_sha256
    return d


__all__ = [
    "BundleInputMeta",
    "BundleMeta",
    "environment_fingerprint",
    "write_bundle",
]

