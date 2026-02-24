#!/usr/bin/env python3
"""MaS baseline runner for pharma suite.

The Python package name `mas` on PyPI is ambiguous; this runner verifies
that the installed module exposes a pharmacometric API before attempting fit.
"""

from __future__ import annotations

import argparse
import importlib.util
import importlib
from importlib import metadata
import json
from pathlib import Path
from typing import Any


def _write(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _has_mas() -> bool:
    try:
        return (
            importlib.util.find_spec("mas") is not None
            or importlib.util.find_spec("MaS") is not None
        )
    except Exception:
        return False


def _load_mas_module():
    last_err: str | None = None
    for name in ("mas", "MaS"):
        try:
            mod = importlib.import_module(name)
            return name, mod, None
        except Exception as e:
            last_err = f"{type(e).__name__}:{e}"
            continue
    return None, None, last_err


def _mas_version() -> str:
    for pkg in ("mas", "MaS"):
        try:
            return metadata.version(pkg)
        except Exception:
            continue
    return "unknown"


def _has_pharm_api(mod: Any) -> bool:
    if mod is None:
        return False
    attrs = set(dir(mod))
    # Minimal expected API surface for pharmacometric engines
    expected_any = {
        "fit",
        "foce",
        "saem",
        "Model",
        "PopulationModel",
        "PharmModel",
    }
    return len(attrs.intersection(expected_any)) > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    args = ap.parse_args()

    out_path = Path(args.out_path)
    case_id = "unknown"
    try:
        case_obj = json.loads(Path(args.in_path).read_text())
        case_id = str(case_obj.get("case", "unknown"))
    except Exception:
        pass

    if not _has_mas():
        _write(out_path, {
            "schema_version": "nextstat.pharma_baseline_result.v1",
            "baseline": "mas",
            "case": case_id,
            "status": "skipped",
            "reason": "MaS not installed",
        })
        return 0

    mod_name, mod, import_err = _load_mas_module()
    ver = _mas_version()
    if mod is None:
        _write(out_path, {
            "schema_version": "nextstat.pharma_baseline_result.v1",
            "baseline": "mas",
            "case": case_id,
            "status": "skipped",
            "reason": f"MaS import failed despite module discovery ({import_err or 'unknown import error'})",
            "packages": {"module": str(mod_name), "version": ver},
        })
        return 0

    if not _has_pharm_api(mod):
        _write(out_path, {
            "schema_version": "nextstat.pharma_baseline_result.v1",
            "baseline": "mas",
            "case": case_id,
            "status": "skipped",
            "reason": "installed 'mas' package does not expose pharmacometric API",
            "packages": {"module": str(mod_name), "version": ver},
        })
        return 0

    # We detected a plausible pharmacometric API but no stable cross-version
    # contract in this harness yet.
    _write(out_path, {
        "schema_version": "nextstat.pharma_baseline_result.v1",
        "baseline": "mas",
        "case": case_id,
        "status": "failed",
        "reason": "MaS module detected but no stable runner contract implemented for this distribution",
        "packages": {"module": str(mod_name), "version": ver},
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
