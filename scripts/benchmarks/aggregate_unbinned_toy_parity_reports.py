#!/usr/bin/env python3
"""Aggregate per-backend unbinned toy parity JSON reports into one matrix report.

Inputs are JSON files produced by tests under:
  tests/python/test_unbinned_fit_toys_cli_parity.py

Stdlib-only for CI portability.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "nextstat.unbinned_toy_parity_report.v1"
OUTPUT_SCHEMA = "nextstat.unbinned_toy_parity_matrix_report.v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class Report:
    path: Path
    payload: dict[str, Any]

    @property
    def backend(self) -> str:
        return str(self.payload.get("backend", "unknown"))

    @property
    def kind(self) -> str:
        return str(self.payload.get("kind", "unknown"))

    @property
    def ok(self) -> bool:
        return bool(self.payload.get("ok", False))


def _load_reports(root: Path) -> list[Report]:
    out: list[Report] = []
    if not root.exists():
        return out
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema_version") != INPUT_SCHEMA:
            continue
        out.append(Report(path=path, payload=payload))
    return out


def _summarize(reports: list[Report]) -> dict[str, Any]:
    grouped: dict[str, list[Report]] = {}
    for report in reports:
        grouped.setdefault(report.backend, []).append(report)

    backends: dict[str, Any] = {}
    overall_ok = True

    for backend, items in sorted(grouped.items()):
        kinds: dict[str, Any] = {}
        backend_ok = True
        for report in items:
            relpath = str(report.path.as_posix())
            payload = dict(report.payload)
            payload["_source"] = relpath
            kinds[report.kind] = payload
            backend_ok = backend_ok and report.ok
        backends[backend] = {
            "ok": backend_ok,
            "n_reports": len(items),
            "kinds": kinds,
        }
        overall_ok = overall_ok and backend_ok

    return {
        "schema_version": OUTPUT_SCHEMA,
        "generated_at": _utc_now_iso(),
        "ok": overall_ok,
        "n_reports": len(reports),
        "backends": backends,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Aggregate unbinned toy parity reports.")
    ap.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded parity report artifacts",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSON path",
    )
    args = ap.parse_args(argv)

    reports = _load_reports(args.reports_dir.resolve())
    summary = _summarize(reports)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Non-blocking aggregator: test jobs provide gating.
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
