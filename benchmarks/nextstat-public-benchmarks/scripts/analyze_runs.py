#!/usr/bin/env python3
"""Analyze local benchmark runs (artifacts + snapshots) into a single summary.

Stdlib-only on purpose: this should work in CI and on remote runners without
extra dependencies.

Input roots:
- artifacts/ (ad-hoc run dirs)
- manifests/snapshots/ (published snapshots via publish_snapshot.py)

Outputs:
- summary.json: machine-readable index
- summary.md: human-readable overview (latest run per suite/device)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _iter_suite_indices(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if name.endswith("_suite.json") or ("_suite_" in name and name.endswith(".json")):
            out.append(p)
        elif name.endswith("_summary.json") and "multiseed" in name:
            out.append(p)
    return sorted(out)


@dataclass(frozen=True)
class RunSummary:
    relpath: str
    suite: str
    device: str
    schema_version: str
    mtime_s: float
    n_cases: int
    n_ok: int
    n_warn: int
    n_failed: int
    meta: dict[str, Any]


def _summarize_index(doc: dict[str, Any]) -> tuple[str, str, str, int, int, int, int, dict[str, Any]]:
    schema_version = str(doc.get("schema_version") or doc.get("schema") or "")
    suite = str(doc.get("suite", "")).strip()
    if not suite:
        if schema_version.startswith("nextstat."):
            parts = schema_version.split(".")
            suite = parts[1] if len(parts) > 1 else "unknown"
        else:
            suite = "unknown"

    # Device varies by suite. Prefer explicit.
    device = str(doc.get("device", "")) or str((doc.get("config", {}) or {}).get("device", "")) or "cpu"

    # Common suite layout: cases[] plus summary counts.
    cases = doc.get("cases")
    if isinstance(cases, list):
        n_cases = len(cases)
        statuses = [str(c.get("status", "")) for c in cases if isinstance(c, dict)]
        n_ok = sum(1 for s in statuses if s == "ok")
        n_warn = sum(1 for s in statuses if s == "warn")
        n_failed = sum(1 for s in statuses if s not in {"ok", "warn"})
        return suite, device, schema_version, n_cases, n_ok, n_warn, n_failed, dict(doc.get("meta", {}) or {})

    # Bayesian multiseed summary: embed by convention.
    if schema_version == "nextstat.bayesian_multiseed_summary.v1":
        # Treat it as 1 synthetic case.
        return "bayesian_multiseed", "cpu", schema_version, 1, 1, 0, 0, dict(doc.get("meta", {}) or {})

    return suite or "unknown", device, schema_version, 0, 0, 0, 0, dict(doc.get("meta", {}) or {})


def _rel(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return str(p)


def _pick_latest(runs: list[RunSummary]) -> dict[tuple[str, str], RunSummary]:
    # Prefer newest file mtime per (suite, device).
    latest: dict[tuple[str, str], RunSummary] = {}
    for r in sorted(runs, key=lambda x: (x.suite, x.device, x.mtime_s)):
        latest[(r.suite, r.device)] = r
    return latest


def _render_md(latest: dict[tuple[str, str], RunSummary]) -> str:
    lines = [
        "# Benchmark Runs — Summary",
        "",
        f"Generated at: `{_utc_now_iso()}`",
        "",
        "| Suite | Device | Cases | OK | Warn | Failed | Latest |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for (suite, device), r in sorted(latest.items(), key=lambda x: (x[0][0], x[0][1])):
        lines.append(
            f"| `{suite}` | `{device}` | {r.n_cases} | {r.n_ok} | {r.n_warn} | {r.n_failed} | `{r.relpath}` |"
        )
    lines.append("")

    needs = [
        r for r in latest.values()
        if (r.n_failed > 0 or r.n_warn > 0) and r.n_cases > 0
    ]
    if needs:
        lines.append("## Needs Attention")
        lines.append("")
        lines.append("| Suite | Device | Warn | Failed | Latest |")
        lines.append("|---|---|---:|---:|---|")
        for r in sorted(needs, key=lambda x: (x.n_failed, x.n_warn, x.suite, x.device), reverse=True):
            lines.append(f"| `{r.suite}` | `{r.device}` | {r.n_warn} | {r.n_failed} | `{r.relpath}` |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root (defaults to CWD).")
    ap.add_argument("--artifacts", default="artifacts", help="Artifacts directory under root.")
    ap.add_argument("--snapshots", default="manifests/snapshots", help="Snapshots directory under root.")
    ap.add_argument("--out-dir", default="out/analysis", help="Output directory under root.")
    ap.add_argument(
        "--include-failed",
        action="store_true",
        help="Include runs where all cases failed (default: skip to reduce noise).",
    )
    args = ap.parse_args()

    root = Path(args.root).resolve()
    artifacts_root = (root / args.artifacts).resolve()
    snapshots_root = (root / args.snapshots).resolve()

    indices = _iter_suite_indices(artifacts_root) + _iter_suite_indices(snapshots_root)
    runs: list[RunSummary] = []
    for p in indices:
        try:
            doc = _load_json(p)
        except Exception:
            continue
        suite, device, schema_version, n_cases, n_ok, n_warn, n_failed, meta = _summarize_index(doc)
        # Drop obviously broken device labels (historical artifacts).
        if "," in str(device):
            continue
        if not args.include_failed and n_cases > 0 and n_ok == 0 and n_warn == 0:
            continue
        try:
            mtime_s = float(p.stat().st_mtime)
        except Exception:
            mtime_s = 0.0
        runs.append(
            RunSummary(
                relpath=_rel(p, root),
                suite=suite,
                device=device,
                schema_version=schema_version,
                mtime_s=mtime_s,
                n_cases=n_cases,
                n_ok=n_ok,
                n_warn=n_warn,
                n_failed=n_failed,
                meta=meta,
            )
        )

    latest = _pick_latest(runs)

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": "nextstat.analysis_runs_summary.v1",
                "generated_at": _utc_now_iso(),
                "cwd": os.getcwd(),
                "inputs": {
                    "artifacts": str(artifacts_root),
                    "snapshots": str(snapshots_root),
                },
                "runs": [r.__dict__ for r in runs],
                "latest": {f"{k[0]}::{k[1]}": v.__dict__ for k, v in latest.items()},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(_render_md(latest) + "\n", encoding="utf-8")

    print(f"Wrote: {out_dir/'summary.json'}")
    print(f"Wrote: {out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
