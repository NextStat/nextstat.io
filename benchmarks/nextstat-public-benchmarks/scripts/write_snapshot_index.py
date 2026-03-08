"""Write a small snapshot index JSON for published artifacts.

This is intentionally stdlib-only so it can run in CI without extra deps.

The index is meant for *discovery* and *auditability*:
- suite name + snapshot id
- git/workflow context (when available)
      - list of artifact files with size + sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


@dataclass(frozen=True)
class Artifact:
    relpath: str
    bytes: int
    sha256: str


def _status_from_passed(passed: bool) -> str:
    return "passed" if passed else "failed"


def _iter_artifacts(root: Path, *, exclude_dirs: set[str]) -> Iterable[Artifact]:
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        rel = p.relative_to(root).as_posix()
        top = rel.split("/", 1)[0] if rel else rel
        if top in exclude_dirs:
            continue
        st = p.stat()
        yield Artifact(relpath=rel, bytes=st.st_size, sha256=_sha256_file(p))


def _metric_extreme(reviewed_cases: list[dict[str, Any]], *, metric_key: str, prefer: str) -> dict[str, Any]:
    best_case: str | None = None
    best_value: float | None = None
    for row in reviewed_cases:
        value = _safe_float(row.get(metric_key))
        if value is None:
            continue
        if best_value is None:
            best_case = str(row.get("case") or "unknown")
            best_value = value
            continue
        if prefer == "max" and value > best_value:
            best_case = str(row.get("case") or "unknown")
            best_value = value
        elif prefer == "min" and value < best_value:
            best_case = str(row.get("case") or "unknown")
            best_value = value
    return {"case": best_case, "value": best_value}


def _derive_review_summary(reviewed_cases: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    failing_cases = sorted({str(row.get("case")) for row in failures if row.get("case")})
    summary: dict[str, Any] = {
        "n_reviewed_cases": len(reviewed_cases),
        "n_failures": len(failures),
        "n_failing_cases": len(failing_cases),
        "failing_cases": failing_cases,
    }
    for metric_key, prefer, summary_key in [
        ("divergence_rate", "max", "worst_divergence_rate"),
        ("max_treedepth_rate", "max", "worst_max_treedepth_rate"),
        ("max_r_hat", "max", "worst_max_r_hat"),
        ("min_ebfmi", "min", "worst_min_ebfmi"),
        ("min_ess_bulk", "min", "worst_min_ess_bulk"),
        ("min_ess_tail", "min", "worst_min_ess_tail"),
        ("min_ess_bulk_per_sec", "min", "worst_min_ess_bulk_per_sec"),
        ("ess_per_sec", "min", "worst_ess_per_sec"),
    ]:
        extreme = _metric_extreme(reviewed_cases, metric_key=metric_key, prefer=prefer)
        if extreme["case"] is not None or extreme["value"] is not None:
            summary[summary_key] = extreme
    return summary


def _normalize_review_summary(
    review_summary: dict[str, Any] | None,
    *,
    reviewed_cases: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    base = _derive_review_summary(reviewed_cases, failures)
    if not isinstance(review_summary, dict):
        return base
    merged = dict(review_summary)
    merged.setdefault("n_reviewed_cases", base["n_reviewed_cases"])
    merged.setdefault("n_failures", base["n_failures"])
    merged.setdefault("n_failing_cases", base["n_failing_cases"])
    merged.setdefault("failing_cases", base["failing_cases"])
    for key, value in base.items():
        if key.startswith("worst_"):
            merged.setdefault(key, value)
    return merged


def _collect_suite_health(root: Path, *, artifacts: list[Artifact]) -> list[dict[str, Any]]:
    by_relpath = {a.relpath: a for a in artifacts}
    out: list[dict[str, Any]] = []
    for relpath, artifact in sorted(by_relpath.items()):
        if not relpath.endswith("_assessment.json"):
            continue
        path = root / relpath
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        schema_version = str(doc.get("schema_version") or "")
        if not schema_version.startswith("nextstat.") or not schema_version.endswith("_assessment.v1"):
            continue
        suite = str(doc.get("suite") or relpath.split("/", 1)[0] or "unknown")
        core = doc.get("core_quality") if isinstance(doc.get("core_quality"), dict) else {}
        gate = doc.get("promotion_gate") if isinstance(doc.get("promotion_gate"), dict) else {}
        reviewed_cases = gate.get("reviewed_cases") if isinstance(gate.get("reviewed_cases"), list) else []
        reviewed_cases = [row for row in reviewed_cases if isinstance(row, dict)]
        failures = gate.get("failures") if isinstance(gate.get("failures"), list) else []
        failures = [row for row in failures if isinstance(row, dict)]
        review_summary = _normalize_review_summary(
            gate.get("review_summary") if isinstance(gate.get("review_summary"), dict) else None,
            reviewed_cases=reviewed_cases,
            failures=failures,
        )
        out.append(
            {
                "suite": suite,
                "assessment_path": relpath,
                "assessment_sha256": artifact.sha256,
                "assessment_schema_version": schema_version,
                "core_quality": {
                    "passed": bool(core.get("passed")),
                    "status": str(core.get("status") or _status_from_passed(bool(core.get("passed")))),
                    "failure_count": len(core.get("failures")) if isinstance(core.get("failures"), list) else 0,
                    "warning_count": len(core.get("warnings")) if isinstance(core.get("warnings"), list) else 0,
                },
                "promotion_gate": {
                    "passed": bool(gate.get("passed")),
                    "status": str(gate.get("status") or _status_from_passed(bool(gate.get("passed")))),
                    "target_backend": str(gate.get("target_backend")) if isinstance(gate.get("target_backend"), str) else None,
                    "failure_count": len(failures),
                    "reviewed_case_count": len(reviewed_cases),
                    "failing_cases": review_summary.get("failing_cases", []),
                    "review_summary": review_summary,
                },
            }
        )
    return out


def _env_get(keys: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k in keys:
        v = os.environ.get(k)
        if v:
            out[k] = v
    return out


def _default_snapshot_id() -> str:
    run_id = os.environ.get("GITHUB_RUN_ID")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    sha = os.environ.get("GITHUB_SHA")
    if run_id and attempt and sha:
        return f"gha-{run_id}-{attempt}-{sha[:12]}"
    if sha:
        return f"local-{sha[:12]}"
    return "local-unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="suite name (e.g. hep, pharma, or a composite label)")
    ap.add_argument("--artifacts-dir", required=True, help="directory containing artifacts to index")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument(
        "--snapshot-id",
        default="",
        help="optional: snapshot id (default: derived from GitHub env or git SHA)",
    )
    args = ap.parse_args()

    root = Path(args.artifacts_dir).resolve()
    if not root.exists():
        raise SystemExit(f"artifacts dir not found: {root}")
    if not root.is_dir():
        raise SystemExit(f"artifacts dir is not a directory: {root}")

    snapshot_id = args.snapshot_id.strip() or _default_snapshot_id()

    # Exclude implementation detail dirs that may exist inside artifact bundles.
    exclude_dirs = {"mplconfig", ".gnupg", ".replication", "__pycache__"}
    artifacts = list(_iter_artifacts(root, exclude_dirs=exclude_dirs))
    suite_health = _collect_suite_health(root, artifacts=artifacts)

    doc: dict[str, Any] = {
        "schema_version": "nextstat.snapshot_index.v1",
        "generated_at": _utc_now_iso(),
        "snapshot_id": snapshot_id,
        "suite": args.suite,
        "git": {
            "sha": os.environ.get("GITHUB_SHA") or "",
            "ref": os.environ.get("GITHUB_REF") or "",
            "repository": os.environ.get("GITHUB_REPOSITORY") or "",
        },
        "workflow": _env_get(
            [
                "GITHUB_WORKFLOW",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_JOB",
                "RUNNER_OS",
                "RUNNER_ARCH",
            ]
        ),
        "artifacts": [{"path": a.relpath, "bytes": a.bytes, "sha256": a.sha256} for a in artifacts],
    }
    if suite_health:
        doc["suite_health"] = suite_health

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
