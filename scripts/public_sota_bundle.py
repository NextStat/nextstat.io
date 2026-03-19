"""Generate the public SOTA bundle v1 from the claim matrix.

Produces a single JSON + Markdown artifact that explains the v1.0 SOTA
claim end-to-end: which domains are SOTA, which are proof-pending,
which are access-pending, and which are scoped out.

Usage:
  python3 -m scripts.public_sota_bundle --check
  python3 -m scripts.public_sota_bundle --out-json tmp/sota_bundle.json --out-md tmp/sota_bundle.md
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.sota_claim_matrix import build_report, load_matrix, validate_matrix

SNAPSHOTS_DIR_REL = Path("benchmarks") / "nextstat-public-benchmarks" / "manifests" / "snapshots"
BAYESIAN_RESULTS_V10_REL = (
    Path("benchmarks")
    / "nextstat-public-benchmarks"
    / "suites"
    / "bayesian"
    / "results_v10"
)

# Map claim-matrix domain names to benchmark suite subdirectory names
_DOMAIN_SUITE_MAP: dict[str, str] = {
    "hep": "hep",
    "pharma": "pharma",
    "econometrics": "econometrics",
    "bayesian": "bayesian",
    "timeseries": "timeseries",
}

_ALLOWED_SNAPSHOT_TAILS: dict[str, dict[str, set[str]]] = {
    "econometrics": {
        "skipped_parity_cases": {"aipw_ate"},
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _classify_case_issue(domain: str, case: dict[str, Any]) -> dict[str, Any] | None:
    issue: dict[str, Any] = {
        "case": case["case"],
        "status": case.get("status", "n/a"),
        "parity_status": case.get("parity_status", "n/a"),
        "allowed": False,
        "reason": "",
    }

    status = case.get("status")
    parity_status = case.get("parity_status")
    if status not in ("warn", "failed") and parity_status not in ("warn", "failed", "skipped"):
        return None

    allowed = _ALLOWED_SNAPSHOT_TAILS.get(domain, {})
    if parity_status == "skipped" and case["case"] in allowed.get("skipped_parity_cases", set()):
        issue["allowed"] = True
        issue["reason"] = "documented competitor-baseline exception"

    return issue


def _load_latest_suite_snapshot(repo: Path, domain: str) -> dict[str, Any] | None:
    """Load the latest benchmark suite snapshot for a domain, if one exists."""
    suite_name = _DOMAIN_SUITE_MAP.get(domain)
    if suite_name is None:
        return None
    snapshots_dir = repo / SNAPSHOTS_DIR_REL
    if not snapshots_dir.exists():
        return None
    latest = None
    for snap_dir in sorted(snapshots_dir.iterdir()):
        suite_json = snap_dir / suite_name / f"{suite_name}_suite.json"
        if suite_json.exists():
            latest = suite_json
    if latest is None:
        return None
    return json.loads(latest.read_text(encoding="utf-8"))


def _load_bayesian_public_proof(repo: Path) -> tuple[Path, dict[str, Any]] | None:
    summary_path = repo / BAYESIAN_RESULTS_V10_REL / "bayesian_multiseed_summary.json"
    if not summary_path.exists():
        return None
    return summary_path, json.loads(summary_path.read_text(encoding="utf-8"))


def _gather_domain_evidence(repo: Path, domain: str, entry: dict) -> dict[str, Any]:
    """Gather machine-readable evidence for a single domain."""
    evidence: dict[str, Any] = {"proof_refs_valid": True, "missing_proof_refs": []}
    for ref in entry.get("proof_refs", []):
        if not (repo / ref).exists():
            evidence["proof_refs_valid"] = False
            evidence["missing_proof_refs"].append(ref)

    if domain == "bayesian":
        bayes = _load_bayesian_public_proof(repo)
        if bayes is not None:
            summary_path, data = bayes
            rows = data.get("cases", [])
            issue_rows = []
            n_ok = 0
            n_warn = 0
            n_failed = 0
            worst_case = None
            worst_ess = None
            for row in rows:
                statuses = list(row.get("statuses", []))
                if any(status == "failed" for status in statuses):
                    n_failed += 1
                    issue_rows.append(
                        {
                            "case": f"{row['case']}::{row['backend']}",
                            "status": "failed",
                            "parity_status": "ok",
                            "allowed": False,
                            "reason": "multiseed_status_failed",
                        }
                    )
                elif any(status == "warn" for status in statuses):
                    n_warn += 1
                    issue_rows.append(
                        {
                            "case": f"{row['case']}::{row['backend']}",
                            "status": "warn",
                            "parity_status": "ok",
                            "allowed": False,
                            "reason": "multiseed_status_warn",
                        }
                    )
                else:
                    n_ok += 1

                ess_values = row.get("min_ess_bulk_per_sec", [])
                if ess_values:
                    candidate = min(float(v) for v in ess_values)
                    if worst_ess is None or candidate < worst_ess:
                        worst_ess = candidate
                        worst_case = f"{row['case']}::{row['backend']}"

            evidence["benchmark_snapshot"] = {
                "schema_version": data.get("schema_version", "unknown"),
                "source": str(summary_path.relative_to(repo)),
                "backends": data.get("backends", ""),
                "n_cases": len(rows),
                "n_ok": n_ok,
                "n_warn": n_warn,
                "n_failed": n_failed,
                "worst_case": worst_case,
                "all_ok": len(issue_rows) == 0,
                "cases_with_issues": issue_rows,
                "unresolved_cases_with_issues": issue_rows,
            }
    else:
        suite = _load_latest_suite_snapshot(repo, domain)
        if suite is not None:
            summary = suite.get("summary", {})
            cases = suite.get("cases", [])
            # Prefer summary-level counts; fall back to deriving from per-case status field
            n_ok = summary.get("n_ok")
            n_warn = summary.get("n_warn", 0)
            n_failed = summary.get("n_failed", 0)
            if n_ok is None:
                # Suites without per-case status (hep, pharma) — count from cases
                has_status = any(c.get("status") is not None for c in cases)
                if has_status:
                    n_ok = sum(1 for c in cases if c.get("status") == "ok")
                    n_warn = sum(1 for c in cases if c.get("status") == "warn")
                    n_failed = sum(1 for c in cases if c.get("status") == "failed")
                else:
                    # No per-case status at all — all cases are implicitly ok
                    n_ok = len(cases)
                    n_warn = 0
                    n_failed = 0
            issue_rows = [
                issue
                for case in cases
                if (issue := _classify_case_issue(domain, case)) is not None
            ]
            unresolved_issues = [issue for issue in issue_rows if not issue["allowed"]]

            snapshots_dir = repo / SNAPSHOTS_DIR_REL
            latest_path = None
            for snap_dir in sorted(snapshots_dir.iterdir()):
                suite_json = snap_dir / _DOMAIN_SUITE_MAP[domain] / f"{_DOMAIN_SUITE_MAP[domain]}_suite.json"
                if suite_json.exists():
                    latest_path = suite_json

            evidence["benchmark_snapshot"] = {
                "schema_version": suite.get("schema_version", "unknown"),
                "source": str(latest_path.relative_to(repo)) if latest_path is not None else "",
                "n_cases": summary.get("n_cases", len(cases)),
                "n_ok": n_ok,
                "n_warn": n_warn,
                "n_failed": n_failed,
                "worst_case": summary.get("worst_case"),
                "all_ok": len(unresolved_issues) == 0,
                "cases_with_issues": issue_rows,
                "unresolved_cases_with_issues": unresolved_issues,
            }

    if domain == "hep":
        hep_matrix = repo / "hep_surface_matrix_v1.json"
        if hep_matrix.exists():
            data = json.loads(hep_matrix.read_text(encoding="utf-8"))
            surfaces = data.get("surfaces", [])
            evidence["governance"] = {
                "total_surfaces": len(surfaces),
                "stable": sum(1 for s in surfaces if s.get("maturity_class") == "stable"),
            }

    if domain == "pharma":
        schema_dir = repo / "docs" / "schemas" / "pharma"
        if schema_dir.exists():
            evidence["validation_schemas"] = len(list(schema_dir.glob("*.schema.json")))

    if domain == "r":
        namespace = repo / "bindings" / "ns-r" / "NAMESPACE"
        testthat_dir = repo / "bindings" / "ns-r" / "tests" / "testthat"
        export_count = 0
        if namespace.exists():
            export_count = sum(
                1 for line in namespace.read_text(encoding="utf-8").splitlines() if line.startswith("export(")
            )
        evidence["r_surface"] = {
            "export_count": export_count,
            "test_file_count": len(list(testthat_dir.glob("test-*.R"))),
            "source_build_doc": "docs/references/r-bindings.md",
        }

    if domain == "wasm":
        workflow = (repo / ".github" / "workflows" / "rust-tests.yml").read_text(encoding="utf-8")
        pkg_dir = repo / "playground" / "pkg"
        evidence["wasm_surface"] = {
            "build_script": "scripts/playground_build_wasm.sh",
            "pkg_artifacts_present": (pkg_dir / "ns_wasm.js").exists() and (pkg_dir / "ns_wasm_bg.wasm").exists(),
            "pkg_file_count": len(list(pkg_dir.glob("*"))) if pkg_dir.exists() else 0,
            "ci_jobs": {
                "wasm_smoke": "wasm-smoke:" in workflow,
                "wasm_playground_build": "wasm-playground-build:" in workflow,
            },
        }

    return evidence


def build_bundle(repo: Path | None = None) -> dict[str, Any]:
    repo = repo or _repo_root()
    matrix = load_matrix(repo)
    validate_matrix(matrix, repo)
    report = build_report(repo)

    domains_detail = []
    for entry in sorted(matrix["domains"], key=lambda e: e["domain"]):
        detail: dict[str, Any] = {
            "domain": entry["domain"],
            "status": entry["status"],
            "scope": entry["scope"],
            "proof_ref_count": len(entry["proof_refs"]),
            "access_ref_count": len(entry["access_refs"]),
            "guard_doc": entry["guard_doc"],
            "evidence": _gather_domain_evidence(repo, entry["domain"], entry),
        }
        domains_detail.append(detail)

    return {
        "schema_version": "nextstat.public_sota_bundle.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "claim_matrix_schema_version": matrix["schema_version"],
        "summary": report["summary"],
        "domains": domains_detail,
    }


def render_bundle_markdown(bundle: dict[str, Any]) -> str:
    s = bundle["summary"]
    lines = [
        "# NextStat Public SOTA Bundle v1",
        "",
        f"Generated: `{bundle['generated_at_utc']}`",
        "",
        "## SOTA Boundary",
        "",
        f"**{s['status_counts'].get('sota', 0)}/{s['total_domains']}** domains currently claim SOTA.",
        "",
    ]

    for status_key, label in [
        ("sota_domains", "SOTA"),
        ("proof_pending_domains", "Proof Pending"),
        ("access_pending_domains", "Access Pending"),
        ("scoped_out_domains", "Scoped Out"),
    ]:
        domains = s.get(status_key, [])
        lines.append(f"### {label}")
        lines.append("")
        if domains:
            for d in domains:
                entry = next(e for e in bundle["domains"] if e["domain"] == d)
                lines.append(f"- **{d}**: {entry['scope']}")
        else:
            lines.append("- none")
        lines.append("")

    lines.extend([
        "## Domain Inventory",
        "",
        "| Domain | Status | Proof Refs | Access Refs | Guard Doc |",
        "| --- | --- | ---: | ---: | --- |",
    ])

    for entry in bundle["domains"]:
        lines.append(
            f"| `{entry['domain']}` | `{entry['status']}` | "
            f"`{entry['proof_ref_count']}` | `{entry['access_ref_count']}` | "
            f"`{entry['guard_doc']}` |"
        )

    lines.extend(["", "## Domain Evidence", ""])

    for entry in bundle["domains"]:
        ev = entry.get("evidence", {})
        lines.append(f"### {entry['domain']} (`{entry['status']}`)")
        lines.append("")
        lines.append(f"**Scope:** {entry['scope']}")
        lines.append("")

        snap = ev.get("benchmark_snapshot")
        if snap:
            ok_label = "ALL OK" if snap["all_ok"] else "HAS ISSUES"
            lines.append(
                f"**Benchmark snapshot:** {snap['n_cases']} cases, "
                f"{snap['n_ok']} ok, {snap['n_warn']} warn, "
                f"{snap['n_failed']} failed — **{ok_label}**"
            )
            if snap["cases_with_issues"]:
                for issue in snap["cases_with_issues"]:
                    suffix = (
                        f" ({issue['reason']})"
                        if issue.get("allowed") and issue.get("reason")
                        else ""
                    )
                    lines.append(
                        f"  - `{issue['case']}`: status={issue['status']}, "
                        f"parity={issue['parity_status']}"
                        f"{suffix}"
                    )
            lines.append("")

        gov = ev.get("governance")
        if gov:
            lines.append(
                f"**Governance:** {gov['stable']}/{gov['total_surfaces']} surfaces stable"
            )
            lines.append("")

        schemas = ev.get("validation_schemas")
        if schemas:
            lines.append(f"**Validation schemas:** {schemas} JSON schemas")
            lines.append("")

        r_surface = ev.get("r_surface")
        if r_surface:
            lines.append(
                f"**R surface:** {r_surface['export_count']} exports, "
                f"{r_surface['test_file_count']} testthat files"
            )
            lines.append("")

        wasm_surface = ev.get("wasm_surface")
        if wasm_surface:
            ok_label = "present" if wasm_surface["pkg_artifacts_present"] else "missing"
            lines.append(
                f"**WASM surface:** pkg artifacts {ok_label}, "
                f"{wasm_surface['pkg_file_count']} pkg files"
            )
            lines.append("")

        if not ev.get("proof_refs_valid", True):
            lines.append(f"**WARNING:** Missing proof refs: {ev['missing_proof_refs']}")
            lines.append("")

    return "\n".join(lines) + "\n"


def check_bundle(repo: Path | None = None) -> tuple[bool, str]:
    try:
        bundle = build_bundle(repo)
    except (ValueError, KeyError) as exc:
        return False, str(exc)

    s = bundle["summary"]
    n_sota = s["status_counts"].get("sota", 0)
    total = s["total_domains"]

    if not bundle["domains"]:
        return False, "Bundle has no domains"

    missing_proof = [
        f"{entry['domain']}: {entry['evidence']['missing_proof_refs']}"
        for entry in bundle["domains"]
        if not entry["evidence"].get("proof_refs_valid", True)
    ]
    if missing_proof:
        return False, "Missing proof refs in public SOTA bundle: " + "; ".join(missing_proof)

    dirty_sota = []
    for entry in bundle["domains"]:
        if entry["status"] != "sota":
            continue
        snap = entry["evidence"].get("benchmark_snapshot")
        if snap is None or snap["all_ok"]:
            continue
        dirty_sota.append(
            f"{entry['domain']}: {snap['unresolved_cases_with_issues']}"
        )

    if dirty_sota:
        return False, "Dirty SOTA snapshot(s): " + "; ".join(dirty_sota)

    return True, (
        f"OK: {n_sota}/{total} domains claim SOTA, "
        f"{len(s.get('proof_pending_domains', []))} proof-pending, "
        f"{len(s.get('access_pending_domains', []))} access-pending, "
        f"{len(s.get('scoped_out_domains', []))} scoped-out"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the public SOTA bundle.")
    parser.add_argument("--check", action="store_true", help="Validate and exit.")
    parser.add_argument("--out-json", help="Write JSON bundle.")
    parser.add_argument("--out-md", help="Write Markdown bundle.")
    args = parser.parse_args()

    ok, message = check_bundle()
    if not ok:
        print(message)
        return 1

    if args.check and not args.out_json and not args.out_md:
        print(message)
        return 0

    bundle = build_bundle()
    md = render_bundle_markdown(bundle)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")

    if not args.out_json and not args.out_md:
        print(json.dumps(bundle, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
