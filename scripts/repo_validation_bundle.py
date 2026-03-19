"""Repo-wide validation bundle — single canonical release truth.

Aggregates governance evidence across ALL domains (HEP, pharma, bayesian,
platform, ads) into one JSON + Markdown bundle. Supersedes the HEP-only
validation bundle for repo-wide release gating.

Sources:
  - repo_surface_matrix_v1.json (193 surfaces)
  - scripts/release_surface_matrix_v1.json (release governance)
  - bundle_slot_contract from RWS-01

Usage:
    python -m scripts.repo_validation_bundle --out-json tmp/repo_validation_bundle.json --out-md tmp/repo_validation_bundle.md
    python -m scripts.repo_validation_bundle --check
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_repo_matrix(repo: Path) -> dict[str, Any]:
    path = repo / "repo_surface_matrix_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_release_matrix(repo: Path) -> dict[str, Any]:
    path = repo / "scripts" / "release_surface_matrix_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ── Per-domain governance summary ────────────────────────────────────────


def _domain_summary(surfaces: list[dict]) -> dict[str, dict[str, Any]]:
    """Per-domain governance stats."""
    domains: dict[str, list[dict]] = {}
    for s in surfaces:
        domains.setdefault(s["domain"], []).append(s)

    out: dict[str, dict[str, Any]] = {}
    for domain, surfs in sorted(domains.items()):
        total = len(surfs)
        stable = sum(1 for s in surfs if s["public_status"] == "stable")
        required = sum(1 for s in surfs if s["release_status"] == "required")
        optional = sum(1 for s in surfs if s["release_status"] == "optional")
        migration = sum(
            1 for s in surfs
            if s["public_status"] == "stable"
            and s["release_status"] == "not_release_governed"
        )
        out[domain] = {
            "total": total,
            "stable": stable,
            "required": required,
            "optional": optional,
            "migration": migration,
            "fully_governed": migration == 0,
        }
    return out


# ── Release slice coverage ───────────────────────────────────────────────


def _release_slice_coverage(
    surfaces: list[dict], release: dict[str, Any]
) -> list[dict[str, Any]]:
    """Per-release-slice: linked surface count and evidence refs."""
    release_ids = {r["id"]: r for r in release["surfaces"]}
    linked: dict[str, list[str]] = {rid: [] for rid in release_ids}

    for s in surfaces:
        ref = s.get("release_surface_ref", "")
        if ref and ref in linked:
            linked[ref].append(s["surface_id"])

    entries = []
    for rid, r in sorted(release_ids.items()):
        entries.append({
            "release_surface_id": rid,
            "name": r["name"],
            "required_for_release": r["required_for_release"],
            "linked_surface_count": len(linked[rid]),
            "make_target": r.get("make_target", ""),
            "workflow_job": r.get("workflow_job", ""),
            "docs": r.get("docs", []),
        })
    return entries


# ── Governance quality checks ────────────────────────────────────────────


def _governance_quality(surfaces: list[dict]) -> dict[str, Any]:
    """Governance quality metrics matching test_governance_quality_smoke.py."""
    governed = [
        s for s in surfaces
        if s["release_status"] in ("required", "optional")
        and s["public_status"] != "internal"
    ]

    def _has_ref(s: dict) -> bool:
        return bool(
            s.get("release_surface_ref")
            or s.get("gate_ref")
            or s.get("support_contract_ref")
        )

    bare = [s["surface_id"] for s in governed if not _has_ref(s)]

    required_no_release_ref = [
        s["surface_id"] for s in governed
        if s["release_status"] == "required"
        and not s.get("release_surface_ref")
    ]

    optional_no_support_gate = [
        s["surface_id"] for s in governed
        if s["release_status"] == "optional"
        and not s.get("support_contract_ref")
        and not s.get("gate_ref")
    ]

    multi_ref = sum(
        1 for s in governed
        if sum(bool(s.get(k)) for k in (
            "release_surface_ref", "gate_ref", "support_contract_ref",
            "acceptance_ref", "validation_bundle_ref",
        )) >= 2
    )

    return {
        "governed_total": len(governed),
        "bare_governed_surfaces": bare,
        "required_without_release_ref": required_no_release_ref,
        "optional_without_support_or_gate": optional_no_support_gate,
        "multi_ref_count": multi_ref,
        "multi_ref_ratio": round(multi_ref / len(governed), 3) if governed else 0,
        "quality_ok": (
            len(bare) == 0
            and len(required_no_release_ref) == 0
            and len(optional_no_support_gate) == 0
        ),
    }


# ── Migration tracking ──────────────────────────────────────────────────


def _migration_summary(surfaces: list[dict]) -> dict[str, Any]:
    """Migration state for honest burndown tracking."""
    migration = [
        s for s in surfaces
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    by_domain = dict(Counter(s["domain"] for s in migration).most_common())
    by_owner = dict(Counter(s["owner_slice"] for s in migration).most_common())

    return {
        "total": len(migration),
        "by_domain": by_domain,
        "by_owner": by_owner,
        "surfaces": [s["surface_id"] for s in migration],
        "zero_migration": len(migration) == 0,
    }


# ── Build bundle ─────────────────────────────────────────────────────────


def build_bundle(repo: Path | None = None) -> dict[str, Any]:
    repo = repo or _repo_root()
    matrix = _load_repo_matrix(repo)
    release = _load_release_matrix(repo)
    surfaces = matrix["surfaces"]

    domain_stats = _domain_summary(surfaces)
    total = len(surfaces)
    total_stable = sum(v["stable"] for v in domain_stats.values())
    total_governed = sum(v["required"] + v["optional"] for v in domain_stats.values())
    total_migration = sum(v["migration"] for v in domain_stats.values())

    return {
        "schema_version": "nextstat.repo_validation_bundle.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "summary": {
            "total_surfaces": total,
            "stable": total_stable,
            "governed": total_governed,
            "migration": total_migration,
            "zero_migration": total_migration == 0,
        },
        "per_domain": domain_stats,
        "release_slice_coverage": _release_slice_coverage(surfaces, release),
        "governance_quality": _governance_quality(surfaces),
        "migration": _migration_summary(surfaces),
        "bundle_slot_contract": matrix.get("bundle_slot_contract", {}),
    }


# ── Check mode ───────────────────────────────────────────────────────────


def check_bundle(repo: Path | None = None) -> tuple[bool, str]:
    """Validate bundle integrity. Returns (ok, message)."""
    bundle = build_bundle(repo)
    issues: list[str] = []

    # 1. Governance quality must be OK
    gq = bundle["governance_quality"]
    if not gq["quality_ok"]:
        if gq["bare_governed_surfaces"]:
            issues.append(
                f"Governed surfaces without any ref: {gq['bare_governed_surfaces']}"
            )
        if gq["required_without_release_ref"]:
            issues.append(
                f"Required surfaces without release_surface_ref: "
                f"{gq['required_without_release_ref']}"
            )
        if gq["optional_without_support_or_gate"]:
            issues.append(
                f"Optional surfaces without support/gate: "
                f"{gq['optional_without_support_or_gate']}"
            )

    # 2. RWS-10 super-gate: zero stable migration (blocking)
    mig = bundle["migration"]
    if mig["total"] > 0:
        issues.append(
            f"BLOCKING: {mig['total']} stable surfaces in migration: "
            f"{mig['surfaces'][:10]}"
        )

    # 3. Every release slice must have at least one linked surface
    for rs in bundle["release_slice_coverage"]:
        if rs["linked_surface_count"] == 0:
            issues.append(
                f"Release slice {rs['release_surface_id']!r} has zero linked surfaces"
            )

    if issues:
        return False, "FAIL:\n" + "\n".join(f"  - {i}" for i in issues)

    s = bundle["summary"]
    return True, (
        f"OK: {s['governed']}/{s['total_surfaces']} governed, "
        f"{s['migration']} in migration"
    )


# ── Markdown renderer ────────────────────────────────────────────────────


def render_markdown(bundle: dict[str, Any]) -> str:
    s = bundle["summary"]
    lines = [
        "# Repo-wide Validation Bundle",
        "",
        f"Generated: `{bundle['generated_at_utc']}`",
        "",
        "## Summary",
        "",
        f"- **{s['total_surfaces']}** total surfaces",
        f"- **{s['stable']}** stable",
        f"- **{s['governed']}** governed (required + optional)",
        f"- **{s['migration']}** in migration",
        f"- Zero migration: **{'YES' if s['zero_migration'] else 'NO'}**",
        "",
        "## Per-Domain Breakdown",
        "",
        "| Domain | Total | Stable | Required | Optional | Migration | Governed |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for domain, stats in sorted(bundle["per_domain"].items()):
        gov = "YES" if stats["fully_governed"] else "NO"
        lines.append(
            f"| {domain} | {stats['total']} | {stats['stable']} | "
            f"{stats['required']} | {stats['optional']} | "
            f"{stats['migration']} | {gov} |"
        )

    lines.extend(["", "## Release Slice Coverage", ""])
    for rs in bundle["release_slice_coverage"]:
        req = "REQUIRED" if rs["required_for_release"] else "optional"
        lines.append(
            f"- **{rs['name']}** (`{rs['release_surface_id']}`) — "
            f"{req}, {rs['linked_surface_count']} linked surfaces"
        )
        if rs["make_target"]:
            lines.append(f"  - make target: `{rs['make_target']}`")
        if rs["workflow_job"]:
            lines.append(f"  - workflow job: `{rs['workflow_job']}`")

    gq = bundle["governance_quality"]
    lines.extend([
        "",
        "## Governance Quality",
        "",
        f"- Governed surfaces: **{gq['governed_total']}**",
        f"- Multi-ref ratio: **{gq['multi_ref_ratio']:.0%}** ({gq['multi_ref_count']}/{gq['governed_total']})",
        f"- Quality: **{'PASS' if gq['quality_ok'] else 'FAIL'}**",
    ])
    if gq["bare_governed_surfaces"]:
        lines.append(f"- Bare governed: {gq['bare_governed_surfaces']}")
    if gq["required_without_release_ref"]:
        lines.append(f"- Required without release ref: {gq['required_without_release_ref']}")
    if gq["optional_without_support_or_gate"]:
        lines.append(f"- Optional without support/gate: {gq['optional_without_support_or_gate']}")

    mig = bundle["migration"]
    lines.extend([
        "",
        "## Migration Burndown",
        "",
        f"- Total in migration: **{mig['total']}**",
    ])
    if mig["by_domain"]:
        lines.append(f"- By domain: {mig['by_domain']}")
    if mig["by_owner"]:
        lines.append(f"- By owner: {mig['by_owner']}")

    return "\n".join(lines) + "\n"


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build canonical repo-wide validation bundle."
    )
    parser.add_argument(
        "--out-json",
        default="tmp/repo_validation_bundle.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--out-md",
        default="tmp/repo_validation_bundle.md",
        help="Output Markdown path",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate bundle integrity and exit",
    )
    args = parser.parse_args()

    if args.check:
        ok, message = check_bundle()
        print(message, file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1

    bundle = build_bundle()
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(
        json.dumps(bundle, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    out_md.write_text(render_markdown(bundle), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
