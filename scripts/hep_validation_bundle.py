"""Canonical HEP validation bundle — single release-facing artifact.

Collects HEP surface matrix snapshot, per-slice evidence links, parity
contracts, and gate references into one JSON + Markdown bundle.

Usage:
    python -m scripts.hep_validation_bundle --out-json tmp/hep_validation_bundle.json --out-md tmp/hep_validation_bundle.md
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_hep_matrix() -> dict[str, Any]:
    path = repo_root() / "hep_surface_matrix_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _load_release_matrix() -> dict[str, Any]:
    path = repo_root() / "scripts" / "release_surface_matrix_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _slice_summary(hep: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per-owner-slice stable/research/total counts."""
    out: dict[str, dict[str, int]] = {}
    for s in hep["surfaces"]:
        owner = s.get("owner_slice", "unknown")
        if owner not in out:
            out[owner] = {"stable": 0, "research": 0, "total": 0}
        out[owner]["total"] += 1
        mc = s.get("maturity_class", "")
        if mc == "stable":
            out[owner]["stable"] += 1
        elif mc == "research":
            out[owner]["research"] += 1
    return out


def _evidence_links(release: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-release-surface evidence links."""
    entries = []
    for s in release["surfaces"]:
        if not s.get("required_for_release"):
            continue
        entries.append(
            {
                "id": s["id"],
                "name": s["name"],
                "make_target": s.get("make_target", ""),
                "workflow_job": s.get("workflow_job", ""),
                "docs": s.get("docs", []),
            }
        )
    return entries


def _support_matrix_refs(hep: dict[str, Any]) -> list[str]:
    """Unique support matrix refs across all stable surfaces."""
    refs: set[str] = set()
    for s in hep["surfaces"]:
        ref = s.get("support_matrix_ref")
        if ref:
            refs.add(ref)
    return sorted(refs)


def build_bundle(
    hep: dict[str, Any],
    release: dict[str, Any],
) -> dict[str, Any]:
    slices = _slice_summary(hep)
    total_stable = sum(v["stable"] for v in slices.values())
    total_research = sum(v["research"] for v in slices.values())
    total = sum(v["total"] for v in slices.values())

    return {
        "schema_version": "nextstat.hep_validation_bundle.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "summary": {
            "total_surfaces": total,
            "stable": total_stable,
            "research": total_research,
            "all_stable": total_stable == total and total_research == 0,
        },
        "per_slice": {
            owner: {
                "stable": counts["stable"],
                "research": counts["research"],
                "total": counts["total"],
                "fully_stable": counts["research"] == 0,
            }
            for owner, counts in sorted(slices.items())
        },
        "release_gates": _evidence_links(release),
        "support_matrix_refs": _support_matrix_refs(hep),
        "parity_contracts": {
            "histfactory": {
                "reference": "pyhf",
                "twice_nll_atol": 1e-8,
                "twice_nll_rtol": 1e-6,
                "param_value_atol": 2e-4,
                "param_uncertainty_atol": 5e-4,
                "gradient_atol": 1e-6,
                "gradient_rtol": 1e-4,
            },
            "simplified_likelihood": {
                "reference": "pyhf (reduced-model)",
                "note": "same fidelity tolerances as histfactory core",
            },
            "gvm": {
                "reference": "literature-backed combinations",
                "solvers": ["auto", "numerical-paper", "analytic-perturbative", "numerical"],
            },
            "hepdata": {
                "reference": "HEPData repository API",
                "note": "deterministic ingest parity with upstream YAML/JSON",
            },
        },
    }


def render_markdown(bundle: dict[str, Any]) -> str:
    s = bundle["summary"]
    lines = [
        "# HEP Validation Bundle",
        "",
        f"Generated: `{bundle['generated_at_utc']}`",
        "",
        "## Summary",
        "",
        f"- **{s['stable']}/{s['total_surfaces']}** stable surfaces",
        f"- **{s['research']}** research surfaces",
        f"- All stable: **{'YES' if s['all_stable'] else 'NO'}**",
        "",
        "## Per-Slice Breakdown",
        "",
        "| Owner Slice | Stable | Research | Total | Fully Stable |",
        "| --- | --- | --- | --- | --- |",
    ]
    for owner, counts in sorted(bundle["per_slice"].items()):
        fs = "YES" if counts["fully_stable"] else "NO"
        lines.append(
            f"| {owner} | {counts['stable']} | {counts['research']} | {counts['total']} | {fs} |"
        )

    lines.extend(["", "## Release Gates", ""])
    for gate in bundle["release_gates"]:
        lines.append(f"- **{gate['name']}** (`{gate['id']}`)")
        if gate["make_target"]:
            lines.append(f"  - make target: `{gate['make_target']}`")
        if gate["workflow_job"]:
            lines.append(f"  - workflow job: `{gate['workflow_job']}`")
        for doc in gate.get("docs", []):
            lines.append(f"  - [{Path(doc).name}](/{doc})")

    lines.extend(["", "## Support Matrices", ""])
    for ref in bundle["support_matrix_refs"]:
        lines.append(f"- [{Path(ref).name}](/{ref})")

    lines.extend(["", "## Parity Contracts", ""])
    for name, contract in bundle["parity_contracts"].items():
        lines.append(f"### {name}")
        lines.append("")
        for k, v in contract.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build canonical HEP validation bundle.")
    parser.add_argument("--out-json", default="tmp/hep_validation_bundle.json", help="Output JSON path")
    parser.add_argument("--out-md", default="tmp/hep_validation_bundle.md", help="Output Markdown path")
    parser.add_argument("--check", action="store_true", help="Validate and exit")
    args = parser.parse_args()

    hep = _load_hep_matrix()
    release = _load_release_matrix()
    bundle = build_bundle(hep, release)

    if args.check:
        if not bundle["summary"]["all_stable"]:
            print(
                f"FAIL: {bundle['summary']['research']} research surface(s) remaining"
            )
            return 1
        print(
            f"OK: {bundle['summary']['stable']}/{bundle['summary']['total_surfaces']} stable"
        )
        return 0

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
