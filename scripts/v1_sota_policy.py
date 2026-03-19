"""Validate and render the v1.0 SOTA policy gate (NSS-10).

This script distinguishes between:

  - the current pre-v1 state, where unresolved domains are allowed but reported
  - the enforced v1.0 state, where every domain must be `sota` or `scoped_out`

Usage:
  python3 -m scripts.v1_sota_policy --release-tag v0.10.1 --check
  python3 -m scripts.v1_sota_policy --release-tag v1.0.0 --check
  python3 -m scripts.v1_sota_policy --release-tag v0.10.1 --out-json tmp/v1_sota_policy_report.json --out-md tmp/v1_sota_policy_report.md
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.release_manifest import validate_release_tag
from scripts.sota_claim_matrix import load_matrix, validate_matrix

READY_STATUSES = {"sota", "scoped_out"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _is_v1_enforced(release_tag: str, require_ready: bool) -> bool:
    version = validate_release_tag(release_tag)
    major = int(version.split(".", 1)[0])
    return require_ready or major >= 1


def build_report(
    release_tag: str,
    repo: Path | None = None,
    *,
    require_ready: bool = False,
) -> dict[str, Any]:
    repo = repo or _repo_root()
    matrix = load_matrix(repo)
    validate_matrix(matrix, repo)

    enforced = _is_v1_enforced(release_tag, require_ready)
    domains = sorted(matrix["domains"], key=lambda entry: entry["domain"])
    unresolved = [
        {"domain": entry["domain"], "status": entry["status"]}
        for entry in domains
        if entry["status"] not in READY_STATUSES
    ]

    return {
        "schema_version": "nextstat.v1_sota_policy_report.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "release_tag": release_tag,
        "enforced": enforced,
        "ready_for_v1": not unresolved,
        "allowed_statuses_for_v1": sorted(READY_STATUSES),
        "status_counts": {
            "sota": sum(1 for entry in domains if entry["status"] == "sota"),
            "proof_pending": sum(1 for entry in domains if entry["status"] == "proof_pending"),
            "access_pending": sum(1 for entry in domains if entry["status"] == "access_pending"),
            "scoped_out": sum(1 for entry in domains if entry["status"] == "scoped_out"),
        },
        "unresolved_domains": unresolved,
        "domains": [
            {
                "domain": entry["domain"],
                "status": entry["status"],
                "scope": entry["scope"],
                "guard_doc": entry["guard_doc"],
                "proof_ref_count": len(entry["proof_refs"]),
                "access_ref_count": len(entry["access_refs"]),
            }
            for entry in domains
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v1.0 SOTA Policy Report",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Release tag: `{report['release_tag']}`",
        f"- Enforced: `{str(report['enforced']).lower()}`",
        f"- Ready for v1.0: `{str(report['ready_for_v1']).lower()}`",
        "",
        "## Status counts",
        "",
    ]

    for status, count in sorted(report["status_counts"].items()):
        lines.append(f"- `{status}`: `{count}`")

    lines.extend(["", "## Unresolved domains", ""])
    if report["unresolved_domains"]:
        for entry in report["unresolved_domains"]:
            lines.append(f"- `{entry['domain']}` (`{entry['status']}`)")
    else:
        lines.append("- none")

    lines.extend(["", "## Domain inventory", "", "| Domain | Status | Proof refs | Access refs | Guard doc |", "| --- | --- | ---: | ---: | --- |"])
    for entry in report["domains"]:
        lines.append(
            f"| `{entry['domain']}` | `{entry['status']}` | "
            f"`{entry['proof_ref_count']}` | `{entry['access_ref_count']}` | "
            f"`{entry['guard_doc']}` |"
        )
    return "\n".join(lines) + "\n"


def check_policy(
    release_tag: str,
    repo: Path | None = None,
    *,
    require_ready: bool = False,
) -> tuple[bool, str]:
    try:
        report = build_report(release_tag, repo, require_ready=require_ready)
    except ValueError as exc:
        return False, str(exc)

    unresolved = report["unresolved_domains"]
    if report["enforced"] and unresolved:
        joined = ", ".join(f"{item['domain']} ({item['status']})" for item in unresolved)
        return False, f"v1.0 blocked by unresolved domains: {joined}"

    if unresolved:
        joined = ", ".join(f"{item['domain']} ({item['status']})" for item in unresolved)
        return True, f"OK: v1.0 policy scaffolded for {release_tag}; unresolved domains: {joined}"

    return True, f"OK: {release_tag} satisfies the v1.0 SOTA policy"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and render the v1.0 SOTA policy gate.")
    parser.add_argument("--release-tag", required=True, help="Release tag in vX.Y.Z form.")
    parser.add_argument("--require-ready", action="store_true", help="Force v1-ready enforcement regardless of release tag.")
    parser.add_argument("--check", action="store_true", help="Validate the policy and exit.")
    parser.add_argument("--out-json", help="Write the policy report JSON to this path.")
    parser.add_argument("--out-md", help="Write the policy report Markdown to this path.")
    args = parser.parse_args()

    ok, message = check_policy(args.release_tag, require_ready=args.require_ready)
    if not ok:
        print(message)
        return 1

    if args.check and not args.out_json and not args.out_md:
        print(message)
        return 0

    report = build_report(args.release_tag, require_ready=args.require_ready)
    markdown = render_markdown(report)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
    if not args.out_json and not args.out_md:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
