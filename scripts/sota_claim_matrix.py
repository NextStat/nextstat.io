"""Validate and render the public SOTA claim matrix.

The claim matrix is the narrow public truth for what NextStat can currently
claim as SOTA, what remains proof-pending, what is access-pending, and what is
deliberately outside the stable product boundary.

Usage:
  python3 -m scripts.sota_claim_matrix --check
  python3 -m scripts.sota_claim_matrix --out-json tmp/sota_claim_matrix_report.json --out-md tmp/sota_claim_matrix_report.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = {"sota", "proof_pending", "access_pending", "scoped_out"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def matrix_path(repo: Path | None = None) -> Path:
    repo = repo or _repo_root()
    return repo / ".internal" / "claims" / "nextstat_sota_claim_matrix_v1.json"


def load_matrix(repo: Path | None = None) -> dict[str, Any]:
    return json.loads(matrix_path(repo).read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_matrix(matrix: dict[str, Any], repo: Path | None = None) -> None:
    repo = repo or _repo_root()
    _require(
        matrix.get("schema_version") == "nextstat.sota_claim_matrix.v1",
        "schema_version must be 'nextstat.sota_claim_matrix.v1'",
    )

    domains = matrix.get("domains")
    _require(isinstance(domains, list) and domains, "domains must be a non-empty list")

    seen_domains: set[str] = set()
    for entry in domains:
        _require(isinstance(entry, dict), "each domain entry must be an object")
        domain = entry.get("domain")
        _require(isinstance(domain, str) and domain, "domain must be a non-empty string")
        _require(domain not in seen_domains, f"duplicate domain: {domain}")
        seen_domains.add(domain)

        status = entry.get("status")
        _require(status in ALLOWED_STATUSES, f"{domain}: invalid status {status!r}")

        scope = entry.get("scope")
        _require(isinstance(scope, str) and scope, f"{domain}: scope must be a non-empty string")

        guard_doc = entry.get("guard_doc")
        _require(isinstance(guard_doc, str) and guard_doc, f"{domain}: guard_doc must be a non-empty string")
        guard_doc_path = repo / guard_doc
        _require(guard_doc_path.exists(), f"{domain}: missing guard_doc {guard_doc}")

        proof_refs = entry.get("proof_refs")
        access_refs = entry.get("access_refs")
        guard_snippets = entry.get("guard_snippets")
        _require(isinstance(proof_refs, list), f"{domain}: proof_refs must be a list")
        _require(isinstance(access_refs, list), f"{domain}: access_refs must be a list")
        _require(
            isinstance(guard_snippets, list) and guard_snippets,
            f"{domain}: guard_snippets must be a non-empty list",
        )

        for ref_group_name, refs in (("proof_refs", proof_refs), ("access_refs", access_refs)):
            for ref in refs:
                _require(isinstance(ref, str) and ref, f"{domain}: invalid {ref_group_name} entry")
                _require((repo / ref).exists(), f"{domain}: missing ref {ref}")

        guard_text = guard_doc_path.read_text(encoding="utf-8")
        for snippet in guard_snippets:
            _require(isinstance(snippet, str) and snippet, f"{domain}: invalid guard_snippets entry")
            _require(snippet in guard_text, f"{domain}: missing guard snippet {snippet!r}")

        if status == "sota":
            _require(proof_refs, f"{domain}: sota domains must have proof_refs")
            _require(access_refs, f"{domain}: sota domains must have access_refs")


def build_report(repo: Path | None = None) -> dict[str, Any]:
    repo = repo or _repo_root()
    matrix = load_matrix(repo)
    validate_matrix(matrix, repo)

    domains = matrix["domains"]
    status_counts = Counter(entry["status"] for entry in domains)
    by_status: dict[str, list[str]] = {status: [] for status in sorted(ALLOWED_STATUSES)}
    for entry in domains:
        by_status[entry["status"]].append(entry["domain"])
    for domains_for_status in by_status.values():
        domains_for_status.sort()

    return {
        "schema_version": "nextstat.sota_claim_matrix_report.v1",
        "claim_matrix_schema_version": matrix["schema_version"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "summary": {
            "total_domains": len(domains),
            "status_counts": dict(sorted(status_counts.items())),
            "sota_domains": by_status["sota"],
            "proof_pending_domains": by_status["proof_pending"],
            "access_pending_domains": by_status["access_pending"],
            "scoped_out_domains": by_status["scoped_out"],
        },
        "domains": [
            {
                "domain": entry["domain"],
                "status": entry["status"],
                "scope": entry["scope"],
                "proof_ref_count": len(entry["proof_refs"]),
                "access_ref_count": len(entry["access_refs"]),
                "guard_doc": entry["guard_doc"],
            }
            for entry in sorted(domains, key=lambda item: item["domain"])
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# SOTA Claim Matrix Report",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Claim matrix schema: `{report['claim_matrix_schema_version']}`",
        f"- Total domains: `{summary['total_domains']}`",
        "",
        "## Status summary",
        "",
    ]

    for status, count in summary["status_counts"].items():
        lines.append(f"- `{status}`: `{count}`")

    lines.extend(
        [
            "",
            "## Current boundary",
            "",
            f"- `sota`: {', '.join(f'`{d}`' for d in summary['sota_domains']) or 'none'}",
            f"- `proof_pending`: {', '.join(f'`{d}`' for d in summary['proof_pending_domains']) or 'none'}",
            f"- `access_pending`: {', '.join(f'`{d}`' for d in summary['access_pending_domains']) or 'none'}",
            f"- `scoped_out`: {', '.join(f'`{d}`' for d in summary['scoped_out_domains']) or 'none'}",
            "",
            "## Domain inventory",
            "",
            "| Domain | Status | Proof refs | Access refs | Guard doc |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )

    for entry in report["domains"]:
        lines.append(
            f"| `{entry['domain']}` | `{entry['status']}` | `{entry['proof_ref_count']}` | "
            f"`{entry['access_ref_count']}` | `{entry['guard_doc']}` |"
        )

    return "\n".join(lines) + "\n"


def check_matrix(repo: Path | None = None) -> tuple[bool, str]:
    try:
        report = build_report(repo)
    except ValueError as exc:
        return False, str(exc)

    summary = report["summary"]
    return True, (
        "OK: "
        f"{summary['status_counts'].get('sota', 0)}/{summary['total_domains']} "
        "domains currently claim SOTA"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and render the public SOTA claim matrix.")
    parser.add_argument("--check", action="store_true", help="Validate the matrix and exit.")
    parser.add_argument("--out-json", help="Write the SOTA claim matrix report JSON to this path.")
    parser.add_argument("--out-md", help="Write the SOTA claim matrix report Markdown to this path.")
    args = parser.parse_args()

    ok, message = check_matrix()
    if not ok:
        print(message)
        return 1

    if args.check and not args.out_json and not args.out_md:
        print(message)
        return 0

    report = build_report()
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
