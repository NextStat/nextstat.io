from __future__ import annotations

import json
from pathlib import Path

from scripts.v1_sota_policy import build_report, check_policy, render_markdown


REPO = Path(__file__).resolve().parents[2]
CLAIM_MATRIX = REPO / ".internal" / "claims" / "nextstat_sota_claim_matrix_v1.json"
V1_ALLOWED_STATUSES = {"sota", "scoped_out"}


def _load_domains() -> list[dict]:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    return data["domains"]


def test_v1_policy_gate_has_no_remaining_blockers() -> None:
    domains = _load_domains()
    unresolved = [
        entry["domain"]
        for entry in domains
        if entry["status"] not in V1_ALLOWED_STATUSES
    ]
    assert unresolved == []


def test_v1_policy_no_unknown_statuses() -> None:
    allowed = {"sota", "proof_pending", "access_pending", "scoped_out"}
    domains = _load_domains()
    for entry in domains:
        assert entry["status"] in allowed, (
            f"{entry['domain']}: unknown status {entry['status']!r}"
        )


def test_v1_sota_domains_have_full_evidence() -> None:
    domains = _load_domains()
    for entry in domains:
        if entry["status"] != "sota":
            continue
        assert entry["proof_refs"], f"{entry['domain']}: sota but no proof_refs"
        assert entry["access_refs"], f"{entry['domain']}: sota but no access_refs"
        assert entry["guard_snippets"], f"{entry['domain']}: sota but no guard_snippets"


def test_v1_scoped_out_domains_have_guard() -> None:
    domains = _load_domains()
    for entry in domains:
        if entry["status"] != "scoped_out":
            continue
        assert entry["guard_snippets"], (
            f"{entry['domain']}: scoped_out but no guard_snippets"
        )


def test_v1_policy_report_for_current_release_is_ready() -> None:
    report = build_report("v0.10.1", REPO)
    assert report["schema_version"] == "nextstat.v1_sota_policy_report.v1"
    assert report["enforced"] is False
    assert report["ready_for_v1"] is True
    assert report["unresolved_domains"] == []


def test_v1_policy_check_passes_for_pre_v1_release() -> None:
    ok, message = check_policy("v0.10.1", REPO)
    assert ok, message
    assert "v0.10.1 satisfies the v1.0 SOTA policy" in message


def test_v1_policy_check_passes_for_v1_release() -> None:
    ok, message = check_policy("v1.0.0", REPO)
    assert ok, message
    assert "v1.0.0 satisfies the v1.0 SOTA policy" in message


def test_v1_policy_check_can_be_force_enforced() -> None:
    ok, message = check_policy("v0.10.1", REPO, require_ready=True)
    assert ok, message
    assert "v0.10.1 satisfies the v1.0 SOTA policy" in message


def test_v1_policy_markdown_mentions_no_unresolved_domains() -> None:
    md = render_markdown(build_report("v0.10.1", REPO))
    assert "# v1.0 SOTA Policy Report" in md
    assert "## Unresolved domains" in md
    assert "- none" in md
