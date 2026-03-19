"""Public stable-status governance tests (RWS-02).

Ensures that no public document can claim ``status: stable`` (or ``stable-first``)
in its YAML frontmatter without a corresponding row in
``repo_surface_matrix_v1.json``.

Rules enforced:
- Every public stable doc has a repo-matrix row (support_contract_ref match).
- Every stable tutorial's parent slice has at least one stable *runtime* surface.
- No orphan stable claims (docs saying stable but matrix row says research/internal).

Exclusions:
- ``docs/ru/**`` — Russian translations mirror English originals.
- ``**/README.md`` — Index/navigation pages, not surfaces.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

STABLE_STATUSES = {"stable", "stable-first"}

# Paths excluded from governance (translations, known index pages)
EXCLUDED_PREFIXES = ("docs/ru/",)
EXCLUDED_PATHS = {
    "docs/tutorials/README.md",
    "docs/quickstarts/README.md",
}

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)
_STATUS_RE = re.compile(r"^status:\s*(.+)$", re.MULTILINE)


def _parse_status(path: Path) -> str | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    fm = _FRONTMATTER_RE.match(text)
    if not fm:
        return None
    m = _STATUS_RE.search(fm.group(1))
    if not m:
        return None
    return m.group(1).strip().strip('"').strip("'")


def _find_stable_docs() -> list[tuple[str, str]]:
    """Return (relative_path, status) for all governed stable docs."""
    results = []
    for md in sorted(REPO.glob("docs/**/*.md")):
        rel = str(md.relative_to(REPO))
        if any(rel.startswith(p) for p in EXCLUDED_PREFIXES):
            continue
        if rel in EXCLUDED_PATHS:
            continue
        status = _parse_status(md)
        if status and status in STABLE_STATUSES:
            results.append((rel, status))
    return results


@pytest.fixture(scope="module")
def repo_matrix() -> dict:
    return json.loads(
        (REPO / "repo_surface_matrix_v1.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def stable_docs() -> list[tuple[str, str]]:
    return _find_stable_docs()


@pytest.fixture(scope="module")
def matrix_refs(repo_matrix: dict) -> set[str]:
    return {
        s["support_contract_ref"]
        for s in repo_matrix["surfaces"]
        if s.get("support_contract_ref")
    }


# ── Core governance: every stable doc has a matrix row ───────────────────


def test_at_least_one_stable_doc_found(stable_docs: list[tuple[str, str]]) -> None:
    assert len(stable_docs) > 0, "No public stable docs found — governance is vacuous"


def test_every_stable_doc_has_matrix_row(
    stable_docs: list[tuple[str, str]], matrix_refs: set[str]
) -> None:
    missing = [doc for doc, _ in stable_docs if doc not in matrix_refs]
    assert not missing, (
        f"{len(missing)} public stable doc(s) without repo-matrix row:\n"
        + "\n".join(f"  - {d}" for d in missing)
    )


# ── No orphan claims: matrix row status must agree with doc status ───────


def test_no_stable_doc_with_nonstable_matrix_row(
    stable_docs: list[tuple[str, str]], repo_matrix: dict
) -> None:
    by_ref = {
        s["support_contract_ref"]: s
        for s in repo_matrix["surfaces"]
        if s.get("support_contract_ref")
    }
    mismatches = []
    for doc, _ in stable_docs:
        row = by_ref.get(doc)
        if row and row["public_status"] not in ("stable",):
            mismatches.append(
                f"{doc}: doc says stable but matrix says {row['public_status']!r}"
            )
    assert not mismatches, (
        f"{len(mismatches)} stable doc(s) with non-stable matrix status:\n"
        + "\n".join(f"  - {m}" for m in mismatches)
    )


# ── Tutorial governance: parent slice must be stable ─────────────────────


def test_stable_tutorials_have_stable_parent_slice(
    repo_matrix: dict,
) -> None:
    tutorials = [
        s
        for s in repo_matrix["surfaces"]
        if s["surface_kind"] == "tutorial" and s["public_status"] == "stable"
    ]
    # For each tutorial, check that its owner_slice has at least one
    # stable runtime surface in the same domain
    runtime_stable = {
        (s["domain"], s["owner_slice"])
        for s in repo_matrix["surfaces"]
        if s["surface_kind"] == "runtime" and s["public_status"] == "stable"
    }
    orphans = []
    for t in tutorials:
        key = (t["domain"], t["owner_slice"])
        if key not in runtime_stable:
            orphans.append(
                f"{t['surface_id']}: tutorial is stable but no stable runtime "
                f"surface for ({t['domain']}, {t['owner_slice']})"
            )
    assert not orphans, (
        f"{len(orphans)} stable tutorial(s) with non-stable parent slice:\n"
        + "\n".join(f"  - {o}" for o in orphans)
    )


# ── Migration state tracking ────────────────────────────────────────────


def test_stable_not_release_governed_count_is_bounded(
    repo_matrix: dict,
) -> None:
    migration_rows = [
        s
        for s in repo_matrix["surfaces"]
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    # This is a migration-only state per ADR. Track the count so it
    # only goes down, never up. Current ceiling based on day-1 import.
    # Day-1 ceiling after RWS-02 bulk import. This number must only go DOWN
    # as RWS-03..08 promote domains into release governance.
    assert len(migration_rows) <= 90, (
        f"Too many stable+not_release_governed rows ({len(migration_rows)}). "
        f"This count should only decrease as domains get promoted."
    )


def test_every_migration_row_has_owner_and_domain(
    repo_matrix: dict,
) -> None:
    migration_rows = [
        s
        for s in repo_matrix["surfaces"]
        if s["public_status"] == "stable"
        and s["release_status"] == "not_release_governed"
    ]
    bad = [
        s["surface_id"]
        for s in migration_rows
        if not s.get("owner_slice") or not s.get("domain")
    ]
    assert not bad, (
        f"Migration rows missing owner/domain: {bad}"
    )
