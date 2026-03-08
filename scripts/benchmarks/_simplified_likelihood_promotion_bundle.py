from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

BUNDLE_SCHEMA_VERSION = "nextstat_simplified_likelihood_promotion_evidence_bundle_v0"
CHECK_SCHEMA_VERSION = "nextstat_simplified_likelihood_promotion_evidence_check_v0"
PROMOTION_REPORT_SCHEMA_VERSION = (
    "nextstat_simplified_likelihood_promotion_bundle_promotion_report_v0"
)
DEFAULT_GENERATED_AT = "1970-01-01T00:00:00Z"

REQUIRED_BENCHMARK_HOST = "nextstat-bench"
REQUIRED_MIN_END_TO_END_UPPER_LIMIT_SPEEDUP = 10.0
PROMOTION_ARTIFACT_SUITE = "simplified_likelihood_promotion_bundle"
DEFAULT_ACCEPTED_BUNDLE_DIR = (
    REPO_ROOT
    / "benchmarks"
    / "artifacts"
    / "simplified_likelihood_promotion_bundles"
    / REQUIRED_BENCHMARK_HOST
    / "accepted"
)
DEFAULT_ACCEPTED_HISTORY_DIR = DEFAULT_ACCEPTED_BUNDLE_DIR.parent / "history"

PROMOTED_COMMANDS = ["audit", "fit", "hypotest", "upper-limit", "scan"]
RESEARCH_GRADE_SURFACES = [
    "significance",
    "hypotest-toys",
    "ranking",
    "derive-export",
    "covariance-source-semantics",
]

STATIC_ARTIFACTS: list[dict[str, str]] = [
    {
        "role": "acceptance_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-stable-surface-acceptance-2026-03-08.md",
    },
    {
        "role": "support_matrix_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md",
    },
    {
        "role": "release_notes_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md",
    },
    {
        "role": "benchmark_snapshot_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-benchmark-snapshot-2026-03-08.md",
    },
    {
        "role": "promotion_runbook_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-promotion-runbook-2026-03-08.md",
    },
    {
        "role": "release_checklist_doc",
        "kind": "doc",
        "source_path": "docs/benchmarks/simplified-likelihood-release-pr-checklist-2026-03-08.md",
    },
    {
        "role": "artifact_reference_doc",
        "kind": "doc",
        "source_path": "docs/references/simplified-likelihood-artifacts.md",
    },
    {
        "role": "input_schema",
        "kind": "schema",
        "source_path": "docs/schemas/hep/simplified_likelihood_v0.schema.json",
    },
    {
        "role": "audit_schema",
        "kind": "schema",
        "source_path": "docs/schemas/hep/simplified_likelihood_audit_v0.schema.json",
    },
    {
        "role": "apex2_report_schema",
        "kind": "schema",
        "source_path": "docs/schemas/apex2/simplified_likelihood_report_v0.schema.json",
    },
    {
        "role": "public_fixture_catalog_schema",
        "kind": "schema",
        "source_path": "docs/schemas/apex2/simplified_likelihood_public_fixture_catalog_v0.schema.json",
    },
    {
        "role": "input_example",
        "kind": "example",
        "source_path": "docs/specs/hep/simplified_likelihood_v0.example.json",
    },
    {
        "role": "covariance_example",
        "kind": "example",
        "source_path": "docs/specs/hep/simplified_likelihood_covariance_public_v0.example.json",
    },
    {
        "role": "derived_example",
        "kind": "example",
        "source_path": "docs/specs/hep/simplified_likelihood_derived_from_workspace_v0.example.json",
    },
    {
        "role": "audit_example",
        "kind": "example",
        "source_path": "docs/specs/hep/simplified_likelihood_audit_v0.example.json",
    },
    {
        "role": "public_fixture_catalog_example",
        "kind": "example",
        "source_path": "docs/specs/apex2_simplified_likelihood_public_fixture_catalog_v0.example.json",
    },
]

REQUIRED_ROLES = ["benchmark_artifact", *[artifact["role"] for artifact in STATIC_ARTIFACTS]]


def now_utc(deterministic: bool) -> str:
    if deterministic:
        return DEFAULT_GENERATED_AT
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def relative_or_absolute(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def bundle_path_for(source_path: Path, *, benchmark: bool) -> Path:
    if benchmark:
        return Path("files/benchmark") / source_path.name
    rel = Path(relative_or_absolute(source_path))
    return Path("files") / rel


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def derive_stamp_from_path(path: Path) -> str | None:
    for candidate in [path.name, path.parent.name, *path.parts]:
        match = re.search(r"(\d{8}T\d{6}Z)", candidate)
        if match:
            return match.group(1)
    return None
