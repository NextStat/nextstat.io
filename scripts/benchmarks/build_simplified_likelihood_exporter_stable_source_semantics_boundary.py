#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _simplified_likelihood_exporter_stable_source_semantics import (
    ACCEPTANCE_DOC,
    ALIGNMENT_DOCUMENTS,
    ARTIFACT_REFERENCE_DOC,
    BOUNDARY_DOC,
    BOUNDARY_EXAMPLE,
    BOUNDARY_SCHEMA,
    CANONICAL_BOUNDARY_CLAIMS,
    CLI_DOC,
    DEFAULT_OUT_PATH,
    PROMOTION_RUNBOOK_DOC,
    PYTHON_API_DOC,
    RELEASE_NOTES_DOC,
    RESEARCH_GRADE_ONLY_OR_REJECTED,
    REQUIRED_OUTPUT_SOURCE_FORMAT,
    RUNTIME_GATE_DOC,
    RUST_API_DOC,
    SCHEMA_VERSION,
    SERVER_API_DOC,
    STABLE_REVIEW_CHECKLIST_DOC,
    SUPPORTED_CONSTRAINT_COVARIANCE_SOURCE,
    SUPPORTED_POI_SCOPE,
    SUPPORTED_SOURCE_CONSTRAINT_FAMILIES,
    SUPPORTED_SOURCE_FORMATS,
    SUPPORT_MATRIX_DOC,
)
from _simplified_likelihood_promotion_bundle import now_utc, relative_or_absolute


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_boundary(*, out_path: Path, deterministic: bool) -> dict[str, Any]:
    bundle_dir = out_path.parent
    return {
        "schema_version": SCHEMA_VERSION,
        "surface": "simplified_likelihood_exporter",
        "support_class": "research-grade",
        "target_support_class": "stable",
        "automatic_stable_promotion": False,
        "status": "published",
        "generated_at_utc": now_utc(deterministic),
        "bundle_dir": relative_or_absolute(bundle_dir),
        "future_stable_boundary": {
            "source_workspace_formats": list(SUPPORTED_SOURCE_FORMATS),
            "poi_scope": SUPPORTED_POI_SCOPE,
            "required_output_source_format": REQUIRED_OUTPUT_SOURCE_FORMAT,
            "supported_constraint_covariance_source": SUPPORTED_CONSTRAINT_COVARIANCE_SOURCE,
            "supported_source_constraint_families": list(
                SUPPORTED_SOURCE_CONSTRAINT_FAMILIES
            ),
            "source_level_nuisance_identity_preserved": False,
            "ranking_source_level_breakdown_supported": False,
            "requires_explicit_provenance": True,
            "requires_partial_bin_selection_reject": True,
            "canonical_claims": list(CANONICAL_BOUNDARY_CLAIMS),
        },
        "research_grade_only_or_rejected": list(RESEARCH_GRADE_ONLY_OR_REJECTED),
        "alignment_documents": {
            "boundary_doc": BOUNDARY_DOC,
            "acceptance_doc": ACCEPTANCE_DOC,
            "runtime_gate_doc": RUNTIME_GATE_DOC,
            "promotion_runbook_doc": PROMOTION_RUNBOOK_DOC,
            "stable_review_checklist_doc": STABLE_REVIEW_CHECKLIST_DOC,
            "support_matrix_doc": SUPPORT_MATRIX_DOC,
            "release_notes_doc": RELEASE_NOTES_DOC,
            "artifact_reference_doc": ARTIFACT_REFERENCE_DOC,
            "cli_doc": CLI_DOC,
            "python_api_doc": PYTHON_API_DOC,
            "server_api_doc": SERVER_API_DOC,
            "rust_api_doc": RUST_API_DOC,
            "schema_path": BOUNDARY_SCHEMA,
            "example_path": BOUNDARY_EXAMPLE,
        },
        "summary": {
            "status": "published",
            "alignment_document_count": len(ALIGNMENT_DOCUMENTS),
            "automatic_stable_promotion": False,
            "blocker_resolution_supported": True,
            "next_action": (
                "keep this narrow boundary aligned across docs and accepted artifacts, "
                "then take an explicit stable promotion decision separately"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the narrow future-stable source-semantics boundary for the "
            "simplified-likelihood exporter."
        )
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args(argv)

    out_path = args.out.resolve()
    boundary = build_boundary(out_path=out_path, deterministic=bool(args.deterministic))
    _write_json(out_path, boundary)
    print(
        "Exporter stable source-semantics boundary:",
        f"status={boundary['summary']['status']}",
        f"out={relative_or_absolute(out_path)}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
