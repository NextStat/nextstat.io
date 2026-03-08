from __future__ import annotations

from pathlib import Path

from scripts.release_stage_assets import stage_release_assets


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_release_asset_staging_chooses_canonical_sources_for_duplicate_basenames(
    tmp_path: Path,
) -> None:
    dist = tmp_path / "dist"
    out = dist / "release-assets"

    _write(dist / "wheels-x86_64-unknown-linux-gnu" / "nextstat-0.10.0-cp312.whl", "wheel")
    _write(dist / "sdist" / "nextstat-0.10.0.tar.gz", "sdist")
    _write(dist / "cli-x86_64-unknown-linux-gnu" / "nextstat-x86_64-unknown-linux-gnu", "bin")

    _write(
        dist / "release-candidate-bundle" / "tmp" / "release_candidate_bundle" / "release_candidate_bundle_manifest.json",
        "bundle-json",
    )
    _write(
        dist / "release-candidate-bundle" / "tmp" / "release_candidate_bundle" / "release_candidate_bundle_manifest.md",
        "bundle-md",
    )
    _write(
        dist / "release-candidate-bundle" / "tmp" / "release_candidate_bundle" / "release_manifest.json",
        "manifest-json",
    )
    _write(
        dist / "release-candidate-bundle" / "tmp" / "release_candidate_bundle" / "release_manifest.md",
        "manifest-md",
    )
    _write(
        dist / "release-candidate-bundle" / "docs" / "schemas" / "releases" / "release_manifest_v1.schema.json",
        "schema",
    )
    _write(
        dist / "release-candidate-bundle" / "docs" / "schemas" / "releases" / "release_candidate_bundle_v1.schema.json",
        "bundle-schema",
    )
    _write(dist / "whitepaper" / "nextstat-whitepaper-v0.10.0.pdf", "pdf")
    _write(dist / "whitepaper" / "nextstat-whitepaper-v0.10.0.pdf.sha256", "sha")

    _write(dist / "validation-pack" / "artifacts" / "validation_report.json", "base-report")
    _write(dist / "validation-pack" / "artifacts" / "validation_report.pdf", "base-pdf")
    _write(dist / "validation-pack" / "artifacts" / "validation_report_v1.schema.json", "base-schema")
    _write(dist / "validation-pack" / "artifacts" / "validation_pack_manifest.json", "base-manifest")
    _write(dist / "validation-pack" / "artifacts" / "validation_pack_manifest.sha256", "base-sha")
    _write(dist / "validation-pack" / "artifacts" / "validation_pack_manifest.sha256.bin", "base-sha-bin")
    _write(dist / "validation-pack" / "artifacts" / "apex2_master_report.json", "base-master")
    _write(dist / "validation-pack" / "artifacts" / "snapshot_index.json", "base-snapshot")

    _write(dist / "validation-pack" / "artifacts_m15" / "validation_report.json", "m15-report")
    _write(dist / "validation-pack" / "artifacts_m15" / "validation_report.pdf", "m15-pdf")
    _write(dist / "validation-pack" / "artifacts_m15" / "validation_pack_manifest.json", "m15-manifest")
    _write(dist / "validation-pack" / "artifacts_m15" / "apex2_master_report.json", "m15-master")
    _write(dist / "validation-pack" / "artifacts_m15" / "m15_config.json", "m15-config")
    _write(dist / "validation-pack" / "artifacts_m15" / "m15_assessment_table.json", "m15-table")
    _write(dist / "validation-pack" / "artifacts_m15" / "m15_map.json", "m15-map")
    _write(dist / "validation-pack" / "artifacts_m15" / "m15_mar.json", "m15-mar")
    _write(dist / "validation-pack" / "artifacts_m15" / "m15_profile_diff_report.json", "m15-diff")
    _write(
        dist / "validation-pack" / "artifacts_m15" / "m15_profile_diff_report_v1.schema.json",
        "m15-diff-schema",
    )
    _write(dist / "validation-pack" / "artifacts_m15" / "m15_report.md", "m15-md")
    _write(dist / "validation-pack" / "artifacts_m15" / "m15_report.pdf", "m15-pdf-report")
    _write(dist / "validation-pack" / "artifacts_m15" / "m15_report.docx", "m15-docx")
    _write(dist / "validation-pack" / "artifacts_m15" / "m15_bundle_manifest.json", "m15-bundle")
    _write(
        dist / "validation-pack" / "artifacts_m15" / "m15_bundle_manifest_v1.schema.json",
        "m15-bundle-schema",
    )
    _write(dist / "validation-pack" / "artifacts_m15" / "m15_bundle_manifest.sha256", "m15-bundle-sha")
    _write(
        dist / "validation-pack" / "artifacts_m15" / "m15_bundle_manifest.sha256.bin",
        "m15-bundle-sha-bin",
    )
    _write(dist / "validation-pack" / "artifacts_m15" / "m15_snapshot_index.json", "m15-snapshot")

    _write(
        dist / "simplified-likelihood-stable-surface-report" / "apex2_simplified_likelihood_report.json",
        "sl-stable",
    )
    exporter_current = (
        dist
        / "simplified-likelihood-exporter-surface-report"
        / "simplified_likelihood_export_benchmarks"
        / "nextstat-bench"
        / "current"
    )
    exporter_accepted = (
        dist
        / "simplified-likelihood-exporter-surface-report"
        / "simplified_likelihood_exporter_promotion_bundles"
        / "nextstat-bench"
        / "accepted"
    )
    _write(exporter_current / "export_benchmark_snapshot_report.json", "export-bench")
    _write(exporter_current / "export_public_validation_report.json", "export-validate")
    _write(
        exporter_accepted / "promotion_evidence.json",
        "promotion-accepted",
    )
    _write(exporter_accepted / "promotion_evidence_check.json", "promotion-check")
    _write(exporter_accepted / "promotion_bundle_promotion_report.json", "promotion-report")
    _write(exporter_accepted / "stable_review_assessment.json", "stable-review")
    _write(exporter_accepted / "stable_evidence_policy.json", "stable-policy")
    _write(exporter_accepted / "stable_evidence_freshness_report.json", "stable-freshness")
    _write(exporter_accepted / "stable_source_semantics_boundary.json", "stable-boundary")
    _write(exporter_accepted / "stable_candidate_blocker_matrix.json", "stable-blockers")
    _write(exporter_accepted / "stable_candidate_review_packet.json", "stable-review-packet")
    _write(exporter_accepted / "stable_promotion_decision.json", "stable-decision")
    _write(exporter_current / "snapshot_index.json", "exporter-snapshot")

    _write(
        dist / "m15-reporting-stable-surface-report" / "m15_reporting_benchmark.json",
        "m15-bench-json",
    )
    _write(
        dist / "m15-reporting-stable-surface-report" / "m15_reporting_benchmark.md",
        "m15-bench-md",
    )
    _write(
        dist / "m15-reporting-stable-surface-report" / "m15_reporting_compare.json",
        "m15-compare",
    )

    staged = stage_release_assets(dist, out)
    names = {path.name for path in staged}

    assert "validation_report.json" in names
    assert (out / "validation_report.json").read_text(encoding="utf-8") == "base-report"
    assert (out / "snapshot_index.json").read_text(encoding="utf-8") == "base-snapshot"
    assert (out / "promotion_evidence.json").read_text(encoding="utf-8") == "promotion-accepted"
    assert (out / "m15_config.json").read_text(encoding="utf-8") == "m15-config"
    assert (out / "nextstat-whitepaper-v0.10.0.pdf").read_text(encoding="utf-8") == "pdf"
    assert (out / "release_candidate_bundle_v1.schema.json").read_text(encoding="utf-8") == "bundle-schema"
    assert len(names) == len(staged)
