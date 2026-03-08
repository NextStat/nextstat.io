from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _iter_unique_matches(root: Path, *patterns: str) -> list[Path]:
    seen: set[Path] = set()
    matches: list[Path] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                matches.append(path)
    return matches


def _require_first_match(root: Path, asset_name: str, patterns: list[str]) -> Path:
    matches = _iter_unique_matches(root, *patterns)
    if not matches:
        joined = ", ".join(patterns)
        raise SystemExit(f"missing required release asset {asset_name!r}; searched: {joined}")
    return matches[0]


def stage_release_assets(dist_root: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for path in _iter_unique_matches(
        dist_root,
        "**/*.whl",
        "**/*.tar.gz",
        "cli-*/nextstat-*",
    ):
        if path.parent == out_dir:
            continue
        target = out_dir / path.name
        _copy(path, target)
        copied.append(target)

    canonical_assets: list[tuple[str | None, list[str]]] = [
        ("release_candidate_bundle_manifest.json", [
            "release-candidate-bundle/release_candidate_bundle_manifest.json",
            "release-candidate-bundle/tmp/release_candidate_bundle/release_candidate_bundle_manifest.json",
        ]),
        ("release_candidate_bundle_manifest.md", [
            "release-candidate-bundle/release_candidate_bundle_manifest.md",
            "release-candidate-bundle/tmp/release_candidate_bundle/release_candidate_bundle_manifest.md",
        ]),
        ("release_manifest.json", [
            "release-candidate-bundle/release_manifest.json",
            "release-candidate-bundle/tmp/release_candidate_bundle/release_manifest.json",
        ]),
        ("release_manifest.md", [
            "release-candidate-bundle/release_manifest.md",
            "release-candidate-bundle/tmp/release_candidate_bundle/release_manifest.md",
        ]),
        ("release_manifest_v1.schema.json", [
            "release-candidate-bundle/release_manifest_v1.schema.json",
            "release-candidate-bundle/docs/schemas/releases/release_manifest_v1.schema.json",
        ]),
        ("release_candidate_bundle_v1.schema.json", [
            "release-candidate-bundle/release_candidate_bundle_v1.schema.json",
            "release-candidate-bundle/docs/schemas/releases/release_candidate_bundle_v1.schema.json",
        ]),
        (None, [
            "whitepaper/nextstat-whitepaper-*.pdf",
        ]),
        (None, [
            "whitepaper/nextstat-whitepaper-*.pdf.sha256",
        ]),
        ("apex2_master_report.json", [
            "validation-pack/artifacts/apex2_master_report.json",
        ]),
        ("validation_report.json", [
            "validation-pack/artifacts/validation_report.json",
        ]),
        ("validation_report.pdf", [
            "validation-pack/artifacts/validation_report.pdf",
        ]),
        ("validation_report_v1.schema.json", [
            "validation-pack/artifacts/validation_report_v1.schema.json",
        ]),
        ("validation_pack_manifest.json", [
            "validation-pack/artifacts/validation_pack_manifest.json",
        ]),
        ("validation_pack_manifest.sha256", [
            "validation-pack/artifacts/validation_pack_manifest.sha256",
        ]),
        ("validation_pack_manifest.sha256.bin", [
            "validation-pack/artifacts/validation_pack_manifest.sha256.bin",
        ]),
        ("snapshot_index.json", [
            "validation-pack/artifacts/snapshot_index.json",
        ]),
        ("m15_config.json", [
            "validation-pack/artifacts_m15/m15_config.json",
        ]),
        ("m15_assessment_table.json", [
            "validation-pack/artifacts_m15/m15_assessment_table.json",
        ]),
        ("m15_map.json", [
            "validation-pack/artifacts_m15/m15_map.json",
        ]),
        ("m15_mar.json", [
            "validation-pack/artifacts_m15/m15_mar.json",
        ]),
        ("m15_profile_diff_report.json", [
            "validation-pack/artifacts_m15/m15_profile_diff_report.json",
        ]),
        ("m15_profile_diff_report_v1.schema.json", [
            "validation-pack/artifacts_m15/m15_profile_diff_report_v1.schema.json",
        ]),
        ("m15_report.md", [
            "validation-pack/artifacts_m15/m15_report.md",
        ]),
        ("m15_report.pdf", [
            "validation-pack/artifacts_m15/m15_report.pdf",
        ]),
        ("m15_report.docx", [
            "validation-pack/artifacts_m15/m15_report.docx",
        ]),
        ("m15_bundle_manifest.json", [
            "validation-pack/artifacts_m15/m15_bundle_manifest.json",
        ]),
        ("m15_bundle_manifest_v1.schema.json", [
            "validation-pack/artifacts_m15/m15_bundle_manifest_v1.schema.json",
        ]),
        ("m15_bundle_manifest.sha256", [
            "validation-pack/artifacts_m15/m15_bundle_manifest.sha256",
        ]),
        ("m15_bundle_manifest.sha256.bin", [
            "validation-pack/artifacts_m15/m15_bundle_manifest.sha256.bin",
        ]),
        ("m15_snapshot_index.json", [
            "validation-pack/artifacts_m15/m15_snapshot_index.json",
        ]),
        ("m15_reporting_benchmark.json", [
            "m15-reporting-stable-surface-report/m15_reporting_benchmark.json",
            "m15-reporting-stable-surface-report/tmp/m15_reporting_stable_surface/m15_reporting_benchmark.json",
        ]),
        ("m15_reporting_benchmark.md", [
            "m15-reporting-stable-surface-report/m15_reporting_benchmark.md",
            "m15-reporting-stable-surface-report/tmp/m15_reporting_stable_surface/m15_reporting_benchmark.md",
        ]),
        ("m15_reporting_compare.json", [
            "m15-reporting-stable-surface-report/m15_reporting_compare.json",
            "m15-reporting-stable-surface-report/tmp/m15_reporting_stable_surface/m15_reporting_compare.json",
        ]),
        ("apex2_simplified_likelihood_report.json", [
            "simplified-likelihood-stable-surface-report/apex2_simplified_likelihood_report.json",
            "simplified-likelihood-stable-surface-report/tmp/simplified-likelihood-stable-surface/apex2_simplified_likelihood_report.json",
        ]),
        ("export_benchmark_snapshot_report.json", [
            "simplified-likelihood-exporter-surface-report/export_benchmark_snapshot_report.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_benchmark_snapshot_report.json",
        ]),
        ("export_public_validation_report.json", [
            "simplified-likelihood-exporter-surface-report/export_public_validation_report.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_export_benchmarks/nextstat-bench/current/export_public_validation_report.json",
        ]),
        ("promotion_evidence.json", [
            "simplified-likelihood-exporter-surface-report/promotion_bundle/promotion_evidence.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence.json",
        ]),
        ("promotion_evidence_check.json", [
            "simplified-likelihood-exporter-surface-report/promotion_bundle/promotion_evidence_check.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_evidence_check.json",
        ]),
        ("promotion_bundle_promotion_report.json", [
            "simplified-likelihood-exporter-surface-report/promotion_bundle_promotion_report.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/promotion_bundle_promotion_report.json",
        ]),
        ("stable_review_assessment.json", [
            "simplified-likelihood-exporter-surface-report/stable_review_assessment.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_review_assessment.json",
        ]),
        ("stable_evidence_policy.json", [
            "simplified-likelihood-exporter-surface-report/stable_evidence_policy.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_policy.json",
        ]),
        ("stable_evidence_freshness_report.json", [
            "simplified-likelihood-exporter-surface-report/stable_evidence_freshness_report.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_evidence_freshness_report.json",
        ]),
        ("stable_source_semantics_boundary.json", [
            "simplified-likelihood-exporter-surface-report/stable_source_semantics_boundary.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_source_semantics_boundary.json",
        ]),
        ("stable_candidate_blocker_matrix.json", [
            "simplified-likelihood-exporter-surface-report/stable_candidate_blocker_matrix.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_blocker_matrix.json",
        ]),
        ("stable_candidate_review_packet.json", [
            "simplified-likelihood-exporter-surface-report/stable_candidate_review_packet.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_candidate_review_packet.json",
        ]),
        ("stable_promotion_decision.json", [
            "simplified-likelihood-exporter-surface-report/stable_promotion_decision.json",
            "simplified-likelihood-exporter-surface-report/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json",
            "simplified-likelihood-exporter-surface-report/benchmarks/artifacts/simplified_likelihood_exporter_promotion_bundles/nextstat-bench/accepted/stable_promotion_decision.json",
        ]),
    ]

    for asset_name, patterns in canonical_assets:
        display_name = asset_name or patterns[0]
        src = _require_first_match(dist_root, display_name, patterns)
        target = out_dir / (asset_name or src.name)
        _copy(src, target)
        copied.append(target)

    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage unique GitHub Release assets from downloaded workflow artifacts.")
    parser.add_argument("--dist-root", type=Path, default=_repo_root() / "dist")
    parser.add_argument("--out-dir", type=Path, default=_repo_root() / "dist" / "release-assets")
    args = parser.parse_args()

    copied = stage_release_assets(args.dist_root.resolve(), args.out_dir.resolve())
    print(f"staged {len(copied)} release assets into {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
