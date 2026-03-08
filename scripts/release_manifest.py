from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.release_surface_matrix import load_manifest as load_release_surface_matrix_manifest


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def schema_path() -> Path:
    return repo_root() / "docs" / "schemas" / "releases" / "release_manifest_v1.schema.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _extract_quoted_version(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("version = "):
            return stripped.split('"')[1]
    raise ValueError(f"missing version in {path}")


def _extract_nextstat_cli_pin(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "nextstat-cli==" in line:
            after = line.split("nextstat-cli==", 1)[1]
            return after.split('"', 1)[0]
    raise ValueError(f"missing nextstat-cli pin in {path}")


def validate_release_tag(release_tag: str) -> str:
    _require(release_tag.startswith("v"), "release_tag must start with 'v'")
    version = release_tag[1:]
    _require(version.count(".") >= 2, "release_tag must be in vX.Y.Z form")
    return version


def validate_mode(mode: str) -> None:
    _require(mode in {"prepare", "publish"}, "mode must be 'prepare' or 'publish'")


def current_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def aligned_versions(version: str) -> dict[str, str]:
    cargo_version = _extract_quoted_version(repo_root() / "Cargo.toml")
    ns_py_version = _extract_quoted_version(repo_root() / "bindings" / "ns-py" / "pyproject.toml")
    ns_cli_py_version = _extract_quoted_version(
        repo_root() / "bindings" / "ns-cli-py" / "pyproject.toml"
    )
    ns_cli_pin = _extract_nextstat_cli_pin(
        repo_root() / "bindings" / "ns-py" / "pyproject.toml"
    )
    for label, candidate in {
        "Cargo.toml": cargo_version,
        "bindings/ns-py/pyproject.toml": ns_py_version,
        "bindings/ns-cli-py/pyproject.toml": ns_cli_py_version,
        "bindings/ns-py/pyproject.toml nextstat-cli pin": ns_cli_pin,
    }.items():
        _require(candidate == version, f"{label} version {candidate} does not match {version}")
    return {
        "cargo_toml": cargo_version,
        "ns_py_pyproject": ns_py_version,
        "ns_cli_py_pyproject": ns_cli_py_version,
        "nextstat_cli_pin": ns_cli_pin,
    }


def build_manifest(release_tag: str, mode: str) -> dict[str, Any]:
    version = validate_release_tag(release_tag)
    validate_mode(mode)
    surface_manifest = load_release_surface_matrix_manifest()
    version_alignment = aligned_versions(version)
    required_surfaces = [
        {
            "id": surface["id"],
            "name": surface["name"],
            "make_target": surface["make_target"],
            "workflow_job": surface["workflow_job"],
            "docs": surface["docs"],
        }
        for surface in surface_manifest["surfaces"]
        if surface["required_for_release"]
    ]
    optional_surfaces = [
        {
            "id": surface["id"],
            "name": surface["name"],
            "make_target": surface["make_target"],
            "workflow_job": surface["workflow_job"],
            "docs": surface["docs"],
        }
        for surface in surface_manifest["surfaces"]
        if not surface["required_for_release"]
    ]

    return {
        "schema_version": "nextstat.release_manifest.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "release_tag": release_tag,
        "version": version,
        "mode": mode,
        "git_sha": current_git_sha(),
        "version_alignment": version_alignment,
        "required_release_surfaces": required_surfaces,
        "optional_manual_surfaces": optional_surfaces,
        "candidate_artifacts": {
            "workflow_artifacts": [
                "simplified-likelihood-stable-surface-report",
                "simplified-likelihood-exporter-surface-report",
                "m15-reporting-stable-surface-report",
                "wheels-*",
                "sdist",
                "cli-wheels-*",
                "cli-sdist",
                "cli-*",
                "whitepaper",
                "validation-pack",
                "release-candidate-bundle",
            ],
            "github_release_asset_globs": [
                "dist/**/*.whl",
                "dist/**/*.tar.gz",
                "dist/cli-*/nextstat-*",
                "dist/**/release_candidate_bundle_manifest.json",
                "dist/**/release_candidate_bundle_manifest.md",
                "dist/**/release_manifest.json",
                "dist/**/release_manifest.md",
                "dist/**/release_manifest_v1.schema.json",
                "dist/**/nextstat-whitepaper-*.pdf",
                "dist/**/nextstat-whitepaper-*.pdf.sha256",
                "dist/**/apex2_master_report.json",
                "dist/**/validation_report.json",
                "dist/**/validation_report.pdf",
                "dist/**/validation_report_v1.schema.json",
                "dist/**/validation_pack_manifest.json",
                "dist/**/validation_pack_manifest.sha256",
                "dist/**/validation_pack_manifest.sha256.bin",
                "dist/**/snapshot_index.json",
                "dist/**/m15_config.json",
                "dist/**/m15_assessment_table.json",
                "dist/**/m15_map.json",
                "dist/**/m15_mar.json",
                "dist/**/m15_profile_diff_report.json",
                "dist/**/m15_profile_diff_report_v1.schema.json",
                "dist/**/m15_report.md",
                "dist/**/m15_report.pdf",
                "dist/**/m15_report.docx",
                "dist/**/m15_bundle_manifest.json",
                "dist/**/m15_bundle_manifest_v1.schema.json",
                "dist/**/m15_bundle_manifest.sha256",
                "dist/**/m15_bundle_manifest.sha256.bin",
                "dist/**/m15_snapshot_index.json",
                "dist/**/m15_reporting_benchmark.json",
                "dist/**/m15_reporting_benchmark.md",
                "dist/**/m15_reporting_compare.json",
                "dist/**/apex2_simplified_likelihood_report.json",
                "dist/**/export_benchmark_snapshot_report.json",
                "dist/**/export_public_validation_report.json",
                "dist/**/promotion_evidence.json",
                "dist/**/promotion_evidence_check.json",
                "dist/**/promotion_bundle_promotion_report.json",
                "dist/**/stable_review_assessment.json",
                "dist/**/stable_evidence_policy.json",
                "dist/**/stable_evidence_freshness_report.json",
                "dist/**/stable_source_semantics_boundary.json",
                "dist/**/stable_candidate_blocker_matrix.json",
                "dist/**/stable_candidate_review_packet.json",
                "dist/**/stable_promotion_decision.json",
            ],
        },
        "pharma_release_policy": {
            "prerelease_python_install_mode": "local_artifact_only",
            "canonical_release_evidence_platform": "linux",
            "cross_platform_saem_mode": "acceptance_envelope",
            "canonical_release_evidence_artifact": "pharma_validation.json",
        },
        "publish_targets": {
            "crates_io": [
                "ns-core",
                "ns-compute",
                "ns-ad",
                "ns-prob",
                "ns-translate",
                "ns-inference",
                "ns-viz",
                "ns-cli",
            ],
            "pypi": ["nextstat-cli", "nextstat"],
        },
    }


def render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Release Manifest",
        "",
        f"- Release tag: `{manifest['release_tag']}`",
        f"- Version: `{manifest['version']}`",
        f"- Mode: `{manifest['mode']}`",
        f"- Git SHA: `{manifest['git_sha']}`",
        f"- Generated: `{manifest['generated_at_utc']}`",
        "",
        "## Required release surfaces",
        "",
    ]
    for surface in manifest["required_release_surfaces"]:
        lines.append(
            f"- `{surface['id']}`: make `{surface['make_target']}` · workflow `{surface['workflow_job']}`"
        )
    lines.extend(
        [
            "",
            "## Candidate workflow artifacts",
            "",
        ]
    )
    for artifact in manifest["candidate_artifacts"]["workflow_artifacts"]:
        lines.append(f"- `{artifact}`")
    lines.extend(
        [
            "",
            "## Pharma release policy",
            "",
            f"- Prerelease Python install mode: `{manifest['pharma_release_policy']['prerelease_python_install_mode']}`",
            f"- Canonical release evidence platform: `{manifest['pharma_release_policy']['canonical_release_evidence_platform']}`",
            f"- Cross-platform SAEM mode: `{manifest['pharma_release_policy']['cross_platform_saem_mode']}`",
            f"- Canonical release evidence artifact: `{manifest['pharma_release_policy']['canonical_release_evidence_artifact']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the machine-readable release manifest.")
    parser.add_argument("--release-tag", required=True, help="Release tag in vX.Y.Z form.")
    parser.add_argument("--mode", required=True, choices=["prepare", "publish"])
    parser.add_argument("--out-json", help="Write the release manifest JSON to this path.")
    parser.add_argument("--out-md", help="Write the release manifest Markdown to this path.")
    parser.add_argument("--check", action="store_true", help="Validate inputs and exit.")
    args = parser.parse_args()

    manifest = build_manifest(args.release_tag, args.mode)
    markdown = render_markdown(manifest)

    if args.check and not args.out_json and not args.out_md:
        print("release_manifest_v1: ok")
        return 0

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(markdown, encoding="utf-8")
    if not args.out_json and not args.out_md:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
