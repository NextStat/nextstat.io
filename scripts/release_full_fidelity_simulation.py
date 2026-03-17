from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.release_manifest import build_manifest, validate_release_tag
from scripts.release_stage_assets import stage_release_assets


@dataclass(frozen=True)
class UploadArtifactStep:
    name: str
    paths: tuple[str, ...]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workflow_path() -> Path:
    return repo_root() / ".github" / "workflows" / "release-candidate.yml"


def _release_manifest_path() -> Path:
    return repo_root() / "tmp" / "release_manifest.json"


def _release_bundle_path() -> Path:
    return repo_root() / "tmp" / "release_candidate_bundle"


def _count_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_upload_artifact_steps(workflow_text: str) -> list[UploadArtifactStep]:
    lines = workflow_text.splitlines()
    steps: list[UploadArtifactStep] = []
    i = 0
    while i < len(lines):
        step_match = re.match(r"^(\s*)- name:\s+(.+)$", lines[i])
        if not step_match:
            i += 1
            continue
        step_indent = len(step_match.group(1))
        step_block = [lines[i]]
        i += 1
        while i < len(lines):
            next_step = re.match(r"^(\s*)- name:\s+(.+)$", lines[i])
            if next_step and len(next_step.group(1)) == step_indent:
                break
            step_block.append(lines[i])
            i += 1
        step_text = "\n".join(step_block)
        if "uses: actions/upload-artifact@v4" not in step_text:
            continue

        artifact_name: str | None = None
        artifact_paths: list[str] = []
        in_with = False
        with_indent = -1
        in_path_block = False
        path_indent = -1
        for line in step_block[1:]:
            stripped = line.strip()
            indent = _count_indent(line)
            if stripped == "with:":
                in_with = True
                with_indent = indent
                in_path_block = False
                continue
            if in_with and stripped and indent <= with_indent:
                in_with = False
                in_path_block = False
            if not in_with:
                continue
            if stripped.startswith("name:"):
                artifact_name = stripped.split("name:", 1)[1].strip()
                continue
            if stripped == "path: |":
                in_path_block = True
                path_indent = indent
                continue
            if stripped.startswith("path: "):
                artifact_paths.append(stripped.split("path:", 1)[1].strip())
                in_path_block = False
                continue
            if in_path_block:
                if stripped and indent > path_indent:
                    artifact_paths.append(stripped)
                elif stripped:
                    in_path_block = False

        if artifact_name and artifact_paths:
            steps.append(UploadArtifactStep(name=artifact_name, paths=tuple(artifact_paths)))
    return steps


def _replace_matrix_placeholder(value: str) -> str:
    return value.replace("${{ matrix.target }}", "sample-target")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _release_bundle_seed_paths() -> list[str]:
    return [
        "tmp/release_candidate_bundle/release_candidate_bundle_manifest.json",
        "tmp/release_candidate_bundle/release_candidate_bundle_manifest.md",
        "tmp/release_candidate_bundle/release_manifest.json",
        "tmp/release_candidate_bundle/release_manifest.md",
        "docs/schemas/releases/release_candidate_bundle_v1.schema.json",
        "docs/schemas/releases/release_manifest_v1.schema.json",
    ]


def _validation_pack_seed_paths() -> list[str]:
    return [
        "artifacts/apex2_master_report.json",
        "artifacts/validation_report.json",
        "artifacts/validation_report.pdf",
        "artifacts/validation_report_v1.schema.json",
        "artifacts/validation_pack_manifest.json",
        "artifacts/validation_pack_manifest.sha256",
        "artifacts/validation_pack_manifest.sha256.bin",
        "artifacts/snapshot_index.json",
    ]


def _validation_pack_m15_seed_paths() -> list[str]:
    return [
        "artifacts_m15/validation_report.json",
        "artifacts_m15/validation_report.pdf",
        "artifacts_m15/m15_config.json",
        "artifacts_m15/m15_assessment_table.json",
        "artifacts_m15/m15_map.json",
        "artifacts_m15/m15_mar.json",
        "artifacts_m15/m15_profile_diff_report.json",
        "artifacts_m15/m15_profile_diff_report_v1.schema.json",
        "artifacts_m15/m15_report.md",
        "artifacts_m15/m15_report.pdf",
        "artifacts_m15/m15_report.docx",
        "artifacts_m15/m15_bundle_manifest.json",
        "artifacts_m15/m15_bundle_manifest_v1.schema.json",
        "artifacts_m15/m15_bundle_manifest.sha256",
        "artifacts_m15/m15_bundle_manifest.sha256.bin",
        "artifacts_m15/m15_snapshot_index.json",
    ]


def seed_paths_for_pattern(pattern: str, version: str) -> list[str]:
    if "*" not in pattern:
        return [_replace_matrix_placeholder(pattern)]
    normalized = _replace_matrix_placeholder(pattern)
    if normalized == "bindings/ns-py/dist/*.whl":
        return [f"bindings/ns-py/dist/nextstat-{version}-cp313-cp313-sample.whl"]
    if normalized == "bindings/ns-py/dist/*.tar.gz":
        return [f"bindings/ns-py/dist/nextstat-{version}.tar.gz"]
    if normalized == "bindings/ns-cli-py/dist/*.whl":
        return [f"bindings/ns-cli-py/dist/nextstat_cli-{version}-py3-none-sample.whl"]
    if normalized == "bindings/ns-cli-py/dist/*.tar.gz":
        return [f"bindings/ns-cli-py/dist/nextstat_cli-{version}.tar.gz"]
    if normalized == "dist/nextstat-*":
        return ["dist/nextstat-sample-target"]
    if normalized == "dist/whitepaper/*":
        return [
            f"dist/whitepaper/nextstat-whitepaper-v{version}.pdf",
            f"dist/whitepaper/nextstat-whitepaper-v{version}.pdf.sha256",
        ]
    if normalized == "artifacts/*":
        return _validation_pack_seed_paths()
    if normalized == "artifacts_m15/*":
        return _validation_pack_m15_seed_paths()
    if normalized == "tmp/release_candidate_bundle/*":
        return _release_bundle_seed_paths()[:4]
    return [normalized.replace("*", "sample")]


def seed_placeholder_sources(source_root: Path, upload_steps: list[UploadArtifactStep], version: str) -> list[Path]:
    created: list[Path] = []
    for step in upload_steps:
        for pattern in step.paths:
            for rel_path in seed_paths_for_pattern(pattern, version):
                path = source_root / rel_path
                if path.exists():
                    continue
                suffix = path.suffix.lower()
                content = "placeholder"
                if suffix == ".json":
                    content = json.dumps({"placeholder": rel_path}, indent=2) + "\n"
                elif suffix in {".md", ".txt", ".sha256", ".bin"} or path.name.endswith(".pdf.sha256"):
                    content = f"{rel_path}\n"
                _write(path, content)
                created.append(path)
    return created


def _download_relative_path(pattern: str, relative_match: Path) -> Path:
    normalized = _replace_matrix_placeholder(pattern)
    if "*" not in normalized:
        return relative_match
    if normalized in {"artifacts/*", "artifacts_m15/*", "tmp/release_candidate_bundle/*"}:
        return relative_match
    if normalized == "dist/whitepaper/*":
        return Path(relative_match.name)
    if normalized in {
        "bindings/ns-py/dist/*.whl",
        "bindings/ns-py/dist/*.tar.gz",
        "bindings/ns-cli-py/dist/*.whl",
        "bindings/ns-cli-py/dist/*.tar.gz",
        "dist/nextstat-*",
    }:
        return Path(relative_match.name)
    return relative_match


def simulate_downloaded_artifacts(
    source_root: Path,
    dist_root: Path,
    upload_steps: list[UploadArtifactStep],
) -> dict[str, list[str]]:
    dist_root.mkdir(parents=True, exist_ok=True)
    inventory: dict[str, list[str]] = {}
    for step in upload_steps:
        copied: list[str] = []
        seen_targets: set[Path] = set()
        for pattern in step.paths:
            matches = sorted(source_root.glob(_replace_matrix_placeholder(pattern)))
            artifact_dir = dist_root / _replace_matrix_placeholder(step.name)
            for src in matches:
                if not src.is_file():
                    continue
                rel = src.relative_to(source_root)
                target = artifact_dir / _download_relative_path(pattern, rel)
                if target in seen_targets:
                    continue
                seen_targets.add(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                copied.append(str(target.relative_to(dist_root)))
        if copied:
            inventory[_replace_matrix_placeholder(step.name)] = copied
    return inventory


def build_simulation_report(
    release_tag: str,
    mode: str,
    artifact_inventory: dict[str, list[str]],
    staged_assets: list[Path],
) -> dict[str, Any]:
    manifest = build_manifest(release_tag, mode)
    return {
        "schema_version": "nextstat.release_full_fidelity_simulation.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "release_tag": release_tag,
        "mode": mode,
        "workflow_artifacts": artifact_inventory,
        "staged_assets": sorted(path.name for path in staged_assets),
        "github_release_asset_globs": manifest["candidate_artifacts"]["github_release_asset_globs"],
    }


def render_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Release Full-Fidelity Simulation",
        "",
        f"- Release tag: `{report['release_tag']}`",
        f"- Mode: `{report['mode']}`",
        f"- Generated: `{report['generated_at_utc']}`",
        "",
        "## Simulated workflow artifacts",
        "",
    ]
    for name, files in sorted(report["workflow_artifacts"].items()):
        lines.append(f"- `{name}`")
        for file_name in files:
            lines.append(f"  - `{file_name}`")
    lines.extend(["", "## Staged release assets", ""])
    for asset in report["staged_assets"]:
        lines.append(f"- `{asset}`")
    return "\n".join(lines) + "\n"


def run_simulation(release_tag: str, mode: str, out_dir: Path) -> tuple[dict[str, Any], Path]:
    version = validate_release_tag(release_tag)
    workflow_text = _workflow_path().read_text(encoding="utf-8")
    upload_steps = parse_upload_artifact_steps(workflow_text)
    source_root = out_dir / "source"
    dist_root = out_dir / "dist"
    staged_root = out_dir / "release-assets"
    seed_placeholder_sources(source_root, upload_steps, version)
    artifact_inventory = simulate_downloaded_artifacts(source_root, dist_root, upload_steps)
    staged_assets = stage_release_assets(dist_root, staged_root)
    report = build_simulation_report(release_tag, mode, artifact_inventory, staged_assets)
    return report, staged_root


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a local full-fidelity simulation of release-candidate artifact download and GitHub Release asset staging."
    )
    parser.add_argument("--release-tag", required=True, help="Release tag in vX.Y.Z form.")
    parser.add_argument("--mode", required=True, choices=["prepare", "publish"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report, _ = run_simulation(args.release_tag, args.mode, out_dir)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render_report_markdown(report), encoding="utf-8")

    if args.check and not args.out_json and not args.out_md:
        print("release_full_fidelity_simulation_v1: ok")
        return 0

    if not args.out_json and not args.out_md:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
