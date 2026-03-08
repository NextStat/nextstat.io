from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_bundle_manifest(
    release_tag: str,
    mode: str,
    required_inputs: dict[str, Path],
    optional_inputs: dict[str, Path],
) -> dict[str, Any]:
    required_entries = []
    for name, path in required_inputs.items():
        _require(path.exists(), f"missing required bundle input: {path}")
        required_entries.append(
            {
                "name": name,
                "required": True,
                "source_path": str(path),
                "bundle_path": path.name,
            }
        )

    optional_entries = []
    for name, path in optional_inputs.items():
        optional_entries.append(
            {
                "name": name,
                "required": False,
                "source_path": str(path),
                "bundle_path": path.name,
                "present": path.exists(),
            }
        )

    return {
        "schema_version": "nextstat.release_candidate_bundle.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "release_tag": release_tag,
        "mode": mode,
        "required_entries": required_entries,
        "optional_entries": optional_entries,
    }


def render_markdown(bundle_manifest: dict[str, Any]) -> str:
    lines = [
        "# Release Candidate Bundle",
        "",
        f"- Release tag: `{bundle_manifest['release_tag']}`",
        f"- Mode: `{bundle_manifest['mode']}`",
        f"- Generated: `{bundle_manifest['generated_at_utc']}`",
        "",
        "## Required entries",
        "",
    ]
    for entry in bundle_manifest["required_entries"]:
        lines.append(f"- `{entry['name']}` → `{entry['bundle_path']}`")

    lines.extend(["", "## Optional entries", ""])
    for entry in bundle_manifest["optional_entries"]:
        state = "present" if entry["present"] else "missing"
        lines.append(f"- `{entry['name']}` → `{entry['bundle_path']}` ({state})")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the release candidate bundle.")
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--mode", required=True, choices=["prepare", "publish"])
    parser.add_argument("--surface-report-json", required=True)
    parser.add_argument("--surface-report-md", required=True)
    parser.add_argument("--release-manifest-json", required=True)
    parser.add_argument("--release-manifest-md", required=True)
    parser.add_argument("--baseline-report-json")
    parser.add_argument("--trex-report-json")
    parser.add_argument("--root-report-json")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required_inputs = {
        "release_surface_matrix_report_json": Path(args.surface_report_json),
        "release_surface_matrix_report_md": Path(args.surface_report_md),
        "release_manifest_json": Path(args.release_manifest_json),
        "release_manifest_md": Path(args.release_manifest_md),
    }
    optional_inputs: dict[str, Path] = {}
    if args.baseline_report_json:
        optional_inputs["baseline_compare_report_json"] = Path(args.baseline_report_json)
    if args.trex_report_json:
        optional_inputs["trex_analysis_spec_compare_report_json"] = Path(args.trex_report_json)
    if args.root_report_json:
        optional_inputs["root_suite_compare_report_json"] = Path(args.root_report_json)

    bundle_manifest = build_bundle_manifest(args.release_tag, args.mode, required_inputs, optional_inputs)

    for path in required_inputs.values():
        shutil.copy2(path, out_dir / path.name)
    for path in optional_inputs.values():
        if path.exists():
            shutil.copy2(path, out_dir / path.name)

    manifest_path = out_dir / "release_candidate_bundle_manifest.json"
    markdown_path = out_dir / "release_candidate_bundle_manifest.md"
    manifest_path.write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(bundle_manifest), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
