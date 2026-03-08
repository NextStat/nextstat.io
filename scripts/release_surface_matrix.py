from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def manifest_path() -> Path:
    return Path(__file__).with_name("release_surface_matrix_v1.json")


def workflow_path() -> Path:
    return repo_root() / ".github" / "workflows" / "release-candidate.yml"


def makefile_path() -> Path:
    return repo_root() / "Makefile"


def load_manifest() -> dict[str, Any]:
    return json.loads(manifest_path().read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(manifest: dict[str, Any]) -> None:
    _require(
        manifest.get("schema_version") == "nextstat.release_surface_matrix.v1",
        "schema_version must be 'nextstat.release_surface_matrix.v1'",
    )
    surfaces = manifest.get("surfaces")
    _require(isinstance(surfaces, list) and surfaces, "surfaces must be a non-empty list")

    seen_ids: set[str] = set()
    workflow = workflow_path().read_text(encoding="utf-8")
    makefile = makefile_path().read_text(encoding="utf-8")

    for surface in surfaces:
        _require(isinstance(surface, dict), "each surface must be an object")
        surface_id = surface.get("id")
        _require(isinstance(surface_id, str) and surface_id, "surface.id must be a non-empty string")
        _require(surface_id not in seen_ids, f"duplicate surface id: {surface_id}")
        seen_ids.add(surface_id)

        _require(
            isinstance(surface.get("name"), str) and surface["name"],
            f"{surface_id}: name must be a non-empty string",
        )
        _require(
            isinstance(surface.get("required_for_release"), bool),
            f"{surface_id}: required_for_release must be boolean",
        )
        _require(
            isinstance(surface.get("make_target"), str),
            f"{surface_id}: make_target must be a string",
        )
        _require(
            isinstance(surface.get("workflow_job"), str),
            f"{surface_id}: workflow_job must be a string",
        )
        docs = surface.get("docs")
        _require(isinstance(docs, list) and docs, f"{surface_id}: docs must be a non-empty list")
        for doc in docs:
            _require(isinstance(doc, str) and doc, f"{surface_id}: doc paths must be strings")
            _require((repo_root() / doc).exists(), f"{surface_id}: missing doc {doc}")

        path_globs = surface.get("path_globs")
        _require(
            isinstance(path_globs, list) and path_globs,
            f"{surface_id}: path_globs must be a non-empty list",
        )
        for pattern in path_globs:
            _require(isinstance(pattern, str) and pattern, f"{surface_id}: invalid path_glob entry")

        make_target = surface["make_target"]
        if make_target:
            _require(f"\n{make_target}:" in f"\n{makefile}", f"{surface_id}: make target {make_target} missing from Makefile")

        workflow_job = surface["workflow_job"]
        if workflow_job:
            _require(f"\n  {workflow_job}:" in f"\n{workflow}", f"{surface_id}: workflow job {workflow_job} missing from release.yml")


def latest_release_tag() -> str | None:
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", "--match", "v*"],
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    tag = result.stdout.strip()
    return tag or None


def changed_paths_since(base_ref: str) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def resolve_changed_paths(explicit_paths: list[str], base_ref: str | None) -> tuple[str | None, list[str]]:
    if explicit_paths:
        return None, explicit_paths
    resolved_base = base_ref or latest_release_tag()
    if not resolved_base:
        return None, []
    return resolved_base, changed_paths_since(resolved_base)


def surface_matches_paths(surface: dict[str, Any], changed_paths: list[str]) -> bool:
    patterns = surface["path_globs"]
    return any(
        fnmatch.fnmatch(path, pattern)
        for path in changed_paths
        for pattern in patterns
    )


def build_report(manifest: dict[str, Any], changed_paths: list[str], base_ref: str | None) -> dict[str, Any]:
    required = []
    optional = []
    touched = []
    for surface in manifest["surfaces"]:
        entry = {
            "id": surface["id"],
            "name": surface["name"],
            "make_target": surface["make_target"],
            "workflow_job": surface["workflow_job"],
            "docs": surface["docs"],
        }
        if surface["required_for_release"]:
            required.append(entry)
        else:
            optional.append(entry)
        if changed_paths and surface_matches_paths(surface, changed_paths):
            touched.append(entry)

    return {
        "schema_version": "nextstat.release_surface_matrix_report.v1",
        "manifest_schema_version": manifest["schema_version"],
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "base_ref": base_ref,
        "changed_paths_count": len(changed_paths),
        "changed_paths": changed_paths,
        "required_release_surfaces": required,
        "advisory_touched_surfaces": touched,
        "optional_manual_surfaces": optional,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Release Surface Matrix Report",
        "",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Base ref: `{report['base_ref'] or 'none'}`",
        f"- Changed paths: `{report['changed_paths_count']}`",
        "",
        "## Required release surfaces",
        "",
    ]
    for surface in report["required_release_surfaces"]:
        lines.append(
            f"- `{surface['id']}`: make `{surface['make_target']}` · workflow `{surface['workflow_job']}`"
        )
    lines.append("")
    lines.append("## Advisory touched surfaces")
    lines.append("")
    if report["advisory_touched_surfaces"]:
        for surface in report["advisory_touched_surfaces"]:
            lines.append(
                f"- `{surface['id']}`: make `{surface['make_target']}` · workflow `{surface['workflow_job']}`"
            )
    else:
        lines.append("- none inferred from changed paths")
    lines.append("")
    lines.append("## Optional manual surfaces")
    lines.append("")
    for surface in report["optional_manual_surfaces"]:
        gate = surface["make_target"] or "manual"
        workflow_job = surface["workflow_job"] or "manual"
        lines.append(f"- `{surface['id']}`: make `{gate}` · workflow `{workflow_job}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and render the release surface matrix.")
    parser.add_argument("--check", action="store_true", help="Validate the manifest and exit.")
    parser.add_argument("--base-ref", help="Git base ref/tag for changed-path advisory diff.")
    parser.add_argument(
        "--changed-path",
        action="append",
        dest="changed_paths",
        default=[],
        help="Explicit changed path (repeatable).",
    )
    parser.add_argument("--out-json", help="Write the release surface report JSON to this path.")
    parser.add_argument("--out-md", help="Write the release surface report Markdown to this path.")
    args = parser.parse_args()

    manifest = load_manifest()
    validate_manifest(manifest)
    if args.check and not args.out_json and not args.out_md:
        print("release_surface_matrix_v1: ok")
        return 0

    resolved_base, changed_paths = resolve_changed_paths(args.changed_paths, args.base_ref)
    report = build_report(manifest, changed_paths, resolved_base)
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
