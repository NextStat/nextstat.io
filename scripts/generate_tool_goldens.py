"""Generate deterministic tool golden outputs.

This script runs against the in-repo Python tool surface layered onto an
installed NextStat core package. That keeps tool goldens hermetic to the
workspace while still reusing the compiled extension module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _activate_repo_python_package() -> None:
    import nextstat

    repo_pkg = _repo_root() / "bindings" / "ns-py" / "python" / "nextstat"
    pkg_path = str(repo_pkg)
    package_paths = getattr(nextstat, "__path__", None)
    if package_paths is None:
        raise RuntimeError("nextstat package has no __path__; cannot overlay repo Python modules")
    if pkg_path in package_paths:
        package_paths.remove(pkg_path)
    package_paths.insert(0, pkg_path)
    sys.modules.pop("nextstat.tools", None)
    sys.modules.pop("nextstat._tool_manifest", None)


def _collect_fixture_text_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        if set(value.keys()) == {"$fixture_text"} and isinstance(value.get("$fixture_text"), str):
            refs.add(value["$fixture_text"])
            return refs
        for nested in value.values():
            refs.update(_collect_fixture_text_refs(nested))
        return refs
    if isinstance(value, list):
        for nested in value:
            refs.update(_collect_fixture_text_refs(nested))
    return refs


def _golden_case_names(tool_records: list[dict[str, Any]]) -> list[str]:
    case_names: set[str] = set()
    for record in tool_records:
        cases = record.get("golden_cases")
        if isinstance(cases, dict):
            case_names.update(case_name for case_name in cases if isinstance(case_name, str))
    return sorted(case_names)


def _infer_case_fixture(tool_records: list[dict[str, Any]], case_name: str) -> str | None:
    refs: set[str] = set()
    for record in tool_records:
        cases = record.get("golden_cases")
        if not isinstance(cases, dict) or case_name not in cases:
            continue
        case = cases[case_name]
        if not isinstance(case, dict):
            continue
        refs.update(_collect_fixture_text_refs(case.get("arguments")))
    if len(refs) > 1:
        raise RuntimeError(f"golden case {case_name!r} resolves multiple fixture refs: {sorted(refs)}")
    return next(iter(refs), None)


def _canonicalize_tool_envelope(value: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(value))
    meta = out.get("meta")
    if isinstance(meta, dict):
        meta.pop("nextstat_version", None)
        meta.pop("threads_applied", None)
        meta.pop("device", None)
        meta.pop("warnings", None)

    def _drop_unstable_fields(v: Any) -> Any:
        if isinstance(v, dict):
            v = dict(v)
            v.pop("n_iter", None)
            v.pop("wall_time_s", None)
            v.pop("elapsed_s", None)
            v.pop("scenarios_per_sec", None)
            v.pop("mu_values", None)
            return {k: _drop_unstable_fields(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [_drop_unstable_fields(vv) for vv in v]
        return v

    return _drop_unstable_fields(out)


def generate_tool_goldens() -> dict[str, dict[str, Any]]:
    import nextstat  # noqa: F401

    _activate_repo_python_package()

    from nextstat.tools import execute_tool
    from nextstat._tool_manifest import get_tool_records, resolve_golden_case_arguments

    tool_records = get_tool_records()
    generated: dict[str, dict[str, Any]] = {}

    for case_name in _golden_case_names(tool_records):
        out: dict[str, Any] = {
            "schema_version": "nextstat.tool_goldens.v1",
            "case_name": case_name,
            "tools": {},
        }
        fixture = _infer_case_fixture(tool_records, case_name)
        if fixture is not None:
            out["fixture"] = fixture

        for record in tool_records:
            name = record.get("name")
            if not isinstance(name, str):
                continue
            cases = record.get("golden_cases")
            if not isinstance(cases, dict) or case_name not in cases:
                continue
            args = resolve_golden_case_arguments(name, case_name, _repo_root())
            out["tools"][name] = _canonicalize_tool_envelope(execute_tool(name, args))

        generated[case_name] = out

    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated output differs")
    args = parser.parse_args(argv)

    out_dir = _repo_root() / "tests" / "fixtures" / "tool_goldens"
    generated_by_case = generate_tool_goldens()
    expected_paths = {f"{case_name}.v1.json" for case_name in generated_by_case}
    stale_paths = sorted(
        path.name for path in out_dir.glob("*.v1.json") if path.name not in expected_paths
    )

    out_of_date: list[Path] = []
    wrote_any = False
    for case_name, payload in generated_by_case.items():
        out_path = out_dir / f"{case_name}.v1.json"
        generated = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        current = out_path.read_text(encoding="utf-8") if out_path.exists() else None
        if current != generated:
            if args.check:
                out_of_date.append(out_path)
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path.write_text(generated, encoding="utf-8")
                print(f"Wrote {out_path}")
                wrote_any = True

    if stale_paths:
        if args.check:
            for stale in stale_paths:
                print(f"stale golden file: {out_dir / stale}", file=sys.stderr)
            return 1
        for stale in stale_paths:
            stale_path = out_dir / stale
            stale_path.unlink()
            print(f"Removed stale {stale_path}")
            wrote_any = True

    if args.check:
        if out_of_date:
            for path in out_of_date:
                print(f"out of date: {path}", file=sys.stderr)
            return 1
        print(f"Up to date: {len(expected_paths)} golden files in {out_dir}")
    elif not wrote_any:
        print(f"No changes: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
