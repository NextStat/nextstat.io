#!/usr/bin/env python3
"""Run a TREx analysis spec YAML (import/fit/scan/report) as a single reproducible workflow.

This script is intentionally "glue": it validates a YAML spec against the JSON Schema and
then executes `nextstat` CLI commands in a deterministic order.

Usage (dry run):
  ./.venv/bin/python scripts/trex/run_analysis_spec.py --spec docs/specs/trex/analysis_spec_v0.yaml --dry-run

Usage (execute):
  ./.venv/bin/python scripts/trex/run_analysis_spec.py --spec /abs/path/to/analysis.yaml
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Iterable

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _fmt_path(parts: Iterable[Any]) -> str:
    out = "$"
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            out += f".{p}"
    return out


def _validate(spec: Any, schema: Any) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Missing dependency jsonschema: {e}")

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(spec), key=lambda e: list(e.path))
    if not errors:
        return

    print(f"FAIL ({len(errors)} error(s))")
    for e in errors[:50]:
        print(f"- {_fmt_path(e.path)}: {e.message}")
    if len(errors) > 50:
        print(f"... ({len(errors) - 50} more)")
    raise SystemExit(2)


def _discover_combination_xml(export_dir: Path, *, dry_run: bool) -> Path:
    if not export_dir.exists():
        if dry_run:
            # For docs/CI dry-runs we still want to print the command sequence.
            return export_dir / "combination.xml"
        raise SystemExit(f"Missing HistFactory export_dir: {export_dir}")
    if not export_dir.is_dir():
        if dry_run:
            return export_dir / "combination.xml"
        raise SystemExit(f"HistFactory export_dir is not a directory: {export_dir}")

    hits = sorted(p for p in export_dir.rglob("combination.xml") if p.is_file())
    if not hits:
        if dry_run:
            return export_dir / "combination.xml"
        raise SystemExit(f"Could not auto-discover combination.xml under: {export_dir}")
    if len(hits) > 1:
        if dry_run:
            return hits[0]
        rendered = "\n".join(f"  - {p}" for p in hits[:10])
        extra = "" if len(hits) <= 10 else f"\n  ... ({len(hits) - 10} more)"
        raise SystemExit(
            "Multiple combination.xml files found; set inputs.histfactory.combination_xml explicitly.\n"
            f"{rendered}{extra}"
        )
    return hits[0]


def _ensure_parent(path: Path) -> None:
    parent = path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    _ensure_parent(path)
    path.write_text(text)


def _trex_yaml_to_txt(cfg: dict[str, Any]) -> str:
    # Matches `crates/ns-translate/src/trex/mod.rs` parsing expectations (subset).
    lines: list[str] = []

    def emit(k: str, v: str) -> None:
        lines.append(f"{k}: {v}")

    read_from = str(cfg["read_from"])
    emit("ReadFrom", read_from)
    emit("TreeName", str(cfg["tree_name"]))
    emit("Measurement", str(cfg["measurement"]))
    emit("POI", str(cfg["poi"]))
    lines.append("")

    for r in cfg["regions"]:
        emit("Region", str(r["name"]))
        emit("Variable", str(r["variable"]))
        edges = [float(x) for x in r["binning_edges"]]
        emit("Binning", ", ".join(f"{x:g}" for x in edges))
        sel = r.get("selection")
        if sel is not None:
            emit("Selection", str(sel))
        lines.append("")

    for s in cfg["samples"]:
        emit("Sample", str(s["name"]))
        kind = str(s["kind"])
        emit("Type", "data" if kind == "data" else "mc")
        emit("File", str(s["file"]))
        if s.get("tree_name") is not None:
            emit("TreeName", str(s["tree_name"]))
        if s.get("weight") is not None:
            emit("Weight", str(s["weight"]))
        regions = s.get("regions")
        if isinstance(regions, list) and regions:
            emit("Regions", ", ".join(map(str, regions)))
        nfs = s.get("norm_factors") or []
        if nfs:
            emit("NormFactor", ", ".join(map(str, nfs)))
        for ns in s.get("norm_sys") or []:
            emit("NormSys", f'{ns["name"]} {float(ns["lo"]):g} {float(ns["hi"]):g}')
        if bool(s.get("stat_error")):
            emit("StatError", "true")
        lines.append("")

    for sys in cfg.get("systematics") or []:
        emit("Systematic", str(sys["name"]))
        st = str(sys["type"])
        emit("Type", st)
        emit("Samples", ", ".join(map(str, sys["samples"])))
        regions = sys.get("regions")
        if isinstance(regions, list) and regions:
            emit("Regions", ", ".join(map(str, regions)))
        if st == "norm":
            emit("Lo", f'{float(sys["lo"]):g}')
            emit("Hi", f'{float(sys["hi"]):g}')
        elif st == "weight":
            emit("WeightUp", str(sys["weight_up"]))
            emit("WeightDown", str(sys["weight_down"]))
        elif st == "tree":
            emit("FileUp", str(sys["file_up"]))
            emit("FileDown", str(sys["file_down"]))
            if sys.get("tree_name") is not None:
                emit("TreeName", str(sys["tree_name"]))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _cmd_str(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print(_cmd_str(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True, help="Path to analysis spec YAML.")
    ap.add_argument(
        "--schema",
        type=Path,
        default=_repo_root() / "docs" / "schemas" / "trex" / "analysis_spec_v0.schema.json",
        help="Path to JSON Schema (default: repo schema).",
    )
    ap.add_argument(
        "--nextstat",
        type=str,
        default=None,
        help="Path to nextstat binary (default: target/release/nextstat if present, else nextstat).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = ap.parse_args()

    if not args.spec.exists():
        raise SystemExit(f"Missing spec: {args.spec}")
    if not args.schema.exists():
        raise SystemExit(f"Missing schema: {args.schema}")

    spec = _load_yaml(args.spec)
    schema = _load_json(args.schema)
    _validate(spec, schema)

    nextstat = args.nextstat
    if nextstat is None:
        local = _repo_root() / "target" / "release" / "nextstat"
        nextstat = str(local) if local.exists() else "nextstat"

    cmd = [nextstat, "run", "--config", str(args.spec)]
    _run(cmd, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
