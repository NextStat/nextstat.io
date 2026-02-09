from __future__ import annotations

import argparse
import json
from pathlib import Path

from .importer import TrexConfigImportError, dump_yaml, trex_config_file_to_analysis_spec_v0


def _ensure_parent(path: Path) -> None:
    parent = path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"refusing to overwrite existing file (pass --overwrite): {path}")
    _ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: object, *, overwrite: bool) -> None:
    _write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n", overwrite=overwrite)


def _cmd_import_config(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    report_path = Path(args.report) if args.report is not None else out_path.with_suffix(".mapping.json")

    try:
        spec, report = trex_config_file_to_analysis_spec_v0(
            args.config,
            out_path=out_path,
            threads=int(args.threads),
            workspace_out=str(args.workspace_out),
        )
    except TrexConfigImportError as e:
        raise SystemExit(str(e)) from e

    _write_text(out_path, dump_yaml(spec), overwrite=bool(args.overwrite))
    _write_json(report_path, report, overwrite=bool(args.overwrite))

    print(json.dumps({"analysis_spec": str(out_path), "mapping_report": str(report_path)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m nextstat.trex_config.cli")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("import-config", help="Convert TRExFitter .config -> analysis spec v0 (best-effort).")
    p.add_argument("--config", type=Path, required=True, help="Path to TRExFitter .config file.")
    p.add_argument("--out", type=Path, required=True, help="Output analysis spec path (YAML).")
    p.add_argument("--report", type=Path, default=None, help="Output mapping report path (JSON).")
    p.add_argument("--threads", type=int, default=1, help="Determinism threads (>=1).")
    p.add_argument(
        "--workspace-out",
        type=str,
        default="tmp/trex_workspace.json",
        help="execution.import.output_json path to write in the generated analysis spec.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")

    args = ap.parse_args(argv)
    if args.command == "import-config":
        return _cmd_import_config(args)
    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

