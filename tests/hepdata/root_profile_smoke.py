#!/usr/bin/env python3
"""HEPData -> ROOT/HistFactory -> NextStat profile-scan smoke runner.

This script is intentionally opt-in and writes only under `tmp/` by default.

Pipeline per workspace JSON:
1) pyhf JSON -> HistFactory export (pyhf.writexml, requires uproot)
2) ROOT reference (hist2workspace + RooFit profiling)
3) NextStat profile_scan (parity mode, threads=1)
4) Record a small summary and (optionally) enforce tolerances.

Typical usage:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/hepdata/root_profile_smoke.py --fetch
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "tests" / "hepdata" / "manifest.json"
DEFAULT_WORKSPACES_OUT = REPO / "tmp" / "external_hepdata" / "workspaces"
DEFAULT_CACHE = REPO / "tmp" / "external_hepdata" / "_cache"
DEFAULT_LOCK = REPO / "tmp" / "external_hepdata" / "workspaces.lock.json"
DEFAULT_WORKDIR = REPO / "tmp" / "hepdata_root_parity"


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _iter_workspace_jsons(root: Path) -> Iterable[Path]:
    return sorted([p for p in root.rglob("*.json") if p.is_file()])


def _measurement_name(workspace_json: Path) -> str:
    ws = _load_json(workspace_json)
    meas = ws.get("measurements") or []
    if not isinstance(meas, list) or not meas:
        raise RuntimeError(f"workspace has no measurements: {workspace_json}")
    name = meas[0].get("name")
    if not isinstance(name, str) or not name:
        raise RuntimeError(f"workspace measurement[0].name missing: {workspace_json}")
    return name


def _run_validate(
    *,
    workspace_json: Path,
    measurement: str,
    start: float,
    stop: float,
    points: int,
    workdir: Path,
    keep: bool,
) -> Dict[str, Any]:
    validate = REPO / "tests" / "validate_root_profile_scan.py"
    cmd = [
        sys.executable,
        str(validate),
        "--pyhf-json",
        str(workspace_json),
        "--measurement",
        str(measurement),
        "--start",
        str(start),
        "--stop",
        str(stop),
        "--points",
        str(points),
        "--workdir",
        str(workdir),
    ]
    if keep:
        cmd.append("--keep")
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "bindings/ns-py/python")
    p = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"validate_root_profile_scan failed ({p.returncode}):\n{p.stdout}")
    # validate prints a JSON summary to stdout.
    try:
        return json.loads(p.stdout)
    except Exception as e:
        raise RuntimeError(f"Failed to parse validate_root_profile_scan JSON output: {e}\n{p.stdout}") from e


def _fetch_hepdata(*, manifest: Path, out: Path, cache: Path, lock: Path, datasets: List[str]) -> None:
    fetch = REPO / "tests" / "hepdata" / "fetch_workspaces.py"
    cmd = [
        sys.executable,
        str(fetch),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--cache",
        str(cache),
        "--lock",
        str(lock),
    ]
    for ds in datasets:
        cmd += ["--dataset", ds]
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", "bindings/ns-py/python")
    p = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"fetch_workspaces failed ({p.returncode}):\n{p.stdout}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--workspaces-out", type=Path, default=DEFAULT_WORKSPACES_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--dataset", action="append", default=[], help="Dataset id to fetch/run (repeatable).")
    ap.add_argument("--fetch", action="store_true", help="Download+materialize workspaces before running.")
    ap.add_argument(
        "--include-bkgonly",
        action="store_true",
        help="Also run BkgOnly.json workspaces (often lack a usable POI in ROOT/HistFactory).",
    )
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--stop", type=float, default=1.0)
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    ap.add_argument("--keep", action="store_true", help="Keep per-run staging artifacts (combination.xml, etc).")
    ap.add_argument("--dq-atol", type=float, default=None)
    ap.add_argument("--mu-hat-atol", type=float, default=None)
    args = ap.parse_args(argv)

    if not _which("root") or not _which("hist2workspace"):
        print("SKIP: missing ROOT prereqs (root/hist2workspace).", file=sys.stderr)
        return 0

    if args.fetch:
        _fetch_hepdata(
            manifest=args.manifest,
            out=args.workspaces_out,
            cache=args.cache,
            lock=args.lock,
            datasets=list(args.dataset or []),
        )

    roots = list(_iter_workspace_jsons(args.workspaces_out))
    if not roots:
        print(f"ERROR: no workspaces under: {args.workspaces_out}", file=sys.stderr)
        return 2

    if not args.include_bkgonly:
        roots = [p for p in roots if p.name != "BkgOnly.json"]

    # Filter if requested.
    if args.dataset:
        allowed = set(str(d).replace("/", "_") for d in args.dataset)
        roots = [p for p in roots if any(a in str(p) for a in allowed)]

    args.workdir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    ok_all = True
    for ws_path in roots:
        try:
            meas = _measurement_name(ws_path)
            rep = _run_validate(
                workspace_json=ws_path,
                measurement=meas,
                start=float(args.start),
                stop=float(args.stop),
                points=int(args.points),
                workdir=args.workdir,
                keep=bool(args.keep),
            )
            diff = rep.get("diff") or {}
            dq = float(diff.get("max_abs_dq_mu", 0.0))
            dmu = float(diff.get("mu_hat", 0.0))
            ok = True
            if args.dq_atol is not None and abs(dq) > float(args.dq_atol):
                ok = False
            if args.mu_hat_atol is not None and abs(dmu) > float(args.mu_hat_atol):
                ok = False
            ok_all = ok_all and ok
            print(f"[hepdata-root] ok={ok} dq={dq:.3e} d_mu_hat={dmu:.3e} meas={meas} ws={ws_path}")
            results.append({"workspace": str(ws_path), "measurement": meas, "ok": ok, "report": rep})
        except Exception as e:
            ok_all = False
            print(f"[hepdata-root] ERROR ws={ws_path}: {e}", file=sys.stderr)
            results.append({"workspace": str(ws_path), "ok": False, "error": str(e)})

    out = args.workdir / "hepdata_root_profile_smoke_aggregate.json"
    _write_json(out, {"inputs": {"start": args.start, "stop": args.stop, "points": args.points}, "results": results})
    print(f"[hepdata-root] wrote: {out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
