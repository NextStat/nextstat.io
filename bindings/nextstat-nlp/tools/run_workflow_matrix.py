#!/usr/bin/env python3
"""Reproducible workflow runner: internet sources -> extraction -> nextstat fit.

This is the "Apex2 verification harness" for nextstat-nlp:
- Fetch internet snippets once (OpenFDA + ClinicalTrials.gov) into sources.json
- Re-run the pipelines 3 times (offline via --sources-json) to verify determinism
- Run a tiny nextstat fit (CoxPH + 1-subject NLME) from extracted structures

Outputs are designed to be copied as artifacts (not committed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    p = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _collect_hashes(out_dir: Path) -> Dict[str, str]:
    # Hash only the stable "product" files.
    keys = [
        "summary.json",
        "regimens.json",
        "survival.json",
        "priors.json",
        "nextstat_regimens.json",
        "nextstat_survival_design.json",
    ]
    out: Dict[str, str] = {}
    for k in keys:
        p = out_dir / k
        if p.exists():
            out[k] = _sha256_file(p)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run nextstat-nlp workflow matrix with reproducibility checks")
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/nlp_workflow_matrix"))
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--num-threads", type=int, default=4)
    ap.add_argument("--providers", nargs="*", default=None)
    ap.add_argument("--backends", nargs="*", default=["heuristic", "onnx"])
    args = ap.parse_args(list(argv) if argv is not None else None)

    root = Path(__file__).resolve().parents[1]
    tools_dir = root / "tools"
    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Fetch sources once (online run) with a single backend (heuristic).
    sources_run_dir = out_root / "sources_fetch"
    _run(
        [
            args.python,
            str(tools_dir / "run_internet_mocks.py"),
            "--backend",
            "heuristic",
            "--out-dir",
            str(sources_run_dir),
        ]
    )
    sources_json = sources_run_dir / "sources.json"
    if not sources_json.exists():
        raise SystemExit(f"missing sources.json at {sources_json}")

    matrix: Dict[str, Any] = {
        "sources_json": str(sources_json),
        "runs": [],
    }

    providers: List[str] = list(args.providers) if args.providers else []

    # 2) Repeat offline runs for each backend and verify bit-exact outputs.
    for backend in args.backends:
        backend_runs: List[Dict[str, Any]] = []
        prev_hashes: Optional[Dict[str, str]] = None
        all_match = True

        for i in range(args.n_repeats):
            run_dir = out_root / f"{backend}/repeat_{i}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Internet mocks (offline, reproducible)
            mocks_dir = run_dir / "internet_mocks"
            mocks_cmd = [
                args.python,
                str(tools_dir / "run_internet_mocks.py"),
                "--backend",
                backend,
                "--out-dir",
                str(mocks_dir),
                "--sources-json",
                str(sources_json),
                "--num-threads",
                str(args.num_threads),
            ]
            if providers:
                mocks_cmd += ["--providers", *providers]
            _run(mocks_cmd)

            # End-to-end demo to nextstat (also offline for regimen side)
            demo_dir = run_dir / "demo_nextstat"
            demo_cmd = [
                args.python,
                str(tools_dir / "demo_pharma_to_nextstat.py"),
                "--backend",
                backend,
                "--out-dir",
                str(demo_dir),
                "--num-threads",
                str(args.num_threads),
                "--seed",
                "42",
                "--sources-json",
                str(sources_json),
            ]
            if providers:
                demo_cmd += ["--providers", *providers]
            _run(demo_cmd)

            h = _collect_hashes(mocks_dir)
            if prev_hashes is not None and h != prev_hashes:
                all_match = False
            prev_hashes = h

            backend_runs.append(
                {
                    "repeat": i,
                    "mocks_dir": str(mocks_dir),
                    "demo_dir": str(demo_dir),
                    "mocks_hashes": h,
                    "mocks_summary": _load_json(mocks_dir / "summary.json"),
                }
            )

        matrix["runs"].append(
            {
                "backend": backend,
                "n_repeats": args.n_repeats,
                "bit_exact_mocks": bool(all_match),
                "runs": backend_runs,
            }
        )

    _write_json(out_root / "matrix_summary.json", matrix)
    print(f"Wrote workflow matrix artifacts to: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
