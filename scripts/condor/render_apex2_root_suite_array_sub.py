#!/usr/bin/env python3
"""Render a HTCondor job-array submit file with the correct `queue N`.

This reads an Apex2 ROOT cases JSON (see `tests/generate_apex2_root_cases.py`) and produces a
`.sub` file based on `scripts/condor/apex2_root_suite_array.sub` with:
- `initialdir = ...` set to the provided path (default: repo root)
- `queue N` set to the number of cases in the JSON

It is a convenience helper to avoid manual editing on lxplus/HTCondor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _count_cases(cases_json: Path) -> int:
    data = _read_json(cases_json)
    if not isinstance(data, dict):
        raise SystemExit("cases JSON must be an object")
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise SystemExit("cases JSON must contain a list under key 'cases'")
    n = len(cases)
    if n <= 0:
        raise SystemExit("cases JSON contains zero cases")
    return int(n)


def _render_lines(lines: List[str], *, initialdir: str, n_cases: int) -> List[str]:
    out: List[str] = []
    queue_replaced = False
    for line in lines:
        if line.startswith("initialdir ="):
            out.append(f"initialdir = {initialdir}\n")
            continue
        if line.strip().startswith("queue "):
            out.append(f"queue {n_cases}\n")
            queue_replaced = True
            continue
        out.append(line)
    if not queue_replaced:
        out.append(f"\nqueue {n_cases}\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True, help="Path to Apex2 ROOT cases JSON.")
    ap.add_argument(
        "--template",
        type=Path,
        default=_repo_root() / "scripts" / "condor" / "apex2_root_suite_array.sub",
        help="Path to the template .sub file.",
    )
    ap.add_argument(
        "--initialdir",
        type=Path,
        default=_repo_root(),
        help="Value for HTCondor initialdir (default: repo root).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("apex2_root_suite_array_rendered.sub"),
        help="Output .sub path (default: apex2_root_suite_array_rendered.sub).",
    )
    args = ap.parse_args()

    if not args.cases.exists():
        raise SystemExit(f"missing cases file: {args.cases}")
    if not args.template.exists():
        raise SystemExit(f"missing template: {args.template}")

    n_cases = _count_cases(args.cases)
    lines = args.template.read_text().splitlines(keepends=True)
    rendered = _render_lines(lines, initialdir=str(args.initialdir), n_cases=n_cases)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(rendered))
    print(f"Wrote: {args.out}")
    print(f"n_cases: {n_cases}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
