#!/usr/bin/env python3
"""
Merge external (Stan/PyMC) reference estimates into an existing ordinal fixture.

Workflow:
1) Create/refresh NextStat fixture:
   PYTHONPATH=bindings/ns-py/python python3 tests/generate_golden_ordinal.py
2) Generate external reference JSON (tool-specific; out of repo scope).
3) Merge:
   python3 tests/external/merge_ordinal_external_goldens.py tests/fixtures/ordinal/ordered_logit_small.json /tmp/external.json

This keeps runtime tests dependency-free: fixtures contain precomputed external
numbers, tests only read JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "usage: merge_ordinal_external_goldens.py <fixture.json> <external.json>",
            file=sys.stderr,
        )
        return 2

    fixture_path = Path(argv[1])
    external_path = Path(argv[2])

    fx = json.loads(fixture_path.read_text())
    ext = json.loads(external_path.read_text())

    if fx.get("kind") != "ordinal_ordered":
        raise SystemExit("fixture kind must be 'ordinal_ordered'")
    if ext.get("tool") not in ("pymc", "stan", "cmdstan", "cmdstanpy"):
        raise SystemExit("external tool must be one of: pymc, stan, cmdstan, cmdstanpy")
    if ext.get("link") not in ("logit", "probit"):
        raise SystemExit("external link must be one of: logit, probit")

    fx["external_reference"] = ext

    fixture_path.write_text(json.dumps(fx, indent=2, sort_keys=True) + "\n")
    print(f"merged external reference into {fixture_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

