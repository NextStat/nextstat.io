#!/usr/bin/env python3
"""
Merge external (lme4) reference estimates into an existing NextStat LMM fixture.

Workflow:
1) Generate lme4 reference:
   Rscript tests/external/generate_lmm_goldens_lme4.R tests/fixtures/lmm/lmm_intercept_small.json > /tmp/lme4.json
2) Merge into fixture:
   python3 tests/external/merge_lmm_external_goldens.py tests/fixtures/lmm/lmm_intercept_small.json /tmp/lme4.json

This keeps tests runtime dependency-free: fixtures contain precomputed external
numbers, tests only read JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: merge_lmm_external_goldens.py <fixture.json> <external.json>", file=sys.stderr)
        return 2

    fixture_path = Path(argv[1])
    external_path = Path(argv[2])

    fx = json.loads(fixture_path.read_text())
    ext = json.loads(external_path.read_text())

    if fx.get("kind") != "lmm_marginal":
        raise SystemExit("fixture kind must be 'lmm_marginal'")
    if ext.get("tool") not in ("lme4", "stan", "cmdstan", "pymer4"):
        raise SystemExit("external tool must be one of: lme4, stan, cmdstan, pymer4")

    fx["external_reference"] = ext

    fixture_path.write_text(json.dumps(fx, indent=2, sort_keys=True) + "\n")
    print(f"merged external reference into {fixture_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

