#!/usr/bin/env python3
"""Generate a pyhf-based golden regression corpus for HistFactory parity.

Why:
  - Existing parity tests compare NextStat vs pyhf at runtime and therefore
    require `pyhf` as a test dependency.
  - For a lightweight regression harness, we record a small deterministic set
    of parameter points and the corresponding pyhf reference outputs once, and
    assert NextStat matches those goldens without importing pyhf.

Run (from repo root):
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/python/generate_pyhf_model_zoo_goldens.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyhf

PY_TESTS_DIR = Path(__file__).resolve().parent
if str(PY_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(PY_TESTS_DIR))

from _pyhf_model_zoo import (  # noqa: E402
    make_synthetic_shapesys_workspace,
    make_workspace_histo_normsys_staterror,
    make_workspace_multichannel,
    make_workspace_shapefactor_control_region,
)

REPO = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO / "tests" / "fixtures"
OUT_PATH = FIXTURES_DIR / "pyhf_model_zoo_goldens.json"


@dataclass(frozen=True)
class Case:
    name: str
    workspace: dict[str, Any]
    measurement: str


def _load_fixture(rel: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / rel).read_text())


def _pyhf_model_and_data(workspace: dict[str, Any], measurement_name: str):
    ws = pyhf.Workspace(workspace)
    model = ws.model(
        measurement_name=measurement_name,
        modifier_settings={
            "normsys": {"interpcode": "code4"},
            "histosys": {"interpcode": "code4p"},
        },
    )
    data = ws.data(model)
    return model, data


def _pyhf_twice_nll(model, data, params) -> float:
    return float(pyhf.infer.mle.twice_nll(params, data, model).item())


def _sample_params(rng: random.Random, init: list[float], bounds: list[tuple[float, float]]):
    out: list[float] = []
    for x0, (lo, hi) in zip(init, bounds):
        lo_f = float(lo)
        hi_f = float(hi)
        if not (lo_f < hi_f):
            out.append(float(x0))
            continue
        # Sample in a tight-ish region around init to avoid extreme tails.
        span = hi_f - lo_f
        center = min(max(float(x0), lo_f), hi_f)
        half = 0.25 * span
        a = max(lo_f, center - half)
        b = min(hi_f, center + half)
        if not (a < b):
            a, b = lo_f, hi_f
        out.append(rng.uniform(a, b))
    return out


def _make_suite() -> list[Case]:
    return [
        Case("fixture_simple", _load_fixture("simple_workspace.json"), "GaussExample"),
        Case("fixture_complex", _load_fixture("complex_workspace.json"), "measurement"),
        Case("fixture_histfactory_nominal", _load_fixture("histfactory/workspace.json"), "NominalMeasurement"),
        Case("zoo_multichannel_3", make_workspace_multichannel(3), "m"),
        Case("zoo_histo_normsys_staterror_10", make_workspace_histo_normsys_staterror(10), "m"),
        Case("zoo_shapefactor_control_4", make_workspace_shapefactor_control_region(4), "m"),
        Case("synthetic_shapesys_16", make_synthetic_shapesys_workspace(16), "m"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-random", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    suite = _make_suite()

    out: dict[str, Any] = {
        "version": 1,
        "meta": {
            "pyhf_version": pyhf.__version__,
            "seed": int(args.seed),
            "n_random": int(args.n_random),
            "sampler": "tight_init_box_v1",
        },
        "cases": [],
    }

    for case in suite:
        model, data = _pyhf_model_and_data(case.workspace, case.measurement)
        init = list(map(float, model.config.suggested_init()))
        bounds = [(float(a), float(b)) for a, b in model.config.suggested_bounds()]

        points: list[dict[str, Any]] = []
        points.append({"label": "suggested_init", "params": init})

        for i in range(args.n_random):
            p = _sample_params(rng, init, bounds)
            points.append({"label": f"random_{i}", "params": p})

        poi_idx = model.config.poi_index
        if poi_idx is not None:
            for mu in [0.0, 0.5, 2.0]:
                p = list(init)
                p[int(poi_idx)] = float(mu)
                points.append({"label": f"poi_mu_{mu}", "params": p})

        # Evaluate pyhf reference for each point.
        for pt in points:
            params = pt["params"]
            pt["expected_full"] = [float(x) for x in model.expected_data(params)]
            pt["expected_main"] = [float(x) for x in model.expected_data(params, include_auxdata=False)]
            pt["twice_nll"] = _pyhf_twice_nll(model, data, params)

        out["cases"].append(
            {
                "name": case.name,
                "measurement": case.measurement,
                "par_names": list(model.config.par_names),
                "workspace": case.workspace,
                "points": points,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUT_PATH} with {len(out['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

