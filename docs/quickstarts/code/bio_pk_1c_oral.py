#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from pathlib import Path

import nextstat


def main() -> int:
    rng = random.Random(42)

    # A small PK dataset: oral dosing, first-order absorption.
    times = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 24.0]
    dose = 100.0
    sigma = 0.20

    # Model parameters (CL, V, ka) in NextStat's convention for OneCompartmentOralPkModel.
    true_params = [1.2, 12.0, 0.9]

    m0 = nextstat.OneCompartmentOralPkModel(times, [0.0] * len(times), dose=dose, sigma=sigma)
    mu = m0.predict(true_params)
    y = [max(0.0, float(mu_i + rng.gauss(0.0, sigma))) for mu_i in mu]

    model = nextstat.OneCompartmentOralPkModel(times, y, dose=dose, sigma=sigma)

    mle = nextstat.MaximumLikelihoodEstimator()
    fit = mle.fit(model)

    est = [float(v) for v in fit.parameters]
    se = [float(v) for v in fit.uncertainties]
    print("converged:", bool(fit.converged))
    print("nll:", float(fit.nll))
    print("params_hat:", [round(v, 6) for v in est])
    print("params_se:", [round(v, 6) for v in se])

    pred = model.predict(est)
    out = {
        "times": [float(t) for t in times],
        "dose": dose,
        "sigma": sigma,
        "y": [float(v) for v in y],
        "true_params": [float(v) for v in true_params],
        "fit": {
            "converged": bool(fit.converged),
            "nll": float(fit.nll),
            "params_hat": est,
            "params_se": se,
        },
        "pred_hat": [float(v) for v in pred],
    }
    out_path = Path("docs/quickstarts/out/bio_pk_1c_oral_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote:", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

