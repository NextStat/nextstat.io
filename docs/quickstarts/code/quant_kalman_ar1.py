#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import nextstat


def main() -> int:
    rng = random.Random(42)

    # Simulate a 1D AR(1) latent state with noisy observations:
    #   x[t] = phi * x[t-1] + w[t],  w ~ N(0, q)
    #   y[t] = x[t] + v[t],         v ~ N(0, r)
    phi_true = 0.95
    q_true = 0.05
    r_true = 0.20
    t_max = 250

    x = 0.0
    ys: list[list[float | None]] = []
    for t in range(t_max):
        x = phi_true * x + rng.gauss(0.0, math.sqrt(q_true))
        y = x + rng.gauss(0.0, math.sqrt(r_true))
        ys.append([float(y)])

    # Start from rough guesses and fit Q/R via EM.
    model0 = nextstat.timeseries.ar1_model(phi=phi_true, q=0.5, r=0.5, m0=0.0, p0=1.0)
    # Note: EM convergence is sensitive to tolerance; for a quickstart we use a looser tol
    # so `em_converged` is typically true while still recovering Q/R accurately.
    fit_out = nextstat.timeseries.kalman_fit(
        model0,
        ys,
        max_iter=100,
        tol=1e-4,
        estimate_q=True,
        estimate_r=True,
        forecast_steps=10,
    )

    em = fit_out["em"]
    q_hat = float(em["q"][0][0])
    r_hat = float(em["r"][0][0])
    print("em_converged:", bool(em["converged"]))
    print("em_n_iter:", int(em["n_iter"]))
    print("q_hat:", round(q_hat, 6), "q_true:", q_true)
    print("r_hat:", round(r_hat, 6), "r_true:", r_true)

    out = {
        "phi_true": phi_true,
        "q_true": q_true,
        "r_true": r_true,
        "q_hat": q_hat,
        "r_hat": r_hat,
        "em": em,
        "forecast": fit_out.get("forecast"),
    }
    out_path = Path("docs/quickstarts/out/quant_kalman_ar1_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote:", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
