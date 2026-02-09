#!/usr/bin/env python3
from __future__ import annotations

import math
import random

import nextstat


def sigmoid(x: float) -> float:
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def log_loss(y: list[int], p: list[float], eps: float = 1e-12) -> float:
    s = 0.0
    for yi, pi in zip(y, p):
        pi2 = min(1.0 - eps, max(eps, float(pi)))
        s += -(yi * math.log(pi2) + (1 - yi) * math.log(1.0 - pi2))
    return s / float(len(y) or 1)


def main() -> int:
    rng = random.Random(42)

    # Synthetic dataset: 2 features, 1 intercept.
    n = 500
    true_w = [-0.25, 1.5, -0.8]  # [Intercept, x1, x2]

    x: list[list[float]] = []
    y: list[int] = []
    for _ in range(n):
        x1 = rng.gauss(0.0, 1.0)
        x2 = rng.gauss(0.0, 1.0)
        eta = true_w[0] + true_w[1] * x1 + true_w[2] * x2
        p = sigmoid(eta)
        yi = 1 if rng.random() < p else 0
        x.append([x1, x2])
        y.append(yi)

    fit = nextstat.glm.logistic.fit(x, y, include_intercept=True, l2=1.0)

    probs = fit.predict_proba(x)
    preds = fit.predict(x)
    acc = sum(1 for yi, pi in zip(y, preds) if yi == pi) / float(n)

    print("converged:", fit.converged)
    print("warnings:", fit.warnings)
    print("coef:", [round(v, 4) for v in fit.coef])
    print("se:", [round(v, 4) for v in fit.standard_errors])
    print("accuracy:", round(acc, 4))
    print("log_loss:", round(log_loss(y, probs), 6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

