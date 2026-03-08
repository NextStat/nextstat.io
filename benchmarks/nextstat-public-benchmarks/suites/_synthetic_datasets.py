"""Shared synthetic dataset builders for public benchmark suites."""

from __future__ import annotations

import math
import random
from typing import Any


def logistic(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def make_logistic_regression_dataset(*, n: int, p: int, seed: int) -> dict[str, Any]:
    rng = random.Random(int(seed))
    beta = [0.6, -1.1, 0.3, 0.0, 0.8][:p]
    intercept = -0.2

    x: list[list[float]] = []
    y: list[int] = []
    for _ in range(int(n)):
        row = [rng.gauss(0.0, 1.0) for _ in range(int(p))]
        z = intercept + sum(b * v for b, v in zip(beta, row))
        pr = logistic(z)
        yi = 1 if rng.random() < pr else 0
        x.append(row)
        y.append(int(yi))

    return {
        "kind": "logistic_regression",
        "n": int(n),
        "p": int(p),
        "seed": int(seed),
        "beta": beta,
        "intercept": float(intercept),
        "x": x,
        "y": y,
    }


def make_hier_random_intercept_dataset(*, n_groups: int, n_per_group: int, seed: int) -> dict[str, Any]:
    rng = random.Random(int(seed))
    beta = [1.0]
    intercept = 0.0
    sigma_alpha = 1.0

    group_alpha = [rng.gauss(0.0, sigma_alpha) for _ in range(int(n_groups))]

    x: list[list[float]] = []
    y: list[int] = []
    group_idx: list[int] = []
    for g in range(int(n_groups)):
        for _ in range(int(n_per_group)):
            row = [rng.gauss(0.0, 1.0)]
            z = intercept + group_alpha[g] + beta[0] * row[0]
            pr = logistic(z)
            yi = 1 if rng.random() < pr else 0
            x.append(row)
            y.append(int(yi))
            group_idx.append(int(g))

    return {
        "kind": "hier_logistic_random_intercept",
        "n_groups": int(n_groups),
        "n_per_group": int(n_per_group),
        "seed": int(seed),
        "beta": beta,
        "intercept": float(intercept),
        "sigma_alpha": float(sigma_alpha),
        "x": x,
        "y": y,
        "group_idx": group_idx,
    }


def make_eight_schools_dataset() -> dict[str, Any]:
    return {
        "kind": "eight_schools",
        "J": 8,
        "y": [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0],
        "sigma": [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0],
    }
