"""Sanity checks for regression "golden" fixtures.

These tests ensure fixture files are internally consistent and represent
valid optima (near-zero gradients) for their stated objectives.

They intentionally do not depend on the Rust extension module.
"""

import json
import math
from pathlib import Path


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "regression"


def log1pexp(x: float) -> float:
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def sigmoid(x: float) -> float:
    if x >= 0.0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def add_intercept(x):
    return [[1.0] + row for row in x]


def mat_vec_mul(a, x):
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def gradient_inf_norm(x, y):
    xd = add_intercept(x)
    p = len(xd[0])
    beta = y["beta_hat"]
    kind = y["kind"]
    obs = y["y"]
    offset = y.get("offset")

    # residuals for objective gradients
    if kind == "ols":
        r = [pred - o for pred, o in zip(mat_vec_mul(xd, beta), obs)]
        grad = [0.0] * p
        for i in range(len(obs)):
            for j in range(p):
                grad[j] += xd[i][j] * r[i]
        return max(abs(g) for g in grad)

    if kind == "logistic":
        grad = [0.0] * p
        for i in range(len(obs)):
            eta = sum(xd[i][j] * beta[j] for j in range(p))
            mu = sigmoid(eta)
            for j in range(p):
                grad[j] += xd[i][j] * (mu - obs[i])
        return max(abs(g) for g in grad)

    if kind == "poisson":
        grad = [0.0] * p
        for i in range(len(obs)):
            eta = sum(xd[i][j] * beta[j] for j in range(p))
            if offset is not None:
                eta += offset[i]
            mu = math.exp(eta)
            for j in range(p):
                grad[j] += xd[i][j] * (mu - obs[i])
        return max(abs(g) for g in grad)

    raise AssertionError(f"unknown kind: {kind}")


def test_golden_regression_fixtures_exist():
    assert FIXTURES_DIR.is_dir()
    names = {p.name for p in FIXTURES_DIR.glob("*.json")}
    assert {"ols_small.json", "logistic_small.json", "poisson_small.json"} <= names


def test_golden_regression_fixtures_have_near_zero_gradients():
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        assert data["include_intercept"] is True
        assert len(data["x"]) == data["n"]
        assert len(data["y"]) == data["n"]
        assert len(data["x"][0]) == data["p"]
        assert len(data["beta_hat"]) == data["p"] + 1  # + intercept

        g_inf = gradient_inf_norm(data["x"], data)
        assert g_inf < 1e-8, f"{path.name}: gradient too large: {g_inf}"


def test_golden_regression_fixtures_nll_matches_recompute():
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        xd = add_intercept(data["x"])
        beta = data["beta_hat"]
        y = data["y"]
        kind = data["kind"]
        offset = data.get("offset")

        if kind == "ols":
            resid = [pred - obs for pred, obs in zip(mat_vec_mul(xd, beta), y)]
            nll = 0.5 * sum(r * r for r in resid)
        elif kind == "logistic":
            nll = 0.0
            for i in range(len(y)):
                eta = sum(xd[i][j] * beta[j] for j in range(len(beta)))
                nll += log1pexp(eta) - y[i] * eta
        elif kind == "poisson":
            nll = 0.0
            for i in range(len(y)):
                eta = sum(xd[i][j] * beta[j] for j in range(len(beta)))
                if offset is not None:
                    eta += offset[i]
                mu = math.exp(eta)
                nll += mu - y[i] * eta
        else:
            raise AssertionError(f"unknown kind: {kind}")

        assert abs(nll - data["nll_at_hat"]) < 1e-8, f"{path.name}: nll mismatch"

