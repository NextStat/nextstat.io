"""Sanity checks for ordinal (ordered outcomes) golden fixtures.

These tests validate fixture internal consistency without relying on the Rust
extension module at runtime.
"""

from __future__ import annotations

import json
import math
from pathlib import Path


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ordinal"


def softplus(x: float) -> float:
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def log1pexp(x: float) -> float:
    if x > 0.0:
        return x + math.log1p(math.exp(-x))
    return math.log1p(math.exp(x))


def log_sigmoid(x: float) -> float:
    return -softplus(-x)


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def Phi(x: float) -> float:
    return 0.5 * math.erfc(-float(x) / math.sqrt(2.0))


def cutpoints_from_raw(raw: list[float]) -> list[float]:
    if not raw:
        return []
    c = [float(raw[0])]
    for r in raw[1:]:
        c.append(float(c[-1] + softplus(float(r))))
    return c


def logdiffexp(a: float, b: float) -> float:
    # log(exp(a) - exp(b)), assuming a>b.
    if not (a > b):
        raise ValueError("logdiffexp requires a>b")
    return float(a + math.log1p(-math.exp(float(b - a))))


def ordered_logit_logp(*, eta: float, cutpoints: list[float], y: int) -> float:
    k = len(cutpoints) + 1
    if not (0 <= y < k):
        raise ValueError("y out of range")

    if y == 0:
        return log_sigmoid(float(cutpoints[0]) - eta)
    if y == k - 1:
        return log_sigmoid(eta - float(cutpoints[-1]))

    a = float(cutpoints[y]) - eta
    b = float(cutpoints[y - 1]) - eta
    la = log_sigmoid(a)
    lb = log_sigmoid(b)
    return logdiffexp(la, lb)


def ordered_probit_logp(*, eta: float, cutpoints: list[float], y: int) -> float:
    k = len(cutpoints) + 1
    if not (0 <= y < k):
        raise ValueError("y out of range")

    if y == 0:
        p = Phi(float(cutpoints[0]) - eta)
    elif y == k - 1:
        p = 1.0 - Phi(float(cutpoints[-1]) - eta)
    else:
        hi = Phi(float(cutpoints[y]) - eta)
        lo = Phi(float(cutpoints[y - 1]) - eta)
        p = hi - lo

    # Fixtures keep this away from extreme tails, but guard anyway.
    p = max(float(p), 1e-300)
    return float(math.log(p))


def predict_proba(*, link: str, coef: list[float], cutpoints: list[float], x: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for row in x:
        eta = sum(float(a) * float(b) for a, b in zip(row, coef))
        if link == "logit":
            cdf = [sigmoid(float(cutpoints[0]) - eta)]
            for ck in cutpoints[1:]:
                cdf.append(sigmoid(float(ck) - eta))
        elif link == "probit":
            cdf = [Phi(float(cutpoints[0]) - eta)]
            for ck in cutpoints[1:]:
                cdf.append(Phi(float(ck) - eta))
        else:
            raise ValueError("unknown link")

        probs = [float(cdf[0])]
        for j in range(1, len(cdf)):
            probs.append(float(cdf[j] - cdf[j - 1]))
        probs.append(float(1.0 - cdf[-1]))

        probs = [0.0 if p < 0.0 and p > -1e-12 else float(p) for p in probs]
        s = sum(probs)
        probs = [p / s for p in probs]
        out.append(probs)
    return out


def test_golden_ordinal_fixtures_exist():
    assert FIXTURES_DIR.is_dir()
    names = {p.name for p in FIXTURES_DIR.glob("*.json")}
    assert {"ordered_logit_small.json", "ordered_probit_small.json"} <= names


def test_golden_ordinal_fixtures_nll_matches_recompute():
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fx = json.loads(path.read_text())
        assert fx["kind"] == "ordinal_ordered"
        link = fx["link"]
        x = fx["x"]
        y = fx["y"]
        n_levels = int(fx["n_levels"])
        assert n_levels >= 2
        assert len(x) == len(y) == int(fx["n"])

        params = list(map(float, fx["params_hat"]))
        p = len(x[0])
        coef = params[:p]
        cut_raw = params[p:]
        assert len(cut_raw) == n_levels - 1
        cutpoints = cutpoints_from_raw(cut_raw)

        nll = 0.0
        for row, yi in zip(x, y):
            eta = sum(float(a) * float(b) for a, b in zip(row, coef))
            if link == "logit":
                lp = ordered_logit_logp(eta=eta, cutpoints=cutpoints, y=int(yi))
            elif link == "probit":
                lp = ordered_probit_logp(eta=eta, cutpoints=cutpoints, y=int(yi))
            else:
                raise AssertionError("unknown link")
            nll -= float(lp)

        assert abs(nll - float(fx["nll_at_hat"])) < 1e-6, f"{path.name}: nll mismatch"


def test_golden_ordinal_fixtures_derived_fields_consistent():
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fx = json.loads(path.read_text())
        d = fx["derived"]
        link = fx["link"]
        params = list(map(float, fx["params_hat"]))
        p = len(fx["x"][0])
        coef = params[:p]
        cut_raw = params[p:]
        cutpoints = cutpoints_from_raw(cut_raw)

        assert all(float(cutpoints[i]) < float(cutpoints[i + 1]) for i in range(len(cutpoints) - 1))
        for a, b in zip(d["cutpoints"], cutpoints):
            assert abs(float(a) - float(b)) < 1e-10

        grid_pred = predict_proba(link=link, coef=coef, cutpoints=cutpoints, x=fx["grid_x"])
        got = d["grid_pred_proba"]
        assert len(got) == len(grid_pred)
        for r0, r1 in zip(got, grid_pred):
            assert len(r0) == len(r1) == int(fx["n_levels"])
            for a, b in zip(r0, r1):
                assert abs(float(a) - float(b)) < 1e-10


def test_golden_ordinal_external_reference_parity_when_present():
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fx = json.loads(path.read_text())
        ext = fx.get("external_reference")
        if not isinstance(ext, dict):
            continue

        if ext.get("tool") not in ("pymc", "stan", "cmdstan", "cmdstanpy"):
            raise AssertionError(f"{path.name}: unsupported external tool")
        if ext.get("link") != fx.get("link"):
            raise AssertionError(f"{path.name}: external link mismatch")

        exp = fx["derived"]["grid_pred_proba"]
        got = ext.get("grid_pred_proba")
        if not isinstance(got, list):
            raise AssertionError(f"{path.name}: external_reference.grid_pred_proba missing")

        # External inference is approximate (MCMC/optimization). Keep this forgiving.
        tol = 0.10
        max_abs = 0.0
        for r_exp, r_got in zip(exp, got):
            for a, b in zip(r_exp, r_got):
                max_abs = max(max_abs, abs(float(a) - float(b)))
        assert max_abs < tol, f"{path.name}: external predict_proba mismatch: max|Δ|={max_abs}"

