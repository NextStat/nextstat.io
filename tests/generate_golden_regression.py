#!/usr/bin/env python3
"""
Generate small, deterministic "golden" regression problems.

These fixtures are used to validate frequentist regression implementations
(OLS / logistic / Poisson) without relying on external libraries at test time.

The generator itself uses only the Python standard library so it can run
in restricted environments.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[0]
FIXTURES_DIR = ROOT / "fixtures" / "regression"


# ---------------------------------------------------------------------------
# Small linear algebra helpers (pure Python, p is small)
# ---------------------------------------------------------------------------

Vector = List[float]
Matrix = List[List[float]]


def mat_t(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def mat_vec_mul(a: Matrix, x: Vector) -> Vector:
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    bt = mat_t(b)
    return [[sum(ai * bj for ai, bj in zip(row, col)) for col in bt] for row in a]


def vec_add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def vec_sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def vec_norm_inf(x: Vector) -> float:
    return max(abs(v) for v in x) if x else 0.0


def solve_linear(a: Matrix, b: Vector) -> Vector:
    """
    Solve a x = b by Gaussian elimination with partial pivoting.

    Assumes a is square and reasonably well-conditioned (golden fixtures keep p small).
    """
    n = len(a)
    if n == 0:
        return []
    if any(len(row) != n for row in a):
        raise ValueError("a must be square")
    if len(b) != n:
        raise ValueError("b has wrong length")

    # Make a working copy
    m = [row[:] + [bi] for row, bi in zip(a, b)]

    for col in range(n):
        # Pivot row
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-15:
            raise ValueError("singular matrix")
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        # Eliminate
        piv = m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / piv
            if f == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]

    # Back-substitute
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / m[i][i]
    return x


def mat_inv(a: Matrix) -> Matrix:
    """Invert a small matrix by solving A x = e_i for each basis vector."""
    n = len(a)
    if n == 0:
        return []
    if any(len(row) != n for row in a):
        raise ValueError("a must be square")
    cols: Matrix = []
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        cols.append(solve_linear(a, e))
    return mat_t(cols)


def add_intercept(x: Matrix) -> Matrix:
    return [[1.0] + row[:] for row in x]


# ---------------------------------------------------------------------------
# Stable scalar functions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model-specific objectives
# ---------------------------------------------------------------------------


def ols_fit(x: Matrix, y: Vector, include_intercept: bool) -> Vector:
    xd = add_intercept(x) if include_intercept else x
    xt = mat_t(xd)
    xtx = mat_mul(xt, xd)
    xty = [sum(xt[i][k] * y[k] for k in range(len(y))) for i in range(len(xt))]
    return solve_linear(xtx, xty)


def logistic_nll_grad_hess(xd: Matrix, y: Vector, beta: Vector) -> Tuple[float, Vector, Matrix]:
    n = len(y)
    p = len(beta)
    nll = 0.0
    grad = [0.0] * p
    hess = [[0.0] * p for _ in range(p)]
    for i in range(n):
        eta = sum(xd[i][j] * beta[j] for j in range(p))
        mu = sigmoid(eta)
        # nll = sum log(1+exp(eta)) - y*eta
        nll += log1pexp(eta) - y[i] * eta
        w = mu * (1.0 - mu)
        for j in range(p):
            grad[j] += xd[i][j] * (mu - y[i])
        for a in range(p):
            xa = xd[i][a]
            for b in range(p):
                hess[a][b] += w * xa * xd[i][b]
    return nll, grad, hess


def poisson_nll_grad_hess(
    xd: Matrix,
    y: Vector,
    beta: Vector,
    offset: Optional[Vector],
) -> Tuple[float, Vector, Matrix]:
    n = len(y)
    p = len(beta)
    nll = 0.0
    grad = [0.0] * p
    hess = [[0.0] * p for _ in range(p)]
    for i in range(n):
        eta = sum(xd[i][j] * beta[j] for j in range(p))
        if offset is not None:
            eta += offset[i]
        mu = math.exp(eta)
        # nll = sum mu - y*eta  (ignoring constant log(y!))
        nll += mu - y[i] * eta
        for j in range(p):
            grad[j] += xd[i][j] * (mu - y[i])
        for a in range(p):
            xa = xd[i][a]
            for b in range(p):
                hess[a][b] += mu * xa * xd[i][b]
    return nll, grad, hess


def negbin_nll(
    xd: Matrix,
    y: Vector,
    beta: Vector,
    log_alpha: float,
    offset: Optional[Vector],
) -> float:
    """
    NB2 negative log-likelihood (mean/dispersion alpha) ignoring constant log(y!).

    Parameterization:
      mu = exp(eta), alpha = exp(log_alpha), theta = 1/alpha
      log p = lgamma(y+theta) - lgamma(theta) + theta ln(theta/(theta+mu)) + y ln(mu/(theta+mu))
    """
    alpha = math.exp(log_alpha)
    if not (alpha > 0.0) or not math.isfinite(alpha):
        return float("inf")
    theta = 1.0 / alpha

    nll = 0.0
    for i in range(len(y)):
        eta = sum(xd[i][j] * beta[j] for j in range(len(beta)))
        if offset is not None:
            eta += offset[i]
        mu = math.exp(eta)
        yi = float(y[i])
        # Ignore constant -lgamma(y+1).
        logp = (
            math.lgamma(yi + theta)
            - math.lgamma(theta)
            + theta * (math.log(theta) - math.log(theta + mu))
            + yi * (math.log(mu) - math.log(theta + mu))
        )
        nll -= logp
    return nll


def negbin_nll_grad_hess_beta(
    xd: Matrix,
    y: Vector,
    beta: Vector,
    log_alpha: float,
    offset: Optional[Vector],
) -> Tuple[float, Vector, Matrix]:
    """NB2 nll/grad/hess with respect to beta only (log_alpha treated as fixed)."""
    alpha = math.exp(log_alpha)
    if not (alpha > 0.0) or not math.isfinite(alpha):
        n = len(y)
        p = len(beta)
        return float("inf"), [float("nan")] * p, [[float("nan")] * p for _ in range(p)]
    theta = 1.0 / alpha

    n = len(y)
    p = len(beta)
    nll = 0.0
    grad = [0.0] * p
    hess = [[0.0] * p for _ in range(p)]
    for i in range(n):
        eta = sum(xd[i][j] * beta[j] for j in range(p))
        if offset is not None:
            eta += offset[i]
        mu = math.exp(eta)
        yi = float(y[i])

        # nll contribution (ignore constant -lgamma(y+1))
        logp = (
            math.lgamma(yi + theta)
            - math.lgamma(theta)
            + theta * (math.log(theta) - math.log(theta + mu))
            + yi * (math.log(mu) - math.log(theta + mu))
        )
        nll -= logp

        # d/deta nll = mu * (theta + y) / (theta + mu) - y
        err = mu * (theta + yi) / (theta + mu) - yi
        for j in range(p):
            grad[j] += xd[i][j] * err

        # d/deta err = mu * theta * (theta + y) / (theta + mu)^2
        w = mu * theta * (theta + yi) / ((theta + mu) ** 2)
        for a in range(p):
            xa = xd[i][a]
            for b in range(p):
                hess[a][b] += w * xa * xd[i][b]
    return nll, grad, hess


def newton_solve(
    nll_grad_hess_fn,
    beta0: Vector,
    max_iter: int = 100,
    tol: float = 1e-10,
    damping: float = 1.0,
) -> Tuple[Vector, float]:
    beta = beta0[:]
    nll, grad, hess = nll_grad_hess_fn(beta)
    for _ in range(max_iter):
        if vec_norm_inf(grad) < tol:
            break

        # Regularize Hessian slightly to avoid rare near-singular steps.
        lam = 1e-9
        hreg = [row[:] for row in hess]
        for i in range(len(hreg)):
            hreg[i][i] += lam

        step = solve_linear(hreg, grad)  # beta_new = beta - step

        # Backtracking line search on nll.
        alpha = damping
        while alpha > 1e-6:
            cand = [b - alpha * s for b, s in zip(beta, step)]
            cand_nll, cand_grad, cand_hess = nll_grad_hess_fn(cand)
            if cand_nll <= nll:
                beta, nll, grad, hess = cand, cand_nll, cand_grad, cand_hess
                break
            alpha *= 0.5
        else:
            # No progress; stop to keep deterministic output.
            break
    return beta, nll


def poisson_knuth(lam: float, rng: random.Random) -> int:
    # Knuth algorithm; fine for the small rates used in fixtures.
    if lam <= 0.0:
        return 0
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= rng.random()
    return k - 1


# ---------------------------------------------------------------------------
# Fixture assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fixture:
    name: str
    kind: str
    x: Matrix
    y: Vector
    include_intercept: bool
    beta_true: Vector
    beta_hat: Vector
    nll_at_hat: float
    log_alpha_true: Optional[float] = None
    log_alpha_hat: Optional[float] = None
    offset: Optional[Vector] = None
    se_hat: Optional[Vector] = None
    cov_hat: Optional[Matrix] = None
    sigma2_hat: Optional[float] = None

    def to_json(self) -> dict:
        out = {
            "name": self.name,
            "kind": self.kind,
            "n": len(self.y),
            "p": len(self.x[0]) if self.x else 0,
            "include_intercept": self.include_intercept,
            "x": self.x,
            "y": self.y,
            "beta_true": self.beta_true,
            "beta_hat": self.beta_hat,
            "nll_at_hat": self.nll_at_hat,
        }
        if self.log_alpha_true is not None:
            out["log_alpha_true"] = self.log_alpha_true
            out["alpha_true"] = math.exp(self.log_alpha_true)
        if self.log_alpha_hat is not None:
            out["log_alpha_hat"] = self.log_alpha_hat
            out["alpha_hat"] = math.exp(self.log_alpha_hat)
        if self.offset is not None:
            out["offset"] = self.offset
        if self.se_hat is not None:
            out["se_hat"] = self.se_hat
        if self.cov_hat is not None:
            out["cov_hat"] = self.cov_hat
        if self.sigma2_hat is not None:
            out["sigma2_hat"] = self.sigma2_hat
        return out


def build_fixtures() -> List[Fixture]:
    rng = random.Random(12345)

    # OLS (Gaussian)
    n_ols = 40
    p_ols = 3
    x_ols = [[rng.uniform(-2.0, 2.0) for _ in range(p_ols)] for _ in range(n_ols)]
    beta_true_ols = [1.5, 0.7, -1.2, 0.4]  # intercept + 3 weights
    y_ols = []
    for row in x_ols:
        eta = beta_true_ols[0] + sum(b * x for b, x in zip(beta_true_ols[1:], row))
        y_ols.append(eta + rng.gauss(0.0, 0.3))
    beta_hat_ols = ols_fit(x_ols, y_ols, include_intercept=True)
    # nll_at_hat is not meaningful without sigma; store SSE/2 for a stable scalar.
    xd_ols = add_intercept(x_ols)
    resid = vec_sub(mat_vec_mul(xd_ols, beta_hat_ols), y_ols)
    nll_ols = 0.5 * sum(r * r for r in resid)
    # Standard OLS sigma^2 estimate and covariance.
    k_ols = len(beta_hat_ols)
    sigma2_hat_ols = sum(r * r for r in resid) / float(n_ols - k_ols)
    xt_ols = mat_t(xd_ols)
    xtx_ols = mat_mul(xt_ols, xd_ols)
    cov_ols = mat_inv(xtx_ols)
    cov_ols = [[sigma2_hat_ols * v for v in row] for row in cov_ols]
    se_ols = [math.sqrt(cov_ols[i][i]) for i in range(k_ols)]

    fixtures: List[Fixture] = [
        Fixture(
            name="ols_small",
            kind="ols",
            x=x_ols,
            y=y_ols,
            include_intercept=True,
            beta_true=beta_true_ols,
            beta_hat=beta_hat_ols,
            nll_at_hat=nll_ols,
            se_hat=se_ols,
            cov_hat=cov_ols,
            sigma2_hat=sigma2_hat_ols,
        )
    ]

    # Logistic regression (Bernoulli)
    n_log = 60
    p_log = 2
    x_log = [[rng.uniform(-1.5, 1.5) for _ in range(p_log)] for _ in range(n_log)]
    beta_true_log = [-0.2, 0.3, -0.8]  # intercept + 2 weights
    y_log = []
    for row in x_log:
        eta = beta_true_log[0] + sum(b * x for b, x in zip(beta_true_log[1:], row))
        p = sigmoid(eta)
        y_log.append(1.0 if rng.random() < p else 0.0)
    xd_log = add_intercept(x_log)

    def log_obj(beta: Vector):
        return logistic_nll_grad_hess(xd_log, y_log, beta)

    beta0_log = [0.0] * (p_log + 1)
    beta_hat_log, nll_log = newton_solve(log_obj, beta0_log, max_iter=100, tol=1e-10)
    _nll, _grad, hess_log = logistic_nll_grad_hess(xd_log, y_log, beta_hat_log)
    cov_log = mat_inv(hess_log)
    se_log = [math.sqrt(cov_log[i][i]) for i in range(len(cov_log))]

    fixtures.append(
        Fixture(
            name="logistic_small",
            kind="logistic",
            x=x_log,
            y=y_log,
            include_intercept=True,
            beta_true=beta_true_log,
            beta_hat=beta_hat_log,
            nll_at_hat=nll_log,
            se_hat=se_log,
            cov_hat=cov_log,
        )
    )

    # Poisson regression (counts) without offset
    n_pois = 60
    p_pois = 2
    x_pois = [[rng.uniform(-1.0, 1.0) for _ in range(p_pois)] for _ in range(n_pois)]
    beta_true_pois = [0.2, 0.4, -0.3]  # intercept + 2 weights
    y_pois = []
    for row in x_pois:
        eta = beta_true_pois[0] + sum(b * x for b, x in zip(beta_true_pois[1:], row))
        lam = math.exp(eta)
        y_pois.append(float(poisson_knuth(lam, rng)))
    xd_pois = add_intercept(x_pois)

    def pois_obj(beta: Vector):
        return poisson_nll_grad_hess(xd_pois, y_pois, beta, offset=None)

    beta0_pois = [0.0] * (p_pois + 1)
    beta_hat_pois, nll_pois = newton_solve(pois_obj, beta0_pois, max_iter=100, tol=1e-10)
    _nll, _grad, hess_pois = poisson_nll_grad_hess(xd_pois, y_pois, beta_hat_pois, offset=None)
    cov_pois = mat_inv(hess_pois)
    se_pois = [math.sqrt(cov_pois[i][i]) for i in range(len(cov_pois))]

    fixtures.append(
        Fixture(
            name="poisson_small",
            kind="poisson",
            x=x_pois,
            y=y_pois,
            include_intercept=True,
            beta_true=beta_true_pois,
            beta_hat=beta_hat_pois,
            nll_at_hat=nll_pois,
            se_hat=se_pois,
            cov_hat=cov_pois,
        )
    )

    # Negative binomial regression (NB2 mean/dispersion) without offset.
    n_nb = 80
    p_nb = 2
    x_nb = [[rng.uniform(-1.0, 1.0) for _ in range(p_nb)] for _ in range(n_nb)]
    beta_true_nb = [0.1, 0.6, -0.4]  # intercept + 2 weights
    alpha_true_nb = 0.7
    y_nb = []
    for row in x_nb:
        eta = beta_true_nb[0] + sum(b * x for b, x in zip(beta_true_nb[1:], row))
        mu = math.exp(eta)
        # NB2 via Gamma-Poisson mixture:
        # lambda ~ Gamma(shape=1/alpha, scale=alpha*mu), y ~ Poisson(lambda).
        shape = 1.0 / alpha_true_nb
        scale = alpha_true_nb * mu
        lam = rng.gammavariate(shape, scale)
        y_nb.append(float(poisson_knuth(lam, rng)))
    xd_nb = add_intercept(x_nb)

    # Alternating Newton (beta) + 1D Newton (log_alpha with numeric derivatives).
    beta0_nb = [0.0] * (p_nb + 1)
    log_alpha_nb = math.log(alpha_true_nb)
    beta_nb = beta0_nb[:]
    for _ in range(25):
        def nb_obj(beta: Vector):
            return negbin_nll_grad_hess_beta(xd_nb, y_nb, beta, log_alpha_nb, offset=None)

        beta_nb, _nll_beta = newton_solve(nb_obj, beta_nb, max_iter=80, tol=1e-10)

        # Numeric derivatives in log_alpha.
        h = 1e-4
        f0 = negbin_nll(xd_nb, y_nb, beta_nb, log_alpha_nb, offset=None)
        fp = negbin_nll(xd_nb, y_nb, beta_nb, log_alpha_nb + h, offset=None)
        fm = negbin_nll(xd_nb, y_nb, beta_nb, log_alpha_nb - h, offset=None)
        g = (fp - fm) / (2.0 * h)
        hh = (fp - 2.0 * f0 + fm) / (h * h)
        if not math.isfinite(g) or not math.isfinite(hh) or abs(hh) < 1e-12:
            break
        step = g / hh
        # Backtracking line search on nll.
        alpha_ls = 1.0
        while alpha_ls > 1e-6:
            cand = log_alpha_nb - alpha_ls * step
            fc = negbin_nll(xd_nb, y_nb, beta_nb, cand, offset=None)
            if fc <= f0:
                log_alpha_nb = cand
                break
            alpha_ls *= 0.5
        else:
            break
        if abs(step) < 1e-10:
            break

    nll_nb = negbin_nll(xd_nb, y_nb, beta_nb, log_alpha_nb, offset=None)

    _nll, _grad, hess_beta_nb = negbin_nll_grad_hess_beta(
        xd_nb, y_nb, beta_nb, log_alpha_nb, offset=None
    )
    cov_nb = mat_inv(hess_beta_nb)
    se_nb = [math.sqrt(cov_nb[i][i]) for i in range(len(cov_nb))]

    fixtures.append(
        Fixture(
            name="negbin_small",
            kind="negbin",
            x=x_nb,
            y=y_nb,
            include_intercept=True,
            beta_true=beta_true_nb,
            beta_hat=beta_nb,
            log_alpha_true=math.log(alpha_true_nb),
            log_alpha_hat=log_alpha_nb,
            nll_at_hat=nll_nb,
            se_hat=se_nb,
            cov_hat=cov_nb,
        )
    )

    return fixtures


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = build_fixtures()
    for fx in fixtures:
        path = FIXTURES_DIR / f"{fx.name}.json"
        path.write_text(json.dumps(fx.to_json(), indent=2, sort_keys=True) + "\n")
        print(f"Wrote {path.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
