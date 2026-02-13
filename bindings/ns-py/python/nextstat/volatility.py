"""Volatility models for financial time series.

Provides GARCH(1,1) and approximate stochastic volatility (SV) estimation.
The heavy lifting lives in Rust (``ns-inference::timeseries::volatility``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .timeseries import garch11_fit as _garch11_fit
from .timeseries import sv_logchi2_fit as _sv_logchi2_fit


def garch(
    ys: Sequence[float],
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
    alpha_beta_max: float = 0.999,
    min_var: float = 1e-18,
) -> Mapping[str, Any]:
    """Fit a Gaussian GARCH(1,1) model by maximum likelihood.

    Parameters
    ----------
    ys : array-like of float
        Return series (e.g. log-returns).
    max_iter : int
        Maximum L-BFGS-B iterations.
    tol : float
        Optimizer convergence tolerance.
    alpha_beta_max : float
        Upper bound on alpha + beta (stationarity constraint). Must be in (0, 1).
    min_var : float
        Floor for conditional variance (numerical safety).

    Returns
    -------
    dict
        - **params** — ``{mu, omega, alpha, beta}``
        - **conditional_variance** — ``list[float]``, per-observation h_t
        - **conditional_sigma** — ``list[float]``, sqrt(h_t)
        - **log_likelihood** — ``float``
        - **converged** — ``bool``
        - **n_iter** — ``int``
        - **message** — ``str``
    """
    return _garch11_fit(
        ys,
        max_iter=int(max_iter),
        tol=float(tol),
        alpha_beta_max=float(alpha_beta_max),
        min_var=float(min_var),
    )


def sv(
    ys: Sequence[float],
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
    log_eps: float = 1e-12,
) -> Mapping[str, Any]:
    """Fit an approximate stochastic volatility (SV) model.

    Uses the log(chi^2_1) Gaussian approximation: transforms returns via
    ``z_t = ln(y_t^2) - E[ln(chi^2_1)]`` and fits an AR(1) latent log-variance
    via Kalman MLE.

    Parameters
    ----------
    ys : array-like of float
        Return series (e.g. log-returns).
    max_iter : int
        Maximum L-BFGS-B iterations.
    tol : float
        Optimizer convergence tolerance.
    log_eps : float
        Small constant added before log to avoid log(0).

    Returns
    -------
    dict
        - **params** — ``{mu, phi, sigma}`` (log-variance mean, AR(1) persistence, vol-of-vol)
        - **smoothed_h** — ``list[float]``, smoothed log-variance h_t
        - **smoothed_sigma** — ``list[float]``, exp(h_t / 2)
        - **log_likelihood** — ``float``
        - **converged** — ``bool``
        - **n_iter** — ``int``
        - **message** — ``str``
    """
    return _sv_logchi2_fit(
        ys,
        max_iter=int(max_iter),
        tol=float(tol),
        log_eps=float(log_eps),
    )


__all__ = ["garch", "sv"]
