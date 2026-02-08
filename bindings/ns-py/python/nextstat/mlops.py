"""MLOps integration helpers (optional dependencies: wandb, mlflow).

Provides lightweight utilities to extract NextStat fit metrics as plain
Python dicts, ready to pipe into Weights & Biases, MLflow, Neptune, or
any logger that accepts ``dict[str, float]``.

No hard dependency on any logging framework — the user calls their
own ``wandb.log()`` / ``mlflow.log_metrics()`` with the dict we return.

Example::

    import nextstat
    from nextstat.mlops import metrics_dict

    result = nextstat.fit(model)
    wandb.log(metrics_dict(result))           # W&B
    mlflow.log_metrics(metrics_dict(result))   # MLflow
"""

from __future__ import annotations

import time
from typing import Any, Optional


def metrics_dict(
    fit_result,
    *,
    prefix: str = "",
    include_time: bool = True,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    """Extract key metrics from a NextStat ``FitResult`` as a flat dict.

    The returned dict is compatible with ``wandb.log()``,
    ``mlflow.log_metrics()``, ``neptune["metrics"].append()``, or any
    logger that accepts ``dict[str, float]``.

    Args:
        fit_result: ``nextstat.FitResult`` from ``nextstat.fit()`` or
            ``MaximumLikelihoodEstimator.fit()``.
        prefix: optional string prepended to every key (e.g. ``"ns/"``
            produces ``"ns/mu"``, ``"ns/nll"``, etc.).
        include_time: if ``True`` (default), includes ``time_ms`` from
            the fit result (if available).
        extra: optional dict of additional metrics to merge in.

    Returns:
        ``dict[str, float]`` with keys:

        - ``mu`` — best-fit signal strength (POI)
        - ``nll`` — negative log-likelihood at minimum
        - ``edm`` — estimated distance to minimum
        - ``n_calls`` — number of likelihood evaluations
        - ``converged`` — 1.0 if converged, 0.0 otherwise
        - ``time_ms`` — fit wall-clock time in milliseconds (if available)
        - per-parameter entries: ``param/<name>`` — best-fit value
        - per-parameter entries: ``error/<name>`` — Hesse error

    Example::

        result = nextstat.fit(model)
        d = metrics_dict(result, prefix="nextstat/")
        wandb.log(d)
        # {'nextstat/mu': 1.05, 'nextstat/nll': 42.3, ...}
    """
    d: dict[str, float] = {}

    # Core scalars (robust extraction — attributes may vary across versions)
    for attr in ("nll", "edm", "n_calls"):
        val = getattr(fit_result, attr, None)
        if val is not None:
            d[f"{prefix}{attr}"] = float(val)

    # Convergence flag as numeric
    converged = getattr(fit_result, "converged", None)
    if converged is not None:
        d[f"{prefix}converged"] = 1.0 if converged else 0.0

    # POI (mu) — try explicit poi first, then first parameter
    mu = getattr(fit_result, "mu", None)
    if mu is not None:
        d[f"{prefix}mu"] = float(mu)

    # Timing
    if include_time:
        time_ms = getattr(fit_result, "time_ms", None)
        if time_ms is not None:
            d[f"{prefix}time_ms"] = float(time_ms)

    # Per-parameter best-fit values and errors
    names = getattr(fit_result, "parameter_names", None)
    values = getattr(fit_result, "parameter_values", None)
    errors = getattr(fit_result, "parameter_errors", None)

    if names and values:
        for name, val in zip(names, values):
            d[f"{prefix}param/{name}"] = float(val)

    if names and errors:
        for name, err in zip(names, errors):
            d[f"{prefix}error/{name}"] = float(err)

    # Merge user-supplied extras
    if extra:
        for k, v in extra.items():
            d[f"{prefix}{k}"] = float(v)

    return d


def significance_metrics(
    z0: float,
    q0: float = 0.0,
    *,
    prefix: str = "",
    step_time_ms: float = 0.0,
) -> dict[str, float]:
    """Build a metrics dict for a single training step (significance loss).

    Use this in a training loop where you compute Z₀ per step and want
    to log it alongside the loss.

    Args:
        z0: discovery significance (Z₀ = sqrt(q₀)).
        q0: raw test statistic q₀ (optional, default 0).
        prefix: key prefix (e.g. ``"train/"``).
        step_time_ms: wall-clock time for this step.

    Returns:
        ``dict[str, float]``

    Example::

        from nextstat.mlops import significance_metrics

        loss = loss_fn(signal_hist)
        z0_val = -loss.item()  # if negate=True
        wandb.log(significance_metrics(z0_val, prefix="train/"))
    """
    d: dict[str, float] = {
        f"{prefix}z0": float(z0),
        f"{prefix}q0": float(q0),
    }
    if step_time_ms > 0:
        d[f"{prefix}step_time_ms"] = float(step_time_ms)
    return d


class StepTimer:
    """Lightweight wall-clock timer for training loop instrumentation.

    Example::

        timer = StepTimer()
        for batch in dataloader:
            timer.start()
            loss = loss_fn(signal_hist)
            loss.backward()
            optimizer.step()
            elapsed = timer.stop()
            wandb.log({"step_time_ms": elapsed})
    """

    def __init__(self):
        self._t0: float = 0.0

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        """Returns elapsed time in milliseconds since last ``start()``."""
        return (time.perf_counter() - self._t0) * 1000.0
