"""Bayesian sampling helpers.

This module provides convenience wrappers around `nextstat._core.sample`:
- optionally convert the raw output dict into an ArviZ `InferenceData`
- keep a stable, user-facing surface in Python (optional deps live here)
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


def to_inferencedata(raw: Mapping[str, Any]):
    """Convert `_core.sample(...)` output into an ArviZ `InferenceData`.

    Requires `arviz` and `numpy` (install via `pip install nextstat[bayes]`).
    """

    try:
        import numpy as np
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: numpy. Install via `pip install nextstat[bayes]`.") from e

    try:
        import arviz as az
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: arviz. Install via `pip install nextstat[bayes]`.") from e

    posterior = raw.get("posterior")
    sample_stats = raw.get("sample_stats")
    if not isinstance(posterior, Mapping) or not isinstance(sample_stats, Mapping):
        raise ValueError("raw must be the dict returned by `nextstat._core.sample(...)`.")

    n_chains = int(raw.get("n_chains", 0) or 0)
    n_samples = int(raw.get("n_samples", 0) or 0)

    # Broadcast per-chain scalar stats to (chain, draw) so ArviZ can ingest them.
    ss: Dict[str, Any] = dict(sample_stats)
    if "step_size" in ss and isinstance(ss["step_size"], (list, tuple)) and n_chains and n_samples:
        step = np.asarray(ss["step_size"], dtype=float).reshape((n_chains, 1))
        ss["step_size"] = np.repeat(step, n_samples, axis=1)

    idata = az.from_dict(posterior=dict(posterior), sample_stats=ss)

    # Keep NextStat-specific diagnostics accessible (not a native ArviZ schema).
    diag = raw.get("diagnostics")
    if isinstance(diag, Mapping):
        try:
            idata.attrs["nextstat_diagnostics"] = dict(diag)
        except Exception:
            # attrs may reject non-serializable objects; keep best-effort.
            pass

    return idata


def sample(
    model,
    *,
    n_chains: int = 4,
    n_warmup: int = 500,
    n_samples: int = 1000,
    seed: int = 42,
    max_treedepth: int = 10,
    target_accept: float = 0.8,
    init_jitter: float = 0.0,
    data: Optional[list[float]] = None,
    return_idata: bool = True,
):
    """Run NUTS sampling.

    By default returns ArviZ `InferenceData` if dependencies are available.
    If `return_idata=False`, returns the raw dict from `_core.sample(...)`.
    """

    from . import _core  # local import to keep import-time light

    raw = _core.sample(
        model,
        n_chains=n_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        seed=seed,
        max_treedepth=max_treedepth,
        target_accept=target_accept,
        init_jitter=init_jitter,
        data=data,
    )

    if not return_idata:
        return raw

    return to_inferencedata(raw)


__all__ = [
    "sample",
    "to_inferencedata",
]
