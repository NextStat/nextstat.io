"""Bayesian sampling helpers.

This module provides convenience wrappers around `nextstat._core.sample`:
- optionally convert the raw output dict into an ArviZ `InferenceData`
- keep a stable, user-facing surface in Python (optional deps live here)
"""

from __future__ import annotations

from pathlib import Path
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


def save(idata, path: str | Path, *, format: str = "json") -> Path:
    """Save an ArviZ `InferenceData` to disk.

    This is a thin convenience wrapper so users can go:
    - `nextstat.bayes.sample(..., out="trace.json")`
    - `az.from_json("trace.json")`

    Supported formats:
    - `json`: uses `arviz.to_json` when available
    - `netcdf`: uses `InferenceData.to_netcdf`
    """

    try:
        import arviz as az
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: arviz. Install via `pip install nextstat[bayes]`.") from e

    path = Path(path)
    fmt = format.lower().strip()
    if fmt in ("json",):
        if hasattr(az, "to_json"):
            az.to_json(idata, path)  # type: ignore[attr-defined]
        else:  # pragma: no cover
            raise RuntimeError("Your ArviZ version does not provide `arviz.to_json(...)`.")
        return path

    if fmt in ("netcdf", "nc", "cdf"):
        if hasattr(idata, "to_netcdf"):
            idata.to_netcdf(path)  # type: ignore[attr-defined]
        else:  # pragma: no cover
            raise RuntimeError("InferenceData object does not provide `to_netcdf(...)`.")
        return path

    raise ValueError(f"Unknown format: {format!r}. Expected 'json' or 'netcdf'.")


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
    init_jitter_rel: float | None = None,
    init_overdispersed_rel: float | None = None,
    data: Optional[list[float]] = None,
    return_idata: bool = True,
    out: str | Path | None = None,
    out_format: str = "json",
):
    """Run NUTS sampling.

    Notes:
    - Multi-chain seeding is deterministic: chain `i` uses `seed + i`.
    - Initialization options are mutually exclusive (set at most one):
      `init_jitter`, `init_jitter_rel`, `init_overdispersed_rel`.

    Parameters:
    - `init_jitter`: add Normal jitter in unconstrained space (`z`).
    - `init_jitter_rel`: jitter scale relative to parameter bounds / magnitude
      (mapped to `z` via a local Jacobian).
    - `init_overdispersed_rel`: like `init_jitter_rel` but allows larger
      excursions; useful for overdispersed initializations across chains.
    - `data`: only supported for `HistFactoryModel` (overrides observed data).

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
        init_jitter_rel=init_jitter_rel,
        init_overdispersed_rel=init_overdispersed_rel,
        data=data,
    )

    if not return_idata:
        if out is not None:
            import json

            Path(out).write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
        return raw

    idata = to_inferencedata(raw)
    if out is not None:
        save(idata, out, format=out_format)
    return idata


__all__ = [
    "sample",
    "save",
    "to_inferencedata",
]
