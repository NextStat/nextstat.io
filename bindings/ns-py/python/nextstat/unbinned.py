"""Event-level (unbinned) likelihood helpers.

This module exposes both:

1) low-level bindings (mirrors ``nextstat._core``), and
2) a higher-level workflow object ``UnbinnedAnalysis`` for production scripts.

The low-level surface (via unified API):
- ``UnbinnedModel.from_config(path)`` — compile a JSON/YAML ``unbinned_spec_v0``
- ``fit_toys(model, params, ...)`` — generate+fit unbinned Poisson toys
- ``profile_scan(model, mu_values)`` — profile likelihood scan (q_mu)
- ``hypotest(poi_test, model)`` — one-sided q_mu (and q0 if applicable)
- ``hypotest_toys(poi_test, model, ...)`` — toy-based CLs (qtilde)
- ``ranking(model)`` — nuisance ranking (impact on POI)

The high-level surface:
- ``UnbinnedAnalysis.from_config(path)`` — compile + attach analysis helpers
- ``analysis.fit()/fit_toys()/scan()/hypotest()/hypotest_toys()/ranking()``
- ``analysis.with_fixed_param(param, value)`` with name or index selectors
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ._core import (  # type: ignore
    fit as _fit,
    fit_toys as _fit_toys,
    hypotest as _hypotest,
    hypotest_toys as _hypotest_toys,
    profile_scan as _profile_scan,
    ranking as _ranking,
    UnbinnedModel,
)


class UnbinnedAnalysis:
    """High-level workflow wrapper for event-level likelihood analyses.

    Typical usage:

    >>> import nextstat
    >>> analysis = nextstat.unbinned.UnbinnedAnalysis.from_config("model.yaml")
    >>> fit = analysis.fit()
    >>> scan = analysis.scan([0.0, 1.0, 2.0])
    >>> ht = analysis.hypotest(1.0)
    """

    def __init__(self, model: UnbinnedModel):
        self.model = model

    @classmethod
    def from_config(cls, path: str | Path) -> UnbinnedAnalysis:
        """Compile an ``UnbinnedModel`` from an ``unbinned_spec_v0`` file."""
        return cls(UnbinnedModel.from_config(str(path)))

    def fit(self, *, init_pars: Iterable[float] | None = None):
        """Run MLE fit and return ``nextstat.FitResult``."""
        init = list(init_pars) if init_pars is not None else None
        return _fit(self.model, init_pars=init)

    def fit_toys(
        self,
        params: Iterable[float] | None = None,
        *,
        n_toys: int = 1000,
        seed: int = 42,
    ):
        """Generate and fit Poisson-fluctuated unbinned toys."""
        p = list(params) if params is not None else list(self.model.suggested_init())
        return _fit_toys(
            self.model,
            p,
            n_toys=int(n_toys),
            seed=int(seed),
        )

    def scan(self, mu_values: Iterable[float]) -> dict[str, Any]:
        """Profile-likelihood scan over POI values."""
        return _profile_scan(self.model, list(mu_values))

    def hypotest(self, mu_test: float) -> dict[str, Any]:
        """Compute one-sided unbinned ``q_mu`` (plus ``q0`` when available)."""
        return _hypotest(float(mu_test), self.model)

    def hypotest_toys(
        self,
        poi_test: float,
        *,
        n_toys: int = 1000,
        seed: int = 42,
        expected_set: bool = False,
        return_tail_probs: bool = False,
        return_meta: bool = False,
    ):
        """Toy-based unbinned CLs (qtilde)."""
        return _hypotest_toys(
            float(poi_test),
            self.model,
            n_toys=int(n_toys),
            seed=int(seed),
            expected_set=bool(expected_set),
            return_tail_probs=bool(return_tail_probs),
            return_meta=bool(return_meta),
        )

    def ranking(self) -> list[dict[str, Any]]:
        """Nuisance ranking (impact on POI)."""
        return _ranking(self.model)

    def parameter_index(self, param: int | str) -> int:
        """Resolve parameter by index or name."""
        if isinstance(param, int):
            n = len(self.model.parameter_names())
            if param < 0 or param >= n:
                raise IndexError(f"parameter index out of range: {param} (n={n})")
            return int(param)
        names = self.model.parameter_names()
        try:
            return names.index(str(param))
        except ValueError as exc:
            raise KeyError(f"unknown parameter name: {param!r}") from exc

    def with_fixed_param(self, param: int | str, value: float) -> UnbinnedAnalysis:
        """Return a new analysis with one parameter fixed to a value."""
        idx = self.parameter_index(param)
        return UnbinnedAnalysis(self.model.with_fixed_param(idx, float(value)))

    def summary(self) -> dict[str, Any]:
        """Return model metadata summary for logging/debugging."""
        names = self.model.parameter_names()
        init = self.model.suggested_init()
        bounds = self.model.suggested_bounds()
        poi_idx = self.model.poi_index()
        params: list[dict[str, Any]] = []
        for i, name in enumerate(names):
            params.append(
                {
                    "index": i,
                    "name": name,
                    "init": init[i],
                    "bounds": list(bounds[i]),
                    "is_poi": poi_idx == i,
                }
            )
        return {
            "schema_version": self.model.schema_version(),
            "n_params": self.model.n_params(),
            "poi_index": poi_idx,
            "parameters": params,
        }


def from_config(path: str | Path) -> UnbinnedAnalysis:
    """Compile config and return a high-level ``UnbinnedAnalysis`` wrapper."""
    return UnbinnedAnalysis.from_config(path)


__all__ = [
    "from_config",
    "UnbinnedAnalysis",
    "UnbinnedModel",
]
