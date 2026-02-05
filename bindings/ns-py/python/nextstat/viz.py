"""Visualization helpers (plots).

This module focuses on producing figures from plot-friendly JSON-like artifacts.

Requires matplotlib (install via `pip install nextstat[viz]` once the extra exists,
or just `pip install matplotlib`).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


def cls_curve(model, scan: Sequence[float], *, alpha: float = 0.05, data: Optional[list[float]] = None):
    """Compute CLs curve + expected bands artifact over a scan grid."""
    from . import _core

    return _core.cls_curve(model, list(scan), alpha=alpha, data=data)


def profile_curve(model, mu_values: Sequence[float], *, data: Optional[list[float]] = None):
    """Compute profile likelihood scan artifact over POI values."""
    from . import _core

    return _core.profile_curve(model, list(mu_values), data=data)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: matplotlib. Install via `pip install matplotlib`.") from e


def plot_cls_curve(
    artifact: Mapping[str, Any],
    *,
    ax=None,
    show_expected: bool = True,
    show_bands: bool = True,
    show_observed: bool = True,
    show_limits: bool = True,
    title: Optional[str] = None,
):
    """Plot CLs(mu) with Brazil bands from a `cls_curve` artifact dict."""
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.2))

    points = artifact.get("points") or []
    mu = [float(p["mu"]) for p in points]
    cls_obs = [float(p["cls"]) for p in points]
    exp = [p["expected"] for p in points]

    # expected order is [2, 1, 0, -1, -2]
    exp_p2 = [float(e[0]) for e in exp]
    exp_p1 = [float(e[1]) for e in exp]
    exp_0 = [float(e[2]) for e in exp]
    exp_m1 = [float(e[3]) for e in exp]
    exp_m2 = [float(e[4]) for e in exp]

    if show_bands and mu:
        ax.fill_between(mu, exp_m2, exp_p2, color="#F2D95C", alpha=0.55, label="Expected ±2σ")
        ax.fill_between(mu, exp_m1, exp_p1, color="#7BD389", alpha=0.65, label="Expected ±1σ")

    if show_expected and mu:
        ax.plot(mu, exp_0, color="#1D4ED8", lw=2.0, ls="--", label="Expected (median)")

    if show_observed and mu:
        ax.plot(mu, cls_obs, color="#111827", lw=2.0, label="Observed")

    alpha = float(artifact.get("alpha", 0.05))
    ax.axhline(alpha, color="#6B7280", lw=1.25, ls=":", label=f"alpha={alpha:g}")

    if show_limits:
        try:
            obs_lim = float(artifact.get("obs_limit"))
            ax.axvline(obs_lim, color="#111827", lw=1.25, ls=":", label="Obs. limit")
        except Exception:
            pass
        try:
            exp_lims = artifact.get("exp_limits")
            if exp_lims is not None:
                exp_med = float(exp_lims[2])
                ax.axvline(exp_med, color="#1D4ED8", lw=1.25, ls=":", label="Exp. limit (median)")
        except Exception:
            pass

    ax.set_xlabel("mu")
    ax.set_ylabel("CLs")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    return ax


def plot_profile_curve(
    artifact: Mapping[str, Any],
    *,
    ax=None,
    title: Optional[str] = None,
):
    """Plot q_mu(mu) from a `profile_curve` artifact dict."""
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.2))

    points = artifact.get("points") or []
    mu = [float(p["mu"]) for p in points]
    q_mu = [float(p["q_mu"]) for p in points]

    ax.plot(mu, q_mu, color="#111827", lw=2.0)

    mu_hat = artifact.get("mu_hat")
    if mu_hat is not None:
        try:
            ax.axvline(float(mu_hat), color="#6B7280", lw=1.25, ls=":")
        except Exception:
            pass

    ax.set_xlabel("mu")
    ax.set_ylabel("q_mu")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    return ax


__all__ = [
    "cls_curve",
    "profile_curve",
    "plot_cls_curve",
    "plot_profile_curve",
]

