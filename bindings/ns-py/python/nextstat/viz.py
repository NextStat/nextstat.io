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


def _nsigma_index(nsigma_order: Sequence[int] | None, sigma: int) -> Optional[int]:
    if nsigma_order is None:
        return None
    try:
        return list(nsigma_order).index(int(sigma))
    except Exception:
        return None


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
    mu = artifact.get("mu_values")
    cls_obs = artifact.get("cls_obs")
    cls_exp = artifact.get("cls_exp")
    nsigma_order = artifact.get("nsigma_order")

    if not (isinstance(mu, list) and isinstance(cls_obs, list)):
        mu = [float(p["mu"]) for p in points]
        cls_obs = [float(p["cls"]) for p in points]

    mu = [float(x) for x in (mu or [])]
    cls_obs = [float(x) for x in (cls_obs or [])]

    exp_p2 = exp_p1 = exp_0 = exp_m1 = exp_m2 = None
    if isinstance(cls_exp, list) and mu:
        i2 = _nsigma_index(nsigma_order, 2)
        i1 = _nsigma_index(nsigma_order, 1)
        i0 = _nsigma_index(nsigma_order, 0)
        im1 = _nsigma_index(nsigma_order, -1)
        im2 = _nsigma_index(nsigma_order, -2)
        if None not in (i2, i1, i0, im1, im2):
            try:
                exp_p2 = [float(x) for x in cls_exp[i2]]  # type: ignore[index]
                exp_p1 = [float(x) for x in cls_exp[i1]]  # type: ignore[index]
                exp_0 = [float(x) for x in cls_exp[i0]]  # type: ignore[index]
                exp_m1 = [float(x) for x in cls_exp[im1]]  # type: ignore[index]
                exp_m2 = [float(x) for x in cls_exp[im2]]  # type: ignore[index]
            except Exception:
                exp_p2 = exp_p1 = exp_0 = exp_m1 = exp_m2 = None

    if exp_0 is None and points:
        exp = [p.get("expected") for p in points]
        if all(isinstance(e, (list, tuple)) and len(e) >= 5 for e in exp):
            # Legacy/default order is [2, 1, 0, -1, -2]
            exp_p2 = [float(e[0]) for e in exp]  # type: ignore[index]
            exp_p1 = [float(e[1]) for e in exp]  # type: ignore[index]
            exp_0 = [float(e[2]) for e in exp]  # type: ignore[index]
            exp_m1 = [float(e[3]) for e in exp]  # type: ignore[index]
            exp_m2 = [float(e[4]) for e in exp]  # type: ignore[index]

    if show_bands and mu and exp_m2 is not None and exp_p2 is not None:
        ax.fill_between(mu, exp_m2, exp_p2, color="#F2D95C", alpha=0.55, label="Expected ±2σ")
    if show_bands and mu and exp_m1 is not None and exp_p1 is not None:
        ax.fill_between(mu, exp_m1, exp_p1, color="#7BD389", alpha=0.65, label="Expected ±1σ")

    if show_expected and mu and exp_0 is not None:
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


def plot_brazil_limits(
    artifact: Mapping[str, Any],
    *,
    ax=None,
    title: Optional[str] = None,
    show_observed: bool = True,
    show_expected: bool = True,
    show_bands: bool = True,
):
    """Plot a standard 1D Brazil band for the upper limit from a `cls_curve` artifact.

    Uses `obs_limit`, `exp_limits`, and `nsigma_order` (if present).
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 1.8))

    nsigma_order = artifact.get("nsigma_order")
    exp_limits = artifact.get("exp_limits")
    obs_limit = artifact.get("obs_limit")

    y0 = 0.0
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)

    if isinstance(exp_limits, (list, tuple)) and len(exp_limits) >= 5:
        i2 = _nsigma_index(nsigma_order, 2) or 0
        i1 = _nsigma_index(nsigma_order, 1) or 1
        i0 = _nsigma_index(nsigma_order, 0) or 2
        im1 = _nsigma_index(nsigma_order, -1) or 3
        im2 = _nsigma_index(nsigma_order, -2) or 4
        try:
            p2 = float(exp_limits[i2])
            p1 = float(exp_limits[i1])
            med = float(exp_limits[i0])
            m1 = float(exp_limits[im1])
            m2 = float(exp_limits[im2])
        except Exception:
            p2 = p1 = med = m1 = m2 = None  # type: ignore[assignment]

        if show_bands and None not in (m2, p2):
            ax.fill_betweenx([y0 - 0.2, y0 + 0.2], m2, p2, color="#F2D95C", alpha=0.55, label="Expected ±2σ")
        if show_bands and None not in (m1, p1):
            ax.fill_betweenx([y0 - 0.2, y0 + 0.2], m1, p1, color="#7BD389", alpha=0.65, label="Expected ±1σ")
        if show_expected and med is not None:
            ax.axvline(med, color="#1D4ED8", lw=2.0, ls="--", label="Expected (median)")

    if show_observed:
        try:
            ax.axvline(float(obs_limit), color="#111827", lw=2.0, label="Observed")
        except Exception:
            pass

    ax.set_xlabel("mu_up")
    ax.grid(True, axis="x", alpha=0.25)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    return ax


def plot_profile_curve(
    artifact: Mapping[str, Any],
    *,
    ax=None,
    y: str = "q_mu",
    title: Optional[str] = None,
):
    """Plot a profile scan series from a `profile_curve` artifact dict.

    Parameters
    - y: `"q_mu"` or `"twice_delta_nll"`
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.2))

    points = artifact.get("points") or []
    mu = artifact.get("mu_values")
    if not isinstance(mu, list):
        mu = [float(p["mu"]) for p in points]
    mu = [float(x) for x in (mu or [])]

    y = str(y)
    if y == "q_mu":
        series = artifact.get("q_mu_values")
        if not isinstance(series, list):
            series = [float(p["q_mu"]) for p in points]
        label = "q_mu"
    elif y == "twice_delta_nll":
        series = artifact.get("twice_delta_nll")
        if not isinstance(series, list):
            nll_hat = artifact.get("nll_hat")
            if nll_hat is not None and points and all("nll_mu" in p for p in points):
                series = [2.0 * (float(p["nll_mu"]) - float(nll_hat)) for p in points]
            else:
                raise ValueError("artifact is missing twice_delta_nll (and cannot infer from points)")
        label = "2ΔNLL"
    else:
        raise ValueError("y must be one of: 'q_mu', 'twice_delta_nll'")

    series = [float(x) for x in (series or [])]

    ax.plot(mu, series, color="#111827", lw=2.0)

    mu_hat = artifact.get("mu_hat")
    if mu_hat is not None:
        try:
            ax.axvline(float(mu_hat), color="#6B7280", lw=1.25, ls=":")
        except Exception:
            pass

    ax.set_xlabel("mu")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    return ax


__all__ = [
    "cls_curve",
    "profile_curve",
    "plot_cls_curve",
    "plot_brazil_limits",
    "plot_profile_curve",
]
