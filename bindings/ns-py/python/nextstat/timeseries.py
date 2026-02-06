"""Time series helpers (Phase 8).

Currently provides a baseline linear-Gaussian Kalman filter + RTS smoother.

The heavy lifting lives in Rust (`ns-inference`). This Python layer keeps a
stable, user-facing surface.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def kalman_filter(model, ys: Sequence[Sequence[float | None]]) -> Mapping[str, Any]:
    """Run Kalman filtering.

    Parameters
    - model: `nextstat._core.KalmanModel`
    - ys: list of observation vectors (shape: T x n_obs)

    Returns
    - dict with keys:
      - log_likelihood
      - predicted_means, predicted_covs
      - filtered_means, filtered_covs
    """
    from . import _core

    return _core.kalman_filter(model, [list(y) for y in ys])


def kalman_smooth(model, ys: Sequence[Sequence[float | None]]) -> Mapping[str, Any]:
    """Run Kalman filtering + RTS smoothing.

    Returns
    - dict with keys:
      - log_likelihood
      - filtered_means, filtered_covs
      - smoothed_means, smoothed_covs
    """
    from . import _core

    return _core.kalman_smooth(model, [list(y) for y in ys])


def kalman_em(
    model,
    ys: Sequence[Sequence[float | None]],
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    estimate_q: bool = True,
    estimate_r: bool = True,
    estimate_f: bool = False,
    estimate_h: bool = False,
    min_diag: float = 1e-12,
) -> Mapping[str, Any]:
    """Fit Q/R with EM while keeping F/H/m0/P0 fixed."""
    from . import _core

    return _core.kalman_em(
        model,
        [list(y) for y in ys],
        max_iter=max_iter,
        tol=tol,
        estimate_q=estimate_q,
        estimate_r=estimate_r,
        estimate_f=estimate_f,
        estimate_h=estimate_h,
        min_diag=min_diag,
    )

def kalman_fit(
    model,
    ys: Sequence[Sequence[float | None]],
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    estimate_q: bool = True,
    estimate_r: bool = True,
    estimate_f: bool = False,
    estimate_h: bool = False,
    min_diag: float = 1e-12,
    forecast_steps: int = 0,
    no_smooth: bool = False,
) -> Mapping[str, Any]:
    """Fit with EM, then run RTS smoother (and optional forecast).

    Returns
    - dict with keys:
      - model: fitted `nextstat._core.KalmanModel`
      - em: EM metadata + fitted matrices (nested lists)
      - smooth: smoother output dict, or None if `no_smooth=True`
      - forecast: forecast output dict, or None if `forecast_steps=0`
    """
    em_out = kalman_em(
        model,
        ys,
        max_iter=max_iter,
        tol=tol,
        estimate_q=estimate_q,
        estimate_r=estimate_r,
        estimate_f=estimate_f,
        estimate_h=estimate_h,
        min_diag=min_diag,
    )

    fitted_model = em_out["model"]
    em_meta = {
        "converged": em_out["converged"],
        "n_iter": em_out["n_iter"],
        "loglik_trace": em_out["loglik_trace"],
        "f": em_out["f"],
        "h": em_out["h"],
        "q": em_out["q"],
        "r": em_out["r"],
    }

    smooth_out = None if no_smooth else kalman_smooth(fitted_model, ys)
    forecast_out = None
    if int(forecast_steps) != 0:
        if int(forecast_steps) < 0:
            raise ValueError("forecast_steps must be >= 0")
        forecast_out = kalman_forecast(fitted_model, ys, steps=int(forecast_steps))

    return {
        "model": fitted_model,
        "em": em_meta,
        "smooth": smooth_out,
        "forecast": forecast_out,
    }

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError("Missing dependency: matplotlib. Install via `pip install matplotlib`.") from e

def _z_for_level(level: float) -> float:
    from statistics import NormalDist

    if not (0.0 < float(level) < 1.0):
        raise ValueError("level must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + 0.5 * float(level))
    if not (z == z and z > 0.0):  # NaN guard
        raise ValueError("invalid level (z-score was not finite)")
    return float(z)

def _bands_from_means_covs(means: list[list[float]], covs: list[list[list[float]]], z: float):
    if len(means) != len(covs):
        raise ValueError("means/covs length mismatch")
    if not means:
        raise ValueError("means must be non-empty")

    t_max = len(means)
    dim = len(means[0])

    lo: list[list[float]] = [[0.0] * dim for _ in range(t_max)]
    hi: list[list[float]] = [[0.0] * dim for _ in range(t_max)]

    for t in range(t_max):
        if len(means[t]) != dim:
            raise ValueError("means must be rectangular")
        if len(covs[t]) != dim or any(len(row) != dim for row in covs[t]):
            raise ValueError("covs must be T x dim x dim")
        for j in range(dim):
            var = float(covs[t][j][j])
            if not (var == var) or var < 0.0:
                var = 0.0
            s = (var ** 0.5) * float(z)
            mu = float(means[t][j])
            lo[t][j] = mu - s
            hi[t][j] = mu + s

    return lo, hi

def _as_float_matrix(name: str, m: Any) -> list[list[float]]:
    if not isinstance(m, (list, tuple)) or not m:
        raise ValueError(f"{name} must be a non-empty nested list")
    out: list[list[float]] = []
    width = None
    for row in m:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError(f"{name} must be a non-empty nested list")
        r = [float(x) for x in row]
        if width is None:
            width = len(r)
        elif len(r) != width:
            raise ValueError(f"{name} must be rectangular")
        out.append(r)
    return out

def _as_float_diag(name: str, m: Any) -> list[float]:
    mm = _as_float_matrix(name, m)
    n = min(len(mm), len(mm[0]))
    return [float(mm[i][i]) for i in range(n)]

def _obs_bands_from_state(
    *,
    means: list[list[float]],
    covs: list[list[list[float]]],
    h: list[list[float]] | None,
    r: list[list[float]] | None,
    z: float,
):
    if h is None or r is None:
        return None, None, None

    h = _as_float_matrix("H", h)
    r_diag = _as_float_diag("R", r)

    n_obs = len(h)
    n_state = len(h[0])
    if any(len(row) != n_state for row in h):
        raise ValueError("H must be rectangular")
    if len(r_diag) < n_obs:
        raise ValueError("R has wrong shape (expected at least n_obs diagonal)")

    if not means:
        raise ValueError("means must be non-empty")
    if len(means) != len(covs):
        raise ValueError("means/covs length mismatch")
    if any(len(m) != n_state for m in means):
        raise ValueError("state mean dim mismatch vs H")
    if any(len(p) != n_state or any(len(row) != n_state for row in p) for p in covs):
        raise ValueError("state cov dim mismatch vs H")

    t_max = len(means)
    obs_mean: list[list[float]] = [[0.0] * n_obs for _ in range(t_max)]
    obs_lo: list[list[float]] = [[0.0] * n_obs for _ in range(t_max)]
    obs_hi: list[list[float]] = [[0.0] * n_obs for _ in range(t_max)]

    for t in range(t_max):
        m = means[t]
        p = covs[t]
        for i in range(n_obs):
            # mean: h_i * m
            mu = 0.0
            for a in range(n_state):
                mu += h[i][a] * m[a]
            obs_mean[t][i] = float(mu)

            # var: h_i * P * h_i^T + R_ii
            var = 0.0
            for a in range(n_state):
                ha = h[i][a]
                if ha == 0.0:
                    continue
                for b in range(n_state):
                    var += ha * p[a][b] * h[i][b]
            var += float(r_diag[i])
            if not (var == var) or var < 0.0:
                var = 0.0
            s = (var ** 0.5) * float(z)
            obs_lo[t][i] = float(mu - s)
            obs_hi[t][i] = float(mu + s)

    return obs_mean, obs_lo, obs_hi

def kalman_viz_artifact(
    fit_out: Mapping[str, Any],
    ys: Sequence[Sequence[float | None]],
    *,
    level: float = 0.95,
) -> Mapping[str, Any]:
    """Build a plot-friendly artifact from `kalman_fit(...)` output."""
    z = _z_for_level(float(level))
    ys_out = [[None if v is None else float(v) for v in row] for row in ys]
    t_max = len(ys_out)
    if t_max == 0:
        raise ValueError("ys must be non-empty")

    em = fit_out.get("em") if isinstance(fit_out, Mapping) else None
    h = None
    r = None
    if isinstance(em, Mapping):
        h = em.get("h")
        r = em.get("r")

    smooth_in = fit_out.get("smooth") if isinstance(fit_out, Mapping) else None
    smooth_art = None
    if isinstance(smooth_in, Mapping):
        sm = smooth_in.get("smoothed_means")
        sp = smooth_in.get("smoothed_covs")
        if not isinstance(sm, list) or not isinstance(sp, list):
            raise ValueError("fit_out.smooth missing smoothed_means/smoothed_covs")

        state_mean = [[float(x) for x in row] for row in sm]
        state_covs = [[[float(x) for x in row] for row in mat] for mat in sp]
        state_lo, state_hi = _bands_from_means_covs(state_mean, state_covs, z)
        obs_mean, obs_lo, obs_hi = _obs_bands_from_state(
            means=state_mean,
            covs=state_covs,
            h=h if isinstance(h, (list, tuple)) else None,
            r=r if isinstance(r, (list, tuple)) else None,
            z=z,
        )
        smooth_art = {
            "state_mean": state_mean,
            "state_lo": state_lo,
            "state_hi": state_hi,
            "obs_mean": obs_mean,
            "obs_lo": obs_lo,
            "obs_hi": obs_hi,
        }

    forecast_in = fit_out.get("forecast") if isinstance(fit_out, Mapping) else None
    forecast_art = None
    if isinstance(forecast_in, Mapping):
        fm = forecast_in.get("state_means")
        fp = forecast_in.get("state_covs")
        yom = forecast_in.get("obs_means")
        yop = forecast_in.get("obs_covs")
        if not (isinstance(fm, list) and isinstance(fp, list) and isinstance(yom, list) and isinstance(yop, list)):
            raise ValueError("fit_out.forecast missing state/obs means/covs")

        fc_state_mean = [[float(x) for x in row] for row in fm]
        fc_state_covs = [[[float(x) for x in row] for row in mat] for mat in fp]
        fc_state_lo, fc_state_hi = _bands_from_means_covs(fc_state_mean, fc_state_covs, z)

        fc_obs_mean = [[float(x) for x in row] for row in yom]
        fc_obs_covs = [[[float(x) for x in row] for row in mat] for mat in yop]
        fc_obs_lo, fc_obs_hi = _bands_from_means_covs(fc_obs_mean, fc_obs_covs, z)

        k = len(fc_state_mean)
        forecast_art = {
            "t": list(range(t_max, t_max + k)),
            "state_mean": fc_state_mean,
            "state_lo": fc_state_lo,
            "state_hi": fc_state_hi,
            "obs_mean": fc_obs_mean,
            "obs_lo": fc_obs_lo,
            "obs_hi": fc_obs_hi,
        }

    out = {
        "level": float(level),
        "t_obs": list(range(t_max)),
        "ys": ys_out,
        "smooth": smooth_art,
        "forecast": forecast_art,
    }
    if isinstance(smooth_in, Mapping) and "log_likelihood" in smooth_in:
        out["log_likelihood"] = float(smooth_in["log_likelihood"])
    return out

def plot_kalman_obs(
    artifact: Mapping[str, Any],
    *,
    obs_index: int = 0,
    ax=None,
    title: str | None = None,
    show_bands: bool = True,
    show_forecast: bool = True,
):
    """Plot observed series with smoothed mean + bands (+ optional forecast)."""
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4.2))

    ys = artifact.get("ys") or []
    t_obs = artifact.get("t_obs") or list(range(len(ys)))
    y_obs = []
    for row in ys:
        v = None
        try:
            v = row[int(obs_index)]
        except Exception:
            v = None
        y_obs.append(v if v is None else float(v))

    t_obs_f = [float(t) for t in t_obs]
    ax.plot(t_obs_f, [float("nan") if v is None else float(v) for v in y_obs], color="#111827", lw=1.25, label="Observed")

    smooth = artifact.get("smooth") or None
    if isinstance(smooth, Mapping) and smooth.get("obs_mean") is not None:
        mu = [float(row[int(obs_index)]) for row in smooth["obs_mean"]]
        ax.plot(t_obs_f, mu, color="#1D4ED8", lw=2.0, label="Smoothed")
        if show_bands and smooth.get("obs_lo") is not None and smooth.get("obs_hi") is not None:
            lo = [float(row[int(obs_index)]) for row in smooth["obs_lo"]]
            hi = [float(row[int(obs_index)]) for row in smooth["obs_hi"]]
            ax.fill_between(t_obs_f, lo, hi, color="#93C5FD", alpha=0.35, label="Smoothed band")

    fc = artifact.get("forecast") or None
    if show_forecast and isinstance(fc, Mapping):
        t_fc = [float(t) for t in fc["t"]]
        mu = [float(row[int(obs_index)]) for row in fc["obs_mean"]]
        ax.plot(t_fc, mu, color="#059669", lw=2.0, ls="--", label="Forecast")
        if show_bands:
            lo = [float(row[int(obs_index)]) for row in fc["obs_lo"]]
            hi = [float(row[int(obs_index)]) for row in fc["obs_hi"]]
            ax.fill_between(t_fc, lo, hi, color="#6EE7B7", alpha=0.25, label="Forecast band")

    ax.set_xlabel("t")
    ax.set_ylabel(f"y[{int(obs_index)}]")
    ax.grid(True, alpha=0.25)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    return ax

def plot_kalman_states(
    artifact: Mapping[str, Any],
    *,
    state_indices: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    ax=None,
    title: str | None = None,
    show_bands: bool = True,
    show_forecast: bool = True,
):
    """Plot smoothed latent states x[t] with uncertainty bands (+ optional forecast)."""
    _require_matplotlib()
    import matplotlib.pyplot as plt

    smooth = artifact.get("smooth") or None
    if not isinstance(smooth, Mapping):
        raise ValueError("artifact.smooth is missing; build artifact from kalman_fit with smoothing enabled")
    state_mean = smooth.get("state_mean")
    state_lo = smooth.get("state_lo")
    state_hi = smooth.get("state_hi")
    if not (isinstance(state_mean, list) and isinstance(state_lo, list) and isinstance(state_hi, list)):
        raise ValueError("artifact.smooth missing state_mean/state_lo/state_hi")

    t_obs = artifact.get("t_obs") or list(range(len(state_mean)))
    if len(t_obs) != len(state_mean):
        raise ValueError("artifact.t_obs length mismatch")
    t_obs_f = [float(t) for t in t_obs]

    n_state = len(state_mean[0]) if state_mean else 0
    if n_state <= 0:
        raise ValueError("state_mean must be non-empty")

    if state_indices is None:
        idxs = list(range(n_state))
    else:
        idxs = [int(i) for i in state_indices]
        for i in idxs:
            if i < 0 or i >= n_state:
                raise ValueError(f"state index out of range: {i} (n_state={n_state})")

    if labels is not None and len(labels) != len(idxs):
        raise ValueError("labels length must match state_indices")

    n = len(idxs)
    if ax is None:
        _, axs = plt.subplots(n, 1, figsize=(7.2, 2.2 * n), sharex=True)
        if n == 1:
            axs = [axs]
    else:
        axs = ax
        if not isinstance(axs, (list, tuple)):
            axs = [axs]
        if len(axs) != n:
            raise ValueError("ax must be None or a list of axes matching the number of states")

    fc = artifact.get("forecast") or None
    fc_t = None
    if show_forecast and isinstance(fc, Mapping):
        fc_t = [float(t) for t in fc.get("t") or []]

    for k, j in enumerate(idxs):
        a = axs[k]
        lab = labels[k] if labels is not None else f"x[{j}]"

        mu = [float(row[j]) for row in state_mean]
        a.plot(t_obs_f, mu, color="#1D4ED8", lw=2.0, label="Smoothed")
        if show_bands:
            lo = [float(row[j]) for row in state_lo]
            hi = [float(row[j]) for row in state_hi]
            a.fill_between(t_obs_f, lo, hi, color="#93C5FD", alpha=0.35, label="Smoothed band")

        if fc_t is not None and isinstance(fc, Mapping) and fc.get("state_mean") is not None:
            fmu = [float(row[j]) for row in fc["state_mean"]]
            a.plot(fc_t, fmu, color="#059669", lw=2.0, ls="--", label="Forecast")
            if show_bands and fc.get("state_lo") is not None and fc.get("state_hi") is not None:
                flo = [float(row[j]) for row in fc["state_lo"]]
                fhi = [float(row[j]) for row in fc["state_hi"]]
                a.fill_between(fc_t, flo, fhi, color="#6EE7B7", alpha=0.25, label="Forecast band")

        a.set_ylabel(lab)
        a.grid(True, alpha=0.25)

    axs[-1].set_xlabel("t")
    if title:
        axs[0].set_title(title)

    # Legend only once (top axis) to reduce clutter.
    axs[0].legend(frameon=False, ncol=2)
    return axs

def plot_kalman_obs_grid(
    artifact: Mapping[str, Any],
    *,
    obs_indices: Sequence[int] | None = None,
    labels: Sequence[str] | None = None,
    ax=None,
    title: str | None = None,
    show_bands: bool = True,
    show_forecast: bool = True,
):
    """Plot observation components y[t] (one subplot per component)."""
    _require_matplotlib()
    import matplotlib.pyplot as plt

    ys = artifact.get("ys") or []
    if not isinstance(ys, list) or not ys:
        raise ValueError("artifact.ys must be a non-empty list")
    if not isinstance(ys[0], list) or not ys[0]:
        raise ValueError("artifact.ys rows must be non-empty lists")

    n_obs = len(ys[0])
    if obs_indices is None:
        idxs = list(range(n_obs))
    else:
        idxs = [int(i) for i in obs_indices]
        for i in idxs:
            if i < 0 or i >= n_obs:
                raise ValueError(f"obs index out of range: {i} (n_obs={n_obs})")

    if labels is not None and len(labels) != len(idxs):
        raise ValueError("labels length must match obs_indices")

    n = len(idxs)
    if ax is None:
        _, axs = plt.subplots(n, 1, figsize=(7.2, 2.2 * n), sharex=True)
        if n == 1:
            axs = [axs]
    else:
        axs = ax
        if not isinstance(axs, (list, tuple)):
            axs = [axs]
        if len(axs) != n:
            raise ValueError("ax must be None or a list of axes matching the number of obs components")

    t_obs = artifact.get("t_obs") or list(range(len(ys)))
    if len(t_obs) != len(ys):
        raise ValueError("artifact.t_obs length mismatch")
    t_obs_f = [float(t) for t in t_obs]

    smooth = artifact.get("smooth") or None
    fc = artifact.get("forecast") or None
    fc_t = None
    if show_forecast and isinstance(fc, Mapping):
        fc_t = [float(t) for t in fc.get("t") or []]

    for k, j in enumerate(idxs):
        a = axs[k]
        lab = labels[k] if labels is not None else f"y[{j}]"

        y_obs = []
        for row in ys:
            v = None
            try:
                v = row[int(j)]
            except Exception:
                v = None
            y_obs.append(v if v is None else float(v))

        a.plot(
            t_obs_f,
            [float("nan") if v is None else float(v) for v in y_obs],
            color="#111827",
            lw=1.25,
            label="Observed",
        )

        if isinstance(smooth, Mapping) and smooth.get("obs_mean") is not None:
            mu = [float(row[int(j)]) for row in smooth["obs_mean"]]
            a.plot(t_obs_f, mu, color="#1D4ED8", lw=2.0, label="Smoothed")
            if show_bands and smooth.get("obs_lo") is not None and smooth.get("obs_hi") is not None:
                lo = [float(row[int(j)]) for row in smooth["obs_lo"]]
                hi = [float(row[int(j)]) for row in smooth["obs_hi"]]
                a.fill_between(t_obs_f, lo, hi, color="#93C5FD", alpha=0.35, label="Smoothed band")

        if fc_t is not None and isinstance(fc, Mapping) and fc.get("obs_mean") is not None:
            fmu = [float(row[int(j)]) for row in fc["obs_mean"]]
            a.plot(fc_t, fmu, color="#059669", lw=2.0, ls="--", label="Forecast")
            if show_bands and fc.get("obs_lo") is not None and fc.get("obs_hi") is not None:
                flo = [float(row[int(j)]) for row in fc["obs_lo"]]
                fhi = [float(row[int(j)]) for row in fc["obs_hi"]]
                a.fill_between(fc_t, flo, fhi, color="#6EE7B7", alpha=0.25, label="Forecast band")

        a.set_ylabel(lab)
        a.grid(True, alpha=0.25)

    axs[-1].set_xlabel("t")
    if title:
        axs[0].set_title(title)
    axs[0].legend(frameon=False, ncol=2)
    return axs


def kalman_forecast(
    model,
    ys: Sequence[Sequence[float | None]],
    *,
    steps: int = 1,
    alpha: float | None = None,
) -> Mapping[str, Any]:
    """Forecast future states/observations after ingesting `ys`."""
    from . import _core

    return _core.kalman_forecast(model, [list(y) for y in ys], steps=steps, alpha=alpha)


def kalman_simulate(model, *, t_max: int, seed: int = 42) -> Mapping[str, Any]:
    """Simulate (xs, ys) from the model."""
    from . import _core

    return _core.kalman_simulate(model, t_max=t_max, seed=seed)

def local_level_model(*, q: float, r: float, m0: float = 0.0, p0: float = 1.0):
    """Construct a 1D local level (random walk) Kalman model."""
    from . import _core

    return _core.KalmanModel([[1.0]], [[float(q)]], [[1.0]], [[float(r)]], [float(m0)], [[float(p0)]])


def local_linear_trend_model(
    *,
    q_level: float,
    q_slope: float,
    r: float,
    level0: float = 0.0,
    slope0: float = 0.0,
    p0_level: float = 1.0,
    p0_slope: float = 1.0,
):
    """Construct a 2D local linear trend (level+slope) Kalman model."""
    from . import _core

    f = [[1.0, 1.0], [0.0, 1.0]]
    q = [[float(q_level), 0.0], [0.0, float(q_slope)]]
    h = [[1.0, 0.0]]
    rr = [[float(r)]]
    m0 = [float(level0), float(slope0)]
    p0 = [[float(p0_level), 0.0], [0.0, float(p0_slope)]]
    return _core.KalmanModel(f, q, h, rr, m0, p0)

def ar1_model(*, phi: float, q: float, r: float, m0: float = 0.0, p0: float = 1.0):
    """Construct a 1D AR(1) Kalman model."""
    from . import _core

    return _core.KalmanModel([[float(phi)]], [[float(q)]], [[1.0]], [[float(r)]], [float(m0)], [[float(p0)]])

def local_level_seasonal_model(
    *,
    period: int,
    q_level: float,
    q_season: float,
    r: float,
    level0: float = 0.0,
    p0_level: float = 1.0,
    p0_season: float = 1.0,
):
    """Construct a local level + seasonal (dummy seasonal) Kalman model."""
    from . import _core

    if int(period) < 2:
        raise ValueError("period must be >= 2")

    sdim = int(period) - 1
    dim = 1 + sdim

    # F: level random walk + seasonal dummy transition
    f = [[0.0 for _ in range(dim)] for _ in range(dim)]
    f[0][0] = 1.0
    for j in range(sdim):
        f[1][1 + j] = -1.0
    for i in range(1, sdim):
        f[1 + i][1 + (i - 1)] = 1.0

    q = [[0.0 for _ in range(dim)] for _ in range(dim)]
    q[0][0] = float(q_level)
    for j in range(sdim):
        q[1 + j][1 + j] = float(q_season)

    h = [[0.0 for _ in range(dim)]]
    h[0][0] = 1.0
    h[0][1] = 1.0

    rr = [[float(r)]]
    m0 = [float(level0)] + [0.0 for _ in range(sdim)]

    p0 = [[0.0 for _ in range(dim)] for _ in range(dim)]
    p0[0][0] = float(p0_level)
    for j in range(sdim):
        p0[1 + j][1 + j] = float(p0_season)

    return _core.KalmanModel(f, q, h, rr, m0, p0)

def local_linear_trend_seasonal_model(
    *,
    period: int,
    q_level: float,
    q_slope: float,
    q_season: float,
    r: float,
    level0: float = 0.0,
    slope0: float = 0.0,
    p0_level: float = 1.0,
    p0_slope: float = 1.0,
    p0_season: float = 1.0,
):
    """Construct a local linear trend + seasonal (dummy seasonal) Kalman model."""
    from . import _core

    if int(period) < 2:
        raise ValueError("period must be >= 2")

    sdim = int(period) - 1
    dim = 2 + sdim

    f = [[0.0 for _ in range(dim)] for _ in range(dim)]
    f[0][0] = 1.0
    f[0][1] = 1.0
    f[1][1] = 1.0
    for j in range(sdim):
        f[2][2 + j] = -1.0
    for i in range(1, sdim):
        f[2 + i][2 + (i - 1)] = 1.0

    q = [[0.0 for _ in range(dim)] for _ in range(dim)]
    q[0][0] = float(q_level)
    q[1][1] = float(q_slope)
    for j in range(sdim):
        q[2 + j][2 + j] = float(q_season)

    h = [[0.0 for _ in range(dim)]]
    h[0][0] = 1.0
    h[0][2] = 1.0

    rr = [[float(r)]]
    m0 = [float(level0), float(slope0)] + [0.0 for _ in range(sdim)]

    p0 = [[0.0 for _ in range(dim)] for _ in range(dim)]
    p0[0][0] = float(p0_level)
    p0[1][1] = float(p0_slope)
    for j in range(sdim):
        p0[2 + j][2 + j] = float(p0_season)

    return _core.KalmanModel(f, q, h, rr, m0, p0)


__all__ = [
    "kalman_filter",
    "kalman_smooth",
    "kalman_em",
    "kalman_fit",
    "kalman_viz_artifact",
    "plot_kalman_obs",
    "plot_kalman_states",
    "plot_kalman_obs_grid",
    "kalman_forecast",
    "kalman_simulate",
    "local_level_model",
    "local_linear_trend_model",
    "ar1_model",
    "local_level_seasonal_model",
    "local_linear_trend_seasonal_model",
]
