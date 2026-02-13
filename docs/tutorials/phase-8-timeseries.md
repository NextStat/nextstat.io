# Phase 8: Time Series (Kalman Filter)

NextStat Phase 8 introduces time series and state space models. The current baseline implementation is a time-invariant linear-Gaussian state space model with:

- Kalman filter (log-likelihood, predicted + filtered states)
- RTS smoother (smoothed states)

See also:
- Phase 6 (regression / GLMs): `docs/tutorials/phase-6-regression.md`
- Phase 7 (hierarchical / multilevel): `docs/tutorials/phase-7-hierarchical.md`

## Assumptions and diagnostics (quick checklist)

- Baseline Phase 8 models are **linear-Gaussian** state space models; likelihood and uncertainty are Normal.
- For AR(1), a common stability condition is `abs(phi) < 1` (stationary). For ARMA baselines, keep parameters in a stable region.
- Diagnostics to watch:
  - EM convergence (`fit["em"]["converged"]`, `fit["em"]["loglik_trace"]`)
  - One-step-ahead forecast errors (innovations) and whether residuals look roughly Gaussian / uncorrelated

## Python quickstart

```python
import nextstat

# 1D local level model
model = nextstat.timeseries.local_level_model(q=0.1, r=0.2, m0=0.0, p0=1.0)
ys = [[0.9], [1.2], [0.8], [1.1]]

fr = nextstat.timeseries.kalman_filter(model, ys)
print(fr["log_likelihood"])
print(fr["filtered_means"][-1], fr["filtered_covs"][-1])

sr = nextstat.timeseries.kalman_smooth(model, ys)
print(sr["smoothed_means"][0], sr["smoothed_covs"][0])

# Fit (EM) + smoother + forecast, then build a plot-friendly artifact
fit = nextstat.timeseries.kalman_fit(model, ys, forecast_steps=10)
art = nextstat.timeseries.kalman_viz_artifact(fit, ys, level=0.95)
# Plotting requires matplotlib:
# nextstat.timeseries.plot_kalman_obs(art, title="Kalman fit")
# nextstat.timeseries.plot_kalman_states(art, title="Latent states")
# nextstat.timeseries.plot_kalman_obs_grid(art, title="Observed components")
```

## AR(1) fit + forecast

```python
import nextstat

# 1D AR(1): x[t] = phi * x[t-1] + eps[t], y[t] = x[t] + eta[t]
true = nextstat.timeseries.ar1_model(phi=0.8, q=0.2, r=0.1, m0=0.0, p0=1.0)
sim = nextstat.timeseries.kalman_simulate(true, t_max=30, seed=123, init="mean")
ys = sim["ys"]  # list of [y_t]

# Start from a rough model, estimate phi (F), and forecast.
init = nextstat.timeseries.ar1_model(phi=0.2, q=1.0, r=1.0, m0=0.0, p0=1.0)
fit = nextstat.timeseries.kalman_fit(
    init,
    ys,
    estimate_q=True,
    estimate_r=True,
    estimate_f=True,  # estimate phi in F[0,0] (1D-only baseline)
    forecast_steps=10,
)
print("converged:", fit["em"]["converged"], "iters:", fit["em"]["n_iter"])
print("phi_hat:", fit["em"]["f"][0][0])
print("forecast t[-1]:", fit["forecast"]["t"][-1], "y_mean[-1]:", fit["forecast"]["obs_means"][-1][0])
```

## ARMA(1,1) forecast (fixed-parameter baseline)

```python
import nextstat

# Note: baseline Kalman implementation requires r > 0 (use tiny r to approximate r=0).
model = nextstat.timeseries.arma11_model(phi=0.6, theta=0.2, sigma2=0.4, r=1e-12)
ys = [[0.1], [0.2], [-0.1], [0.0], [0.15], [0.05]]

fc = nextstat.timeseries.kalman_forecast(model, ys, steps=5, alpha=0.05)
print(fc["obs_means"])
```

## CLI quickstart

Create an input JSON like:

```json
{
  "local_level": { "q": 0.1, "r": 0.2, "m0": 0.0, "p0": 1.0 },
  "ys": [[0.9], [1.2], [0.8], [1.1]]
}
```

Then run:

```sh
nextstat timeseries kalman-filter --input kalman_1d.json
nextstat timeseries kalman-smooth --input kalman_1d.json
nextstat timeseries kalman-em --input kalman_1d.json --max-iter 50 --tol 1e-6
nextstat timeseries kalman-fit --input kalman_1d.json --max-iter 50 --tol 1e-6 --forecast-steps 10
nextstat timeseries kalman-viz --input kalman_1d.json --max-iter 50 --tol 1e-6 --level 0.95 --forecast-steps 10
nextstat timeseries kalman-forecast --input kalman_1d.json --steps 10 --alpha 0.05
nextstat timeseries kalman-simulate --input kalman_1d.json --t-max 50 --seed 123
```

## EM options

`kalman-em` can also estimate scalar `F`/`H` for 1D models (`n_state=1`, `n_obs=1`):

- CLI: `--estimate-f true` to estimate `F[0,0]`.
- CLI: `--estimate-h true` to estimate `H[0,0]`.
- Python: `nextstat.timeseries.kalman_em(model, ys, estimate_f=True, estimate_h=True)`.

## Fit helper

- Python: `nextstat.timeseries.kalman_fit(model, ys, forecast_steps=10)` runs EM + RTS smoothing (+ optional forecast) and returns a dict with `model`, `em`, `smooth`, `forecast`.
- CLI: `nextstat timeseries kalman-fit --input kalman_1d.json --forecast-steps 10` outputs the same sections in JSON.

## Missing observations

- Python: use `None` inside `ys` (per component).
- CLI JSON: use `null` inside `ys` (per component).

## Standard models

- CLI: specify exactly one of `model`, `local_level`, `local_linear_trend`, `ar1`, `arma11`, `local_level_seasonal`, `local_linear_trend_seasonal`.
- Python: use `nextstat.timeseries.local_level_model(q=0.1, r=0.2, m0=0.0, p0=1.0)`, `nextstat.timeseries.local_linear_trend_model(q_level=0.1, q_slope=0.05, r=0.2, level0=0.0, slope0=0.0, p0_level=1.0, p0_slope=1.0)`, `nextstat.timeseries.ar1_model(phi=0.8, q=0.2, r=0.1, m0=0.0, p0=1.0)`, `nextstat.timeseries.arma11_model(phi=0.6, theta=0.2, sigma2=0.4, r=1e-12, m0_x=0.0, m0_eps=0.0, p0_x=1.0, p0_eps=1.0)`, `nextstat.timeseries.local_level_seasonal_model(period=12, q_level=0.1, q_season=0.01, r=0.2, level0=0.0, p0_level=1.0, p0_season=1.0)`, or `nextstat.timeseries.local_linear_trend_seasonal_model(period=12, q_level=0.1, q_slope=0.05, q_season=0.01, r=0.2, level0=0.0, slope0=0.0, p0_level=1.0, p0_slope=1.0, p0_season=1.0)`.

ARMA(1,1) baseline:

- CLI: `{"arma11": {"phi": 0.6, "theta": 0.2, "sigma2": 0.4, "r": 1e-12}, "ys": [[0.1], [0.2], [-0.1], [0.0], [0.15], [0.05]]}`
- Python: `nextstat.timeseries.arma11_model(phi=0.6, theta=0.2, sigma2=0.4, r=1e-12, m0_x=0.0, m0_eps=0.0, p0_x=1.0, p0_eps=1.0)`
- Limitation: the baseline Kalman implementation requires `r > 0` (use a tiny value like `1e-12` to approximate `r=0`).

## JSON contract (Python)

Both `kalman_filter` and `kalman_smooth` return plain Python dicts containing nested lists for vectors/matrices:

- `*_means`: `T x n_state`
- `*_covs`: `T x n_state x n_state`

Forecast intervals:

- `kalman_forecast(model, ys, steps=10, alpha=0.05)` adds `obs_lower` and `obs_upper` (marginal normal intervals).
