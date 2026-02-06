# Phase 8: Time Series (Kalman Filter)

NextStat Phase 8 introduces time series and state space models. The current baseline implementation is a time-invariant linear-Gaussian state space model with:

- Kalman filter (log-likelihood, predicted + filtered states)
- RTS smoother (smoothed states)

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
	nextstat timeseries kalman-forecast --input kalman_1d.json --steps 10 --alpha 0.05
	nextstat timeseries kalman-simulate --input kalman_1d.json --t-max 50 --seed 123
	```

## EM options

`kalman-em` can also estimate scalar `F`/`H` for 1D models (`n_state=1`, `n_obs=1`):

- CLI: `--estimate-f true` to estimate `F[0,0]`.
- CLI: `--estimate-h true` to estimate `H[0,0]`.
- Python: `nextstat.timeseries.kalman_em(..., estimate_f=True, estimate_h=True)`.

## Fit helper

- Python: `nextstat.timeseries.kalman_fit(model, ys, forecast_steps=10)` runs EM + RTS smoothing (+ optional forecast) and returns a dict with `model`, `em`, `smooth`, `forecast`.
- CLI: `nextstat timeseries kalman-fit --input ... --forecast-steps 10` outputs the same sections in JSON.

## Missing observations

- Python: use `None` inside `ys` (per component).
- CLI JSON: use `null` inside `ys` (per component).

## Standard models

- CLI: specify exactly one of `model`, `local_level`, `local_linear_trend`, `ar1`, `local_level_seasonal`, `local_linear_trend_seasonal`.
- Python: use `nextstat.timeseries.local_level_model(...)`, `nextstat.timeseries.local_linear_trend_model(...)`, `nextstat.timeseries.ar1_model(...)`, `nextstat.timeseries.local_level_seasonal_model(...)`, or `nextstat.timeseries.local_linear_trend_seasonal_model(...)`.

## JSON contract (Python)

Both `kalman_filter` and `kalman_smooth` return plain Python dicts containing nested lists for vectors/matrices:

- `*_means`: `T x n_state`
- `*_covs`: `T x n_state x n_state`

Forecast intervals:

- `kalman_forecast(..., alpha=0.05)` adds `obs_lower` and `obs_upper` (marginal normal intervals).
