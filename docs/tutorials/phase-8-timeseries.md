# Phase 8: Time Series (Kalman Filter)

NextStat Phase 8 introduces time series and state space models. The current baseline implementation is a time-invariant linear-Gaussian state space model with:

- Kalman filter (log-likelihood, predicted + filtered states)
- RTS smoother (smoothed states)

## Python quickstart

```python
import nextstat

# 1D local level model:
# x_t = x_{t-1} + w_t,  w_t ~ N(0, q)
# y_t = x_t     + v_t,  v_t ~ N(0, r)
f = [[1.0]]
q = [[0.1]]
h = [[1.0]]
r = [[0.2]]
m0 = [0.0]
p0 = [[1.0]]

model = nextstat.KalmanModel(f, q, h, r, m0, p0)
ys = [[0.9], [1.2], [0.8], [1.1]]

fr = nextstat.timeseries.kalman_filter(model, ys)
print(fr["log_likelihood"])
print(fr["filtered_means"][-1], fr["filtered_covs"][-1])

sr = nextstat.timeseries.kalman_smooth(model, ys)
print(sr["smoothed_means"][0], sr["smoothed_covs"][0])
```

## JSON contract (Python)

Both `kalman_filter` and `kalman_smooth` return plain Python dicts containing nested lists for vectors/matrices:

- `*_means`: `T x n_state`
- `*_covs`: `T x n_state x n_state`

