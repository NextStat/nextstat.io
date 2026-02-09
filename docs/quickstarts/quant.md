# Quickstart (Quant): Kalman Filter + EM Fit + Forecast

What you'll do:

- simulate a noisy AR(1) time series
- fit process/measurement noise (Q/R) using EM
- produce a short forecast

## Install

```bash
python -m pip install /path/to/nextstat-*.whl
```

Repo dev mode (run from a NextStat checkout without installing):

```bash
cd /path/to/nextstat.io
export PYTHONPATH=bindings/ns-py/python
```

## Run

```bash
python docs/quickstarts/code/quant_kalman_ar1.py
```

Output:

- prints fitted Q/R and EM convergence summary
- writes a small JSON artifact: `docs/quickstarts/out/quant_kalman_ar1_result.json`

Next steps:

- use `nextstat.timeseries.local_level_model(...)` for local-level tracking
- enable missing observations by inserting `None` in the `ys` series
