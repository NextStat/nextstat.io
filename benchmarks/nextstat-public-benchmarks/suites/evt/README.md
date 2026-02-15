# EVT Suite (Seed)

Extreme Value Theory benchmark suite for NextStat: GEV and GPD model fitting with parity checks against scipy.

Cases:
- `gev_block_maxima_500` -- GEV fit on 500 block maxima samples
- `gev_block_maxima_5000` -- GEV fit on 5000 block maxima samples
- `gpd_threshold_500` -- GPD fit on 500 threshold exceedances
- `gpd_threshold_5000` -- GPD fit on 5000 threshold exceedances

Data is generated synthetically via inverse CDF sampling with fixed seeds for reproducibility.

Baseline: `scipy.stats.genextreme` / `scipy.stats.genpareto` (skipped with `status="warn"` if scipy is not installed).

This suite produces **publishable** JSON artifacts under pinned schemas:
- `nextstat.evt_benchmark_result.v1` (per case)
- `nextstat.evt_benchmark_suite_result.v1` (suite index)

## Run

Single case:

```bash
python run.py --case gev_block_maxima_500 --model gev --n 500 --out ../../out/evt_gev_500.json
```

Full suite:

```bash
python suite.py --deterministic --out-dir ../../out/evt
```
