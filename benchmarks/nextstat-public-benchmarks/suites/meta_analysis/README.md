# Meta-Analysis Suite

This directory contains the **meta-analysis benchmark suite** for the standalone public benchmarks repo. It covers inverse-variance fixed-effects and DerSimonian-Laird random-effects pooling across synthetic study sets of varying size.

The suite produces **publishable** JSON artifacts under pinned schemas:
- `nextstat.meta_analysis_benchmark_result.v1` (per case)
- `nextstat.meta_analysis_benchmark_suite_result.v1` (suite index)

Cases: `fixed_effects_10` (10-study fixed-effects), `random_effects_10` (10-study DL random-effects), `random_effects_50` (50-study DL random-effects). Parity is checked against `pymare` (FixedEffectsMeta and DerSimonianLaird estimators). Both NextStat and pymare are timed with N=100 repeats by default.

## Run

```bash
python3 suites/meta_analysis/suite.py --deterministic --out-dir out/meta_analysis
```

Optional baseline (skipped with `status="warn"` if missing):
- `pymare` for fixed-effects and DerSimonian-Laird parity
