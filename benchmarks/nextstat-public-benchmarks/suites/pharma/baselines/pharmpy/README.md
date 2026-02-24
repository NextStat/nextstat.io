# Pharmpy Baseline

Python baseline entrypoint for the pharma suite:

```bash
python run.py --in <case.json> --out <baseline.json> --repeat 5
```

Current status:
- Detects `pharmpy` availability.
- Emits schema-valid baseline JSON (`nextstat.pharma_baseline_result.v1`).
- Supports generated `pop_pk_1c_oral` cases with additive error.
- Runs `pharmpy -> nlmixr` and reports timed `fit_time_s` (`min` over repeats).
- Includes fallback parsing of `run1.RDATA` when pharmpy result parsing fails
  (known issue in some `pharmpy-core` versions).
- Returns `status=skipped` with explicit reason when unsupported/not installed.
