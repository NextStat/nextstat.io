# MaS Baseline

Python baseline entrypoint for the pharma suite:

```bash
python run.py --in <case.json> --out <baseline.json> --repeat 5
```

Current status:
- Detects `MaS` package availability (common module names: `mas`, `MaS`).
- Emits schema-valid baseline JSON (`nextstat.pharma_baseline_result.v1`).
- Returns `status=skipped` with explicit import/API reason when the installed
  package is not the pharmacometric MaS engine.
- Runner integration is intentionally strict to avoid benchmarking an unrelated
  `mas` package from PyPI.
