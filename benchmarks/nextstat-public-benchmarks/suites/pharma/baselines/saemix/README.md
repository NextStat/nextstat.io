# saemix Baseline

R baseline entrypoint for the pharma suite:

```bash
Rscript run.R --in <case.json> --out <baseline.json> --repeat 5
```

Current status:
- Detects `saemix` availability.
- Emits schema-valid baseline JSON (`nextstat.pharma_baseline_result.v1`).
- Returns `status=skipped` with an explicit reason when not installed or when the
  case-to-saemix mapping is not yet configured.

