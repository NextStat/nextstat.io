# nlmixr2 Baseline

This directory holds an **R baseline runner** for the pharma suite using `nlmixr2`.

Status: runnable (for additive-error 1-compartment oral population PK cases produced by
`suites/pharma/run.py` for `foce_*` / `saem_*`).

The runner:
- reads a case JSON produced by the pharma suite,
- reconstructs a NONMEM-style event table (`ID/TIME/DV/EVID/AMT/CMT`),
- fits `nlmixr2` with `focei` or `saem`,
- reports timing + parameter recovery in `nextstat.pharma_baseline_result.v1`.

Run:

```bash
R_LIBS_USER=/path/to/r-lib \
Rscript run.R \
  --in ../../out/pharma/cases/foce_1c_oral_nsub_100.json \
  --out ../../out/pharma/baselines/nlmixr2_foce_100.json \
  --method focei \
  --repeat 3
```

Arguments:
- `--in` (required): pharma case JSON path
- `--out` (required): output baseline JSON path
- `--method` (optional): `focei` (default) or `saem`
- `--repeat` (optional): timed fit repetitions after one warmup compile (default: `1`)

Pinning guidance:

- Use a pinned R toolchain + `renv.lock` (or a Docker image) for reproducible installs.
- Record `sessionInfo()` and package versions in the output JSON.
