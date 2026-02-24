# Pharma Suite (Seed)

This is a **pharmacometrics** benchmark suite for NextStat:

- PK (1-compartment oral) likelihood timing
- optional MLE fit wall-time (NextStat)
- optional external baselines for population PK:
  - `nlmixr2` (FOCEI/SAEM)
  - `torsten` label (CmdStan/Stan MAP baseline runner)
  - `saemix` (R; SAEM baseline runner)
  - `mas` (Python/C++; package detection + explicit skip reason if unavailable)
  - `pharmpy` (Python; pharmpy+nlmixr baseline runner)

Run a single case:

```bash
python run.py --deterministic --out ../../out/pharma_pk_1c_oral.json
```

Suite runner (multiple generated cases):

```bash
python suite.py --deterministic --out-dir ../../out/pharma
```

Suite runner with external baselines:

```bash
python suite.py \
  --deterministic \
  --fit --fit-repeat 7 \
  --run-baselines \
  --baselines nlmixr2,torsten,saemix,mas,pharmpy \
  --baseline-repeat 5 \
  --torsten-iter 1200 \
  --baseline-r-libs-user /path/to/r-lib \
  --baseline-cmdstan /path/to/cmdstan-2.38.0 \
  --out-dir ../../out/pharma
```

Harness defaults:
- If `--baseline-r-libs-user` is not provided and `<repo>/.r_libs` exists, suite
  exports `R_LIBS_USER=<repo>/.r_libs` for all R baselines.
- If `--baseline-cmdstan` is not provided and `<repo>/.cache/cmdstan/cmdstan-*`
  exists, suite exports `CMDSTAN` with the newest detected directory.
- Baseline runner logs are written to `out_dir/baselines/logs/*.log`.
