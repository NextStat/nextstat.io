# ROOT Profile Scan Parity (Apex2)

This doc describes the ROOT/HistFactory profile-likelihood parity loop for `q(mu)`:

- ROOT reference: RooFit/RooStats via `hist2workspace` + `root` macro scan.
- NextStat candidate: `nextstat.infer.profile_scan`.

The source of truth runner is:

- `tests/validate_root_profile_scan.py` (single-case runner)
- `tests/apex2_root_suite_report.py` (multi-case aggregator)

## Prerequisites

- ROOT tools in `PATH`: `root`, `hist2workspace`
- Python deps (recommended): use `./.venv/bin/python` so `pyhf[xmlio]` and `uproot` are available.

## Quick run (realistic fixture)

```sh
PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/apex2_root_suite_report.py \
  --cases tests/fixtures/trex_parity_pack/cases_realistic_fourtop.json \
  --dq-atol 2e-2 \
  --mu-hat-atol 5e-2 \
  --deterministic \
  --out tmp/apex2_root_suite_realistic_fourtop.json
```

This writes a per-run directory under `tmp/root_parity_suite/<case>/run_<ts>/` containing:

- `root_profile_scan.json`
- `nextstat_profile_scan.json`
- `summary.json` (diff metrics)

## What is checked

The key parity metrics are stored in `summary.json`:

- `diff.max_abs_dq_mu`: maximum absolute difference in `q(mu)` over the scan grid
- `diff.mu_hat`: difference in fitted `mu_hat` (NextStat - ROOT)

The Apex2 default tolerances are intentionally robust to optimizer/version noise on realistic exports:

- `dq_atol = 2e-2`
- `mu_hat_atol = 5e-2`

## Notes

- `tests/validate_root_profile_scan.py` supports both `--pyhf-json` and `--histfactory-xml` modes.
- For deterministic runs, it sets NextStat to parity mode and single-thread evaluation:
  `nextstat.set_eval_mode("parity")` and `nextstat.set_threads(1)` (best-effort).

