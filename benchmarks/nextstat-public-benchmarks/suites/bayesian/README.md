# Bayesian Suite (ESS/sec vs Stan + PyMC)

This directory is reserved for the **Bayesian benchmark suite** in the standalone public benchmarks repo.

Canonical methodology + runbook lives in the main docs site:

- `docs/benchmarks/suites/bayesian.md`

Planned measurements (publishable snapshots):

- ESS/sec (bulk + tail) with declared inference settings
- divergence / treedepth saturation rates
- max R-hat, min ESS, min E-BFMI
- wall-time distributions and environment manifest

Seed baselines (today):

- `histfactory_simple_8p` — simple HistFactory workspace (few params)
- `glm_logistic_regression` — synthetic logistic regression
- `hier_random_intercept_non_centered` — synthetic hierarchical logistic random intercepts (non-centered)

Note: the seed reports an ESS/sec proxy computed as `min(ESS_bulk) / wall_time`, where wall time includes warmup + sampling.

Status: runnable seed.

- `nextstat` backend: always available (dependency-light).
- `cmdstanpy` and `pymc` backends: optional (best-effort). If deps are missing, artifacts are emitted as `warn` with an actionable `reason`.
- The canonical `histfactory_simple_8p` case now has exact external-backend mappings for `cmdstanpy` and `pymc`: `mu` stays bounded on `[0, 10]`, each `shapesys` nuisance stays positive, and the auxiliary constraint is preserved as the same Poisson/Gamma Barlow-Beeston term used by the tracked HistFactory workspace. This support is intentionally scoped to the tracked `datasets/simple_workspace.json` fixture, not arbitrary HistFactory workspaces.

Run (NextStat-only):

```bash
python3 suites/bayesian/suite.py --deterministic --out-dir out/bayesian
python3 suites/bayesian/assess.py out/bayesian
```

Run multi-seed (stability check, keeps dataset fixed via `--dataset-seed`):

```bash
python3 suites/bayesian/multiseed.py --deterministic --out-dir out/bayesian_multiseed --seeds 42,0,123 --dataset-seed 12345
python3 suites/bayesian/derive_metrics.py out/bayesian_multiseed
```

Regenerate the multiseed summary from existing `seed_*` artifacts without rerunning the suite:

```bash
python3 suites/bayesian/multiseed.py --out-dir out/bayesian_multiseed --seeds 42,0,123 --reuse-existing
```

Run with optional backends:

```bash
python3 suites/bayesian/suite.py --deterministic --out-dir out/bayesian --backends nextstat,cmdstanpy,pymc
python3 suites/bayesian/assess.py out/bayesian --promotion-backend nextstat
```

## Remote

Run the canonical suite on the shared benchmark host:

```bash
scripts/benchmarks/bench_bayesian_suite_remote.sh
```

Run the canonical multi-seed repeatability lane on the shared benchmark host:

```bash
scripts/benchmarks/bench_bayesian_multiseed_remote.sh
```

The remote runner:

- syncs the minimal required workspace snapshot for Bayesian benchmarking,
- builds `ns-py` in release mode on `nextstat-bench`,
- runs `suites/bayesian/suite.py`,
- renders `bayesian_benchmark_report.md`,
- writes `bayesian_assessment.json` / `bayesian_assessment.md` so core quality and promotion readiness stay separate,
- validates the JSON artifacts against the tracked schema,
- syncs the result bundle back into `benchmarks/artifacts/`.

Run the canonical Bayesian publisher lane on the shared benchmark host:

```bash
scripts/benchmarks/publish_bayesian_snapshot_remote.sh
```

That runner executes `scripts/publish_snapshot.py --bayesian` on `nextstat-bench`, validates the produced publish root, and syncs back the full publish root so the bundle includes both the snapshot directory and the publisher-refreshed `snapshot_registry.json`. The generated `README_snippet_bayesian.md` now includes a health verdict section sourced from `bayesian_assessment.json`, and `bayesian_assessment.json` itself now carries a machine-readable `review_summary` so worst-case health metrics do not need to be reconstructed from `reviewed_cases`.

Default remote policy uses `--backends nextstat` so the canonical host lane stays stable and dependency-light.

When you request optional external backends on `nextstat-bench`, the runner provisions the needed Python stack itself:

- `BENCH_BACKENDS=nextstat,cmdstanpy` installs `cmdstanpy` and provisions or reuses a local tracked `vendor/cmdstan/cmdstan-*` toolchain inside the synchronized seed repo snapshot.
- `BENCH_BACKENDS=nextstat,pymc` installs `pymc` and `arviz` in the remote venv before running the suite.
- `BENCH_EXTRA_PIP_PACKAGES=...` remains additive for non-default stacks such as `numpyro`; it is no longer required for the standard `cmdstanpy` or `pymc` remote path.

When a synced `vendor/cmdstan/cmdstan-*` toolchain exists, the suite prefers it over ambient host `~/.cmdstan` state. This keeps the public benchmark lane self-contained and makes the recorded `backend_meta.cmdstan_path` reproducible from the synchronized seed repo snapshot.

For PyMC on `nextstat-bench`, the runner now sets `PYTENSOR_FLAGS=blas__ldflags=-lblas` by default unless the operator already provided `PYTENSOR_FLAGS`. This removes the pip/PyTensor BLAS warning on the shared host and makes the compute path explicit in `backend_meta`.

`numpyro` remains additive and does not currently implement the canonical HistFactory workspace mapping; that path still emits `backend_not_supported_for_model` for `histfactory_simple`.

The suite publishes two separate artifact layers:

- `bayesian_suite.json` for raw benchmark execution evidence
- `bayesian_assessment.json` / `bayesian_assessment.md` for explicit `core_quality` vs `promotion_gate`, plus canonical aggregate health summary fields for the reviewed backend

The committed public snapshot set now includes a host-backed health-complete Bayesian publish bundle at `manifests/snapshots/bayesian-publisher-20260309T111048Z/`, and the committed `manifests/snapshot_registry.json` surfaces that bundle's Bayesian health row.

Keep these layers separate. The raw suite artifact should stay policy-free; promotion or discovery claims belong in the assessment artifact.

For checked-in repeatability evidence, the suite also supports a host-backed multi-seed lane:

- `scripts/benchmarks/bench_bayesian_multiseed_remote.sh` runs `multiseed.py` on `nextstat-bench`
- `suites/bayesian/derive_metrics.py` derives supplementary `derived_metrics.json` from the per-seed case artifacts
- the full multi-seed directory is now strict-schema-valid because both `bayesian_multiseed_summary.json` and `derived_metrics.json` are schema-backed
- when `BENCH_BACKENDS` includes `pymc`, the multiseed remote lane also self-provisions `pymc` + `arviz`, exports the same explicit `PYTENSOR_FLAGS` BLAS policy used by the single-run host-backed lane, and includes schema-backed PyMC ESS/leapfrog rows in `derived_metrics.json`
- `bayesian_multiseed_summary.json` now also carries health arrays (`divergence_rate`, `max_treedepth_rate`, `min_ebfmi`, `min_ess_tail`), and the Markdown summary includes a worst-across-seeds health table instead of hiding those diagnostics in per-case JSON only
