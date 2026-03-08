# NextStat Public Benchmarks (Seed Repo Skeleton)

This directory is a **seed** for a standalone public benchmarks repository (e.g. `nextstat-public-benchmarks`).

Goal: make benchmark snapshots **rerunnable by outsiders** with pinned environments, correctness gates, and raw artifact publishing.

Canonical benchmark program spec (in `nextstat.io`): `docs/benchmarks/public-benchmarks.md`.

## Quickstart (local)

1. Create a Python venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install pinned harness deps:

```bash
pip install -r env/python/requirements.txt
```

3. Install NextStat:

- If you have a wheel:
  - `pip install /path/to/nextstat-*.whl`
- If you are running from the `nextstat.io` monorepo:
  - build a wheel via `maturin build --release` in `bindings/ns-py`, then `pip install target/wheels/*.whl`.

4. Run the minimal HEP suite (NLL parity + timing):

```bash
python suites/hep/run.py --deterministic --out out/hep_simple_nll.json
```

5. Run the Bayesian suite (NUTS diagnostics + ESS/sec proxies):

```bash
python suites/bayesian/suite.py --deterministic --out-dir out/bayesian
python suites/bayesian/assess.py out/bayesian
```

Optional local backends:

```bash
python suites/bayesian/suite.py --deterministic --out-dir out/bayesian --backends nextstat,cmdstanpy,pymc
python suites/bayesian/assess.py out/bayesian --promotion-backend nextstat
```

In the monorepo, the canonical host-backed lane lives at `scripts/benchmarks/bench_bayesian_suite_remote.sh`. Its default contract stays `backends=nextstat`, but when you request `cmdstanpy` or `pymc` on `nextstat-bench` it now provisions those optional backends itself. If a synced `vendor/cmdstan/cmdstan-*` toolchain is present, the suite prefers it over ambient host `~/.cmdstan` state, and the PyMC lane defaults to `PYTENSOR_FLAGS=blas__ldflags=-lblas` unless the operator overrides `PYTENSOR_FLAGS`. The tracked `histfactory_simple_8p` fixture is also mapped exactly for `cmdstanpy` and `pymc` through the same bounded-`mu` plus Poisson/Gamma `shapesys` contract used by NextStat. Only non-default stacks such as `numpyro` still rely on additive `BENCH_EXTRA_PIP_PACKAGES`, and `numpyro` does not currently implement that HistFactory mapping.

For the canonical published Bayesian snapshot lane, the monorepo now also ships `scripts/benchmarks/publish_bayesian_snapshot_remote.sh`. That runner executes `scripts/publish_snapshot.py --bayesian` on `nextstat-bench`, validates the produced publish root, and syncs the full publish root back locally so the bundle includes both the snapshot directory and the publisher-refreshed `snapshot_registry.json`. The generated `README_snippet_bayesian.md` is now health-complete: it surfaces `core_quality`, `promotion_gate`, failing cases, and worst-case sampler-health metrics from `bayesian_assessment.json` instead of only showing throughput tables.

For the canonical published MAMS snapshot lane, the monorepo now also ships `scripts/benchmarks/publish_mams_snapshot_remote.sh`. That runner executes `scripts/publish_snapshot.py --mams` on `nextstat-bench`, validates the produced publish root, and syncs the full publish root back locally so the bundle includes both the snapshot directory and the publisher-refreshed `snapshot_registry.json`. Published MAMS snapshots now preserve the suite-local `mams/mams_benchmark_report.md`, while `README_snippet_mams.md` is health-complete in the same style as Bayesian: it surfaces `core_quality`, `promotion_gate`, failing cases, and worst reviewed-case metrics from `mams_assessment.json`. The stabilized public MAMS regime now uses `n_warmup=3500`, `target_accept=0.985`, `max_leapfrog=1024`, and `eps_jitter=0.0`, and its divergence telemetry now counts early-terminated divergent transitions correctly.

For repeatability evidence, the monorepo now also ships `scripts/benchmarks/bench_bayesian_multiseed_remote.sh`. That runner executes `suites/bayesian/multiseed.py` on `nextstat-bench`, derives schema-backed supplementary `derived_metrics.json`, validates the full multi-seed directory, and makes it possible to refresh the checked-in Bayesian repeatability snapshot from the same host-backed lane instead of leaving historical seed artifacts stale. When you request `pymc` in `BENCH_BACKENDS`, the same multiseed lane now self-provisions `pymc` + `arviz`, exports the explicit `PYTENSOR_FLAGS` BLAS policy, and carries PyMC repeatability rows into `derived_metrics.json`. The multiseed summary itself now also exposes health arrays for divergence rate, treedepth saturation, E-BFMI, and ESS_tail, and can be regenerated from existing `seed_*` artifacts via `--reuse-existing`.

For canonical MAMS repeatability evidence, the monorepo now also ships `scripts/benchmarks/bench_mams_multiseed_remote.sh`. That runner executes `suites/mams/multiseed.py` plus `suites/mams/assess_multiseed.py` on `nextstat-bench`, validates the full multi-seed directory, and refreshes both the schema-backed `mams_multiseed_summary.json` / `mams_multiseed_summary.md` aggregate evidence and the separate `mams_multiseed_assessment.json` / `mams_multiseed_assessment.md` repeatability verdict. The MAMS repeatability contract keeps `dataset_seed=12345` fixed across sampler seeds so `glm_logistic` measures sampler variation instead of regenerated-data variation, uses the stabilized MAMS regime (`n_warmup=3500`, `target_accept=0.985`, `max_leapfrog=1024`, `eps_jitter=0.0`), carries the tracked MAMS-vs-NUTS parity rows into the aggregate summary, and can also be regenerated from existing `seed_*` artifacts via `--reuse-existing` followed by `assess_multiseed.py`. The checked-in host-backed repeatability bundle now lives at `suites/mams/results_v1/`; with the refreshed tracked evidence its repeatability gate passes on the canonical seed set.

For expanded MAMS stress evidence, the monorepo now also ships `scripts/benchmarks/bench_mams_stress_multiseed_remote.sh`. That runner executes `suites/mams/stress_multiseed.py` plus `suites/mams/assess_stress_multiseed.py` on `nextstat-bench`, validates the full stress directory, and refreshes a separate tracked bundle at `suites/mams/stress_results_v1/`. This lane is intentionally distinct from canonical promotion evidence: it reviews supported stress cases `neal_funnel_ncp_10d` and `hier_random_intercept_non_centered`, keeps `neal_funnel_10d_centered` as a pathological control, and emits a separate `mams_stress_assessment.json` that reports `stress_readiness`, `supported_repeatability_gate`, and `pathological_control_health` without polluting the canonical `mams_assessment.json` / `mams_multiseed_assessment.json` surfaces. The current tracked host-backed stress bundle is sourced from `benchmarks/artifacts/mams_stress_validation_20260309T182549Z/nextstat-bench/`; its honest verdict is `stress_readiness=failed` because `hier_random_intercept_non_centered` reaches `max_r_hat=1.0158 > 1.01`, while `neal_funnel_ncp_10d` passes and the centered 10D funnel control stays non-gating.

6. Run the ML suite (compile vs execution, seed):

```bash
python suites/ml/suite.py --deterministic --out-dir out/ml
```

7. Run the econometrics suite (panel/DiD/IV/AIPW, seed):

```bash
python suites/econometrics/suite.py --deterministic --out-dir out/econometrics
```

Optional: enable econometrics parity baselines (statsmodels + linearmodels):

```bash
pip install -r env/python/requirements-econometrics-baselines.txt
```

Optional: enable JAX backends for ML suite (CPU):

```bash
pip install -r env/python/requirements-ml-jax-cpu.txt
```

Optional: enable JAX backends for ML suite (CUDA, on a GPU runner):

```bash
pip install -r env/python/requirements-ml-jax-cuda12.txt
```

Optional: also benchmark full MLE fits (more expensive):

```bash
python suites/hep/run.py --deterministic --fit --fit-repeat 3 --out out/hep_simple_nll_fit.json
```

## Smoke runs

Some suites support `--smoke` to reduce runtime (useful for GPU/Metal sanity checks and CI).

Example: Monte Carlo safety suite (CPU+CUDA in the same output directory):

```bash
python suites/montecarlo_safety/suite.py --out-dir out/mc_safety --deterministic --smoke --device cpu
python suites/montecarlo_safety/suite.py --out-dir out/mc_safety --deterministic --smoke --device cuda
python suites/montecarlo_safety/report.py out/mc_safety > out/mc_safety/report.md
```

## Export to a standalone repo (from the monorepo seed)

If you are starting from the `nextstat.io` monorepo and want a clean standalone repo directory
(excluding local outputs like `out/`, `tmp/`, `manifests/snapshots/`), use:

```bash
# From the monorepo root:
python3 benchmarks/nextstat-public-benchmarks/scripts/export_seed_repo.py \
  --out /path/to/nextstat-public-benchmarks
```

Optional: also stage GitHub Actions workflows into `.github/workflows/` in the exported repo:

```bash
python3 benchmarks/nextstat-public-benchmarks/scripts/export_seed_repo.py \
  --out /path/to/nextstat-public-benchmarks \
  --with-github-workflows
```

Then:

```bash
cd /path/to/nextstat-public-benchmarks
git init
git add -A
git commit -m "Initial public benchmarks harness"
```

## CI configuration (template)

The workflow templates under `ci/` expect a **pinned** NextStat wheel to be installed (so published snapshots can record the exact build being measured).

- In a standalone GitHub repo, copy these files into `.github/workflows/` (GitHub Actions only runs workflows from that folder).
- The wheel **must** match the runner OS/arch and Python version. The templates use `ubuntu-latest` + Python `3.13` by default.
- `ci/verify.yml` (PR/push): set GitHub Actions variables:
  - `NEXTSTAT_WHEEL_URL` — URL to the wheel file
  - `NEXTSTAT_WHEEL_SHA256` — SHA-256 of the wheel file (hex)
- `ci/publish.yml` (manual): either:
  - provide `nextstat_wheel_url` + `nextstat_wheel_sha256`, or
  - leave them empty and provide `nextstat_ref` to build the wheel from source (optionally override `nextstat_repo` / `nextstat_py_subdir`).
  You can also toggle `run_hep` / `run_pharma` / `run_econometrics`.

## Publish A Local Snapshot (Seed)

Generate a local snapshot directory (suite outputs + `baseline_manifest.json` + `snapshot_index.json` + README snippet) under `manifests/snapshots/<snapshot_id>/`:

```bash
python scripts/publish_snapshot.py --snapshot-id snapshot-YYYY-MM-DD --deterministic --fit --fit-repeat 3
```

`snapshot_index.json` now remains the hash inventory for the full artifact set, but it can also surface normalized `suite_health` rows when the snapshot contains assessment artifacts such as `bayesian/bayesian_assessment.json` or `mams/mams_assessment.json`. This is the registry-facing summary layer that lets published snapshots expose `core_quality` / `promotion_gate` verdicts without requiring deep inspection of nested suite directories.

`publish_snapshot.py` now refreshes the commit-backed registry automatically after each successful publish. In the canonical layout this writes `manifests/snapshot_registry.json`; for custom `--out-root` values it writes `<out-root>/snapshot_registry.json` by default so ad hoc bundles stay self-contained. Use `--registry-out` to override the location or `--no-write-registry` to disable the refresh explicitly.

The registry writer is now deterministic and supports a freshness gate:

```bash
python3 benchmarks/nextstat-public-benchmarks/scripts/write_snapshot_registry.py \
  --snapshots-root benchmarks/nextstat-public-benchmarks/manifests/snapshots \
  --out benchmarks/nextstat-public-benchmarks/manifests/snapshot_registry.json \
  --check
```

That check fails if the committed `manifests/snapshot_registry.json` no longer matches the committed snapshot bundles.

To generate or refresh the commit-backed registry manually across published snapshots, point the standalone writer at one or more snapshot directories or explicit `snapshot_index.json` files:

```bash
python3 benchmarks/nextstat-public-benchmarks/scripts/write_snapshot_registry.py \
  --snapshots-root benchmarks/nextstat-public-benchmarks/manifests/snapshots \
  --out benchmarks/nextstat-public-benchmarks/manifests/snapshot_registry.json
python3 benchmarks/nextstat-public-benchmarks/scripts/validate_artifacts.py --strict \
  benchmarks/nextstat-public-benchmarks/manifests/snapshot_registry.json
```

`snapshot_registry.json` validates under `nextstat.snapshot_registry.v1` and simply lifts the same snapshot-level `suite_health` rows that already live in each published `snapshot_index.json`, so registry, automation, and live pages consume the same health vocabulary.

The committed snapshot set is now no longer health-empty: it includes host-backed publish bundles under `manifests/snapshots/bayesian-publisher-20260309T111048Z/`, the earlier MAMS snapshots, and the refreshed promotion-passing MAMS bundle `manifests/snapshots/mams-publisher-20260309T173500Z/`. `manifests/snapshot_registry.json` now surfaces both a promotion-passing Bayesian row and a promotion-passing MAMS stable-surface row directly from tracked `nextstat-bench` evidence. Repeatability evidence is tracked separately under `suites/bayesian/results_v10/` and `suites/mams/results_v1/`; the MAMS lane now includes a separate `mams_multiseed_assessment.json` repeatability verdict so worst-across-seed health and parity failures do not need to be reconstructed from raw arrays.
Expanded stress evidence for MAMS is tracked separately again under `suites/mams/stress_results_v1/`, with its own `mams_stress_multiseed_summary.json` and `mams_stress_assessment.json` so stress-lane failures do not silently rewrite the canonical stable-surface verdict.

By default this runs the `hep` and `pharma` suites. Add `--bayesian` to include the Bayesian suite:

```bash
python scripts/publish_snapshot.py --snapshot-id snapshot-YYYY-MM-DD --deterministic --fit --fit-repeat 3 --bayesian
```

Add `--econometrics` to include the econometrics suite:

```bash
python scripts/publish_snapshot.py --snapshot-id snapshot-YYYY-MM-DD --deterministic --econometrics
```

If you want the baseline manifest to pin the exact measured NextStat build, pass the wheel path so the manifest records `nextstat.wheel_sha256`:

```bash
python scripts/publish_snapshot.py --snapshot-id snapshot-YYYY-MM-DD --deterministic --nextstat-wheel /path/to/nextstat-*.whl
```

When `--nextstat-wheel` is provided, the wheel is copied into the snapshot directory as `nextstat_wheel.whl` (so the snapshot can be self-contained for DOI publishing).

Suites:

- `hep/` (pyhf vs NextStat)
- `pharma/` (NextStat-only seed + baseline templates)
- `bayesian/` (NextStat-only seed: NUTS diagnostics + ESS/sec proxies; only when `--bayesian` is passed)
- `ml/` (NextStat-only seed: cold-start TTFR vs warm-call throughput; optional JAX cases)
- `econometrics/` (NextStat seed + optional parity vs statsmodels/linearmodels; only when `--econometrics` is passed)
- `glm/` (GLM parity vs statsmodels/sklearn/glum; only when `--glm` is passed)
- `survival/` (survival + truth-recovery modes; only when `--survival` is passed)
- `timeseries/` (Kalman + GARCH; only when `--timeseries` is passed)
- `evt/` (GEV/GPD; only when `--evt` is passed)
- `insurance/` (loss reserving; only when `--insurance` is passed)
- `meta_analysis/` (fixed/random effects; only when `--meta-analysis` is passed)
- `mams/` (MAMS/NUTS/MCLMC comparisons; only when `--mams` is passed)
- `montecarlo_safety/` (fault-tree MC throughput; only when `--montecarlo-safety` is passed)

When `--mams` is enabled, the seed publisher now emits both:

- `mams/mams_suite.json` — raw suite index
- `mams/mams_assessment.json` — explicit separation between `core_quality` and `promotion_gate`
- `mams/mams_benchmark_report.md` — the tracked human-facing MAMS suite report
- `README_snippet_mams.md` — a top-level health-complete MAMS snapshot summary driven by `mams_assessment.json`

When `--bayesian` is enabled, the seed publisher now emits both:

- `bayesian/bayesian_suite.json` — raw suite index
- `bayesian/bayesian_assessment.json` — explicit separation between `core_quality` and `promotion_gate`, plus a machine-readable `review_summary` for worst-case health metrics across reviewed cases

The top-level `README_snippet_bayesian.md` generated by `publish_snapshot.py` now consumes that assessment artifact and surfaces the published Bayesian health verdict directly in the human-facing snapshot summary.

## Validate Artifacts

Validate any produced artifacts (suite indexes validate their referenced case JSONs):

```bash
python scripts/validate_artifacts.py --strict out
python scripts/validate_artifacts.py --strict manifests/snapshots
```

## DOI Publishing (Template)

For Zenodo/DOI publishing guidance and metadata templates, see `zenodo/`.

## What This Seed Provides

- `env/` pinned environment scaffolding (Python + Rust + Docker templates)
- `CITATION.cff` citation metadata template (fill DOI/version on release)
- `manifests/schema/` JSON Schemas for results + baseline manifests
- `suites/` suite layout (runnable: `hep`, `pharma`, `bayesian`, `ml`, `econometrics`, `glm`, `survival`, `timeseries`, `evt`, `insurance`, `meta_analysis`, `mams`, `montecarlo_safety`)
- `ci/` workflow templates for verify/publish (standalone repo)

## Notes

- The harness is intentionally separate from the product repo: it should be auditable and runnable without building all of NextStat.
- Use `--json-only` in the validation-pack tooling when you want reproducibility artifacts without `matplotlib`/PDF.
