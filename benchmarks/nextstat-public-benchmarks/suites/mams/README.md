# MAMS Suite

Canonical tracked benchmark suite for CPU MAMS vs CPU NUTS.

## Local

Run the suite locally from the repo root:

```bash
python3 benchmarks/nextstat-public-benchmarks/suites/mams/suite.py \
  --out-dir benchmarks/nextstat-public-benchmarks/out/mams \
  --deterministic
python3 benchmarks/nextstat-public-benchmarks/suites/mams/report.py \
  benchmarks/nextstat-public-benchmarks/out/mams
python3 benchmarks/nextstat-public-benchmarks/suites/mams/assess.py \
  benchmarks/nextstat-public-benchmarks/out/mams
python3 benchmarks/nextstat-public-benchmarks/scripts/validate_artifacts.py \
  --strict benchmarks/nextstat-public-benchmarks/out/mams

python3 benchmarks/nextstat-public-benchmarks/suites/mams/multiseed.py \
  --out-dir benchmarks/nextstat-public-benchmarks/out/mams_multiseed \
  --deterministic
python3 benchmarks/nextstat-public-benchmarks/suites/mams/assess_multiseed.py \
  benchmarks/nextstat-public-benchmarks/out/mams_multiseed
python3 benchmarks/nextstat-public-benchmarks/scripts/validate_artifacts.py \
  --strict benchmarks/nextstat-public-benchmarks/out/mams_multiseed

python3 benchmarks/nextstat-public-benchmarks/suites/mams/stress_multiseed.py \
  --out-dir benchmarks/nextstat-public-benchmarks/out/mams_stress \
  --deterministic
python3 benchmarks/nextstat-public-benchmarks/suites/mams/assess_stress_multiseed.py \
  benchmarks/nextstat-public-benchmarks/out/mams_stress
python3 benchmarks/nextstat-public-benchmarks/scripts/validate_artifacts.py \
  --strict benchmarks/nextstat-public-benchmarks/out/mams_stress
```

Default canonical config:

- `backends=nextstat_mams,nextstat_nuts`
- `seeds=42`
- `n_chains=4`
- `n_warmup=3500`
- `n_samples=2000`
- `target_accept=0.985`
- `max_leapfrog=1024`
- `eps_jitter=0.0`
- `init_l=0.0` means the stable default `L = sqrt(d)` in preconditioned space

Default canonical multi-seed repeatability config:

- `backends=nextstat_mams,nextstat_nuts`
- `seeds=42,0,123`
- `dataset_seed=12345`
- `n_chains=4`
- `n_warmup=3500`
- `n_samples=2000`
- `target_accept=0.985`
- `max_leapfrog=1024`
- `eps_jitter=0.0`
- `parity_warn_z=8`
- `parity_fail_z=12`

Default expanded stress config:

- `backends=nextstat_mams,nextstat_nuts`
- `seeds=42,0,123`
- `dataset_seed=12345`
- `n_chains=4`
- `n_warmup=3500`
- `n_samples=2000`
- `target_accept=0.985`
- `n_groups=20`
- `n_per_group=20`
- supported cases: `neal_funnel_ncp_10d`, `hier_random_intercept_non_centered`
- pathological control: `neal_funnel_10d_centered`

## Remote

Run the canonical suite on the shared benchmark host:

```bash
scripts/benchmarks/bench_mams_suite_remote.sh
```

The remote runner:

- syncs the minimal required workspace snapshot for MAMS benchmarking,
- builds `ns-py` in release mode on `nextstat-bench`,
- runs `suites/mams/suite.py`,
- renders `mams_benchmark_report.md`,
- writes `mams_assessment.json` / `mams_assessment.md` so core quality and promotion readiness stay separate,
- uses the stabilized CPU MAMS regime (`warmup=3500`, `target_accept=0.985`, `max_leapfrog=1024`, `eps_jitter=0.0`) for the canonical stable-surface lane,
- validates the JSON artifacts against the tracked schema,
- syncs the result bundle back into `benchmarks/artifacts/`.

For the canonical host-backed repeatability lane on `nextstat-bench`, use:

```bash
scripts/benchmarks/bench_mams_multiseed_remote.sh
```

That repeatability runner:

- runs `suites/mams/multiseed.py` on `nextstat-bench`,
- keeps `dataset_seed=12345` fixed across sampler seeds so `glm_logistic` measures sampler variance instead of regenerated-data variance,
- uses the same stabilized CPU MAMS regime (`warmup=3500`, `target_accept=0.985`, `max_leapfrog=1024`, `eps_jitter=0.0`) as the canonical stable-surface lane,
- writes `mams_multiseed_summary.json` / `mams_multiseed_summary.md` with aggregate health and parity tables,
- writes `mams_multiseed_assessment.json` / `mams_multiseed_assessment.md` with the machine-readable repeatability gate and worst-across-seed review summary,
- validates the full multi-seed directory under `validate_artifacts.py --strict`,
- syncs the host-backed repeatability bundle back into `benchmarks/artifacts/`.

For the expanded stress repeatability lane on `nextstat-bench`, use:

```bash
scripts/benchmarks/bench_mams_stress_multiseed_remote.sh
```

That stress runner:

- runs `suites/mams/stress_multiseed.py` plus `suites/mams/assess_stress_multiseed.py` on `nextstat-bench`,
- keeps canonical MAMS promotion evidence separate from expanded stress evidence,
- reviews `neal_funnel_ncp_10d` and `hier_random_intercept_non_centered` as supported stress cases,
- keeps `neal_funnel_10d_centered` as a pathological control that is reported but not parity-gating,
- writes `mams_stress_multiseed_summary.json` / `.md` and `mams_stress_assessment.json` / `.md`,
- validates the full stress bundle via `validate_artifacts.py --strict`,
- syncs the host-backed bundle back into `benchmarks/artifacts/`.

For the canonical published snapshot lane on `nextstat-bench`, use:

```bash
scripts/benchmarks/publish_mams_snapshot_remote.sh
```

That publisher runner:

- runs `scripts/publish_snapshot.py --mams` on `nextstat-bench`,
- validates the full publish root instead of only the nested snapshot directory,
- syncs the full publish root back locally so the returned bundle includes `snapshot_registry.json`,
- preserves the suite-local `mams/mams_benchmark_report.md`,
- renders a health-complete top-level `README_snippet_mams.md` from `mams_assessment.json`,
- keeps human-facing snapshot review and machine-readable stable-surface verdicts in the same publish bundle.

The committed public snapshot set now includes a refreshed health-complete MAMS publish bundle at `benchmarks/nextstat-public-benchmarks/manifests/snapshots/mams-publisher-20260309T173500Z/`. Its `README_snippet_mams.md` now surfaces the same `core_quality=passed` / `promotion_gate=passed` verdict and worst reviewed-case metrics that appear in `mams/mams_assessment.json`, and the committed `snapshot_registry.json` lifts that same row directly for registry consumers.

The stable public surface now also reports divergent MAMS transitions honestly in `sample_stats["diverging"]` and `diagnostics["divergence_rate"]`: early-terminated non-finite energy-error transitions are no longer silently counted as clean rejects.

The committed host-backed repeatability bundle now lives at `benchmarks/nextstat-public-benchmarks/suites/mams/results_v1/`. It is sourced from `benchmarks/artifacts/mams_multiseed_validation_20260309T164500Z/nextstat-bench/`, keeps `dataset_seed=12345` fixed across canonical seeds, and now includes `mams_multiseed_assessment.json` / `mams_multiseed_assessment.md` beside the raw summary. In the refreshed tracked bundle the repeatability gate is `passed`; worst reviewed `max_r_hat` is `1.0087` on `neal_funnel_2d`, while all parity rows remain `ok`. It can be regenerated in place via:

```bash
python3 benchmarks/nextstat-public-benchmarks/suites/mams/multiseed.py \
  --out-dir benchmarks/nextstat-public-benchmarks/suites/mams/results_v1 \
  --seeds 42,0,123 \
  --reuse-existing
python3 benchmarks/nextstat-public-benchmarks/suites/mams/assess_multiseed.py \
  benchmarks/nextstat-public-benchmarks/suites/mams/results_v1
```

The committed host-backed stress bundle now lives at `benchmarks/nextstat-public-benchmarks/suites/mams/stress_results_v1/`. It is sourced from `benchmarks/artifacts/mams_stress_validation_20260309T182549Z/nextstat-bench/` and intentionally carries a separate stress verdict instead of rewriting the canonical stable lane. In the current tracked evidence:

- `stress_readiness=failed`
- `supported_repeatability_gate=failed`
- `pathological_control_health=passed`
- sole supported blocker: `hier_random_intercept_non_centered` with `max_r_hat=1.0158`
- `neal_funnel_ncp_10d` passes cleanly with `max_r_hat=1.0066`
- `neal_funnel_10d_centered` remains a non-gating pathological control and stays `ok,ok,ok`

It can be regenerated in place via:

```bash
python3 benchmarks/nextstat-public-benchmarks/suites/mams/stress_multiseed.py \
  --out-dir benchmarks/nextstat-public-benchmarks/suites/mams/stress_results_v1 \
  --seeds 42,0,123 \
  --reuse-existing
python3 benchmarks/nextstat-public-benchmarks/suites/mams/assess_stress_multiseed.py \
  benchmarks/nextstat-public-benchmarks/suites/mams/stress_results_v1
```

Important: this is the stable verification lane for canonical MAMS benchmarking. Do not use dirty local sampler-matrix WIP harnesses to make promotion or stable-surface claims about MAMS.
