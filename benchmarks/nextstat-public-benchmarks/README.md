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

Optional: also benchmark full MLE fits (more expensive):

```bash
python suites/hep/run.py --deterministic --fit --fit-repeat 3 --out out/hep_simple_nll_fit.json
```

## CI configuration (template)

The workflow templates under `ci/` expect a **pinned** NextStat wheel to be installed (so published snapshots can record the exact build being measured).

- `ci/verify.yml` (PR/push): set GitHub Actions variables:
  - `NEXTSTAT_WHEEL_URL` — URL to the wheel file
  - `NEXTSTAT_WHEEL_SHA256` — SHA-256 of the wheel file (hex)
- `ci/publish.yml` (manual): provide `nextstat_wheel_url` + `nextstat_wheel_sha256` as workflow inputs.

## Publish A Local Snapshot (Seed)

Generate a local snapshot directory (suite outputs + `baseline_manifest.json` + `snapshot_index.json` + README snippet) under `manifests/snapshots/<snapshot_id>/`:

```bash
python scripts/publish_snapshot.py --snapshot-id snapshot-YYYY-MM-DD --deterministic --fit --fit-repeat 3
```

If you want the baseline manifest to pin the exact measured NextStat build, pass the wheel path so the manifest records `nextstat.wheel_sha256`:

```bash
python scripts/publish_snapshot.py --snapshot-id snapshot-YYYY-MM-DD --deterministic --nextstat-wheel /path/to/nextstat-*.whl
```

This currently includes both:

- `hep/` (pyhf vs NextStat)
- `pharma/` (NextStat-only seed + baseline templates)

## DOI Publishing (Template)

For Zenodo/DOI publishing guidance and metadata templates, see `zenodo/`.

## What This Seed Provides

- `env/` pinned environment scaffolding (Python + Rust + Docker templates)
- `manifests/schema/` JSON Schemas for results + baseline manifests
- `suites/` runnable suite layout (currently: `hep` and `pharma` seeds)
- `ci/` workflow templates for verify/publish (standalone repo)

## Notes

- The harness is intentionally separate from the product repo: it should be auditable and runnable without building all of NextStat.
- Use `--json-only` in the validation-pack tooling when you want reproducibility artifacts without `matplotlib`/PDF.
