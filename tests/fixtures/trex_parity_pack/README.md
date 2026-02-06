# Apex2 ROOT/TREx parity fixture pack

This directory contains **committed** Apex2 case packs for the ROOT/HistFactory parity loop.
They are meant to be **minimal but representative** and to work without any external TREx exports.

## Minimal pack

- `cases_minimal.json`
  - `simple_fixture` (`mode=pyhf-json`) — quick smoke case.
  - `histfactory_fixture` (`mode=histfactory-xml`) — uses `tests/fixtures/histfactory/combination.xml` + `data.root`.

## How to run (requires ROOT)

From the repo root (in an environment with `root` + `hist2workspace` and Python deps):

```sh
PYTHONPATH=bindings/ns-py/python python3 tests/apex2_root_suite_report.py \
  --cases tests/fixtures/trex_parity_pack/cases_minimal.json \
  --keep-going \
  --deterministic \
  --out tmp/apex2_root_suite_minimal.json
```

To record a baseline manifest (cluster / ROOT env):

```sh
PYTHONPATH=bindings/ns-py/python python3 tests/record_baseline.py \
  --only root \
  --root-suite-existing tmp/apex2_root_suite_minimal.json \
  --root-cases-existing tests/fixtures/trex_parity_pack/cases_minimal.json \
  --out-dir tmp/baselines
```

