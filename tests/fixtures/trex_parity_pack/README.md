# Apex2 ROOT/TREx parity fixture pack

This directory contains **committed** Apex2 case packs for the ROOT/HistFactory parity loop.
They are meant to be **minimal but representative** and to work without any external TREx exports.

## Minimal pack

- `cases_minimal.json`
  - `simple_fixture` (`mode=pyhf-json`) — quick smoke case.
  - `histfactory_fixture` (`mode=histfactory-xml`) — repo HistFactory fixture.
  - `pyhf_xmlimport` (`mode=histfactory-xml`) — pyhf validation fixture (1-channel; OverallSys + StatError + NormFactor).
  - `pyhf_multichannel` (`mode=histfactory-xml`) — pyhf validation fixture (signal+control; ShapeSys).
  - `pyhf_coupled_histosys` (`mode=histfactory-xml`) — pyhf validation fixture (coupled NP / shared HistoSys-like behavior).

## Realistic packs (pyhf JSON fixtures)

- `cases_realistic.json`
  - Larger/realistic pyhf workspaces committed under `tests/fixtures/` (can be slow; some cases are known to fail strict ROOT-vs-NextStat thresholds).
- `cases_realistic_fourtop.json`
  - Single realistic case (`tttt-prod_workspace.json`) intended to be “always runnable” on dev machines.

## TREx export dirs (HistFactory export directories)

- `cases_trex_exports.json`
  - Uses committed HistFactory export directories under `tests/fixtures/trex_exports/` that include `combination.xml` + `data.root` + channel XMLs.
  - These are intended to cover “real export dir” path semantics beyond `pyhf.writexml` fixtures.

## How to run (requires ROOT)

From the repo root (in an environment with `root` + `hist2workspace` and Python deps):

```sh
PYTHONPATH=bindings/ns-py/python python3 tests/apex2_root_suite_report.py \
  --cases tests/fixtures/trex_parity_pack/cases_minimal.json \
  --keep-going \
  --deterministic \
  --out tmp/apex2_root_suite_minimal.json
```

Realistic (fast):

```sh
PYTHONPATH=bindings/ns-py/python python3 tests/apex2_root_suite_report.py \
  --cases tests/fixtures/trex_parity_pack/cases_realistic_fourtop.json \
  --deterministic \
  --out tmp/apex2_root_suite_realistic_fourtop.json
```

TREx export dirs:

```sh
PYTHONPATH=bindings/ns-py/python python3 tests/apex2_root_suite_report.py \
  --cases tests/fixtures/trex_parity_pack/cases_trex_exports.json \
  --keep-going \
  --deterministic \
  --out tmp/apex2_root_suite_trex_exports.json
```

To record a baseline manifest (cluster / ROOT env):

```sh
PYTHONPATH=bindings/ns-py/python python3 tests/record_baseline.py \
  --only root \
  --root-suite-existing tmp/apex2_root_suite_minimal.json \
  --root-cases-existing tests/fixtures/trex_parity_pack/cases_minimal.json \
  --out-dir tmp/baselines
```
