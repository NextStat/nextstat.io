# DOI Publishing (Zenodo) Template

Goal: publish **immutable benchmark snapshots** with a DOI so third parties can cite the exact artifact set.

This directory contains templates and a runbook. The actual publication step requires manual Zenodo access.

## What To Publish

For each snapshot, publish a zipped artifact set that contains at least:

- `baseline_manifest.json`
- `snapshot_index.json` (hash inventory of the artifact set)
- `nextstat_wheel.whl` (recommended: self-contained exact build)
- suite outputs (e.g. `hep/hep_suite.json` and per-case JSONs)
- `README_snippet.md` (human summary table)

Recommended: include the harness commit SHA and NextStat version in the snapshot id and in the manifest.

## Zenodo Integration Options

1. GitHub releases -> Zenodo archive (recommended for the standalone benchmarks repo)
2. Manual Zenodo deposition upload (useful for one-off snapshots)

## CITATION.cff Guidance

The harness repo should include a `CITATION.cff` describing how to cite:

- the harness repo itself
- the snapshot DOI (when published)

For snapshots: prefer citing the DOI + snapshot id rather than “latest”.

Template: `CITATION.cff` at the repo root (fill in `doi`/`version`/`date-released` on release).

## Templates

- `zenodo.json` contains a minimal metadata stub you can adapt for Zenodo deposits.
- `package_snapshot.py` packages a local snapshot directory into a `.tar.gz` plus sha256 and a derived `zenodo_deposition.json`.
- `publish_to_zenodo.py` creates a Zenodo deposition draft and uploads the packaged archive (token read from `ZENODO_TOKEN`).

## Published (Sandbox)

Sandbox is for pipeline validation only (the DOI is not a “real” DOI for citation).

- DOI: `10.5072/zenodo.437330`
- Record URL: `https://sandbox.zenodo.org/records/437330`
- Deposition ID: `437330`

## Published (Production)

- DOI: `10.5281/zenodo.18542624`
- Record URL: `https://zenodo.org/records/18542624`
