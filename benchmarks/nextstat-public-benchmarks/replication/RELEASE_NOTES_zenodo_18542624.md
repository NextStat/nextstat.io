# Replication Bundle: Zenodo Record 18542624

This release publishes a replication bundle for the production Zenodo snapshot:

- DOI: 10.5281/zenodo.18542624
- Record: https://zenodo.org/records/18542624

Bundle contents:

- `snapshot_comparison.json` (machine diff; `ok=true` indicates dataset IDs + hashes matched and HEP parity checks passed)
- `signed_report_draft.md` (human template with DOI/URL filled; intended to be finalized and signed by an external replicator)
- `rerun_snapshots/` (local rerun artifacts: baseline manifest + raw suite outputs)

Artifacts:

- `replication-rerun-prod-doi-18542624.tar.gz`
- `replication-rerun-prod-doi-18542624.tar.gz.sha256`

Verification:

1. Verify checksum: `sha256sum -c replication-rerun-prod-doi-18542624.tar.gz.sha256`
2. Extract: `tar -xzf replication-rerun-prod-doi-18542624.tar.gz`
3. Inspect: `replication-rerun-prod-doi-18542624/snapshot_comparison.json`

