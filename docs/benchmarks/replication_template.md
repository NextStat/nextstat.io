---
title: "Replication Report Template (Fill-In)"
status: draft
last_updated: 2026-02-08
---

# Replication Report (Template)

## Summary

- Original snapshot id: `<snapshot_id>`
- Original suite: `<suite>`
- Rerun snapshot id: `<snapshot_id>`
- Rerun suite: `<suite>`
- Result: `PASS` or `FAIL`

## Rerun environment

- Operator / org: `<name>`
- Date: `<YYYY-MM-DD>`
- OS: `<os>`
- CPU: `<cpu model>`
- RAM: `<ram>`
- GPU (if any): `<gpu>`
- Rust toolchain: `<rustc -V>`
- Python: `<python -V>`

## Procedure

1. Downloaded original artifacts:
   - `snapshot_index.json`
   - `validation_pack_manifest.json`
   - `validation_report.json` (+ optional PDF)
2. Verified original signatures (if present):
   - `<verification command + output>`
3. Reran suite:
   - `<commands + key flags>`
4. Produced rerun artifacts:
   - `snapshot_index.json`
   - `validation_pack_manifest.json` (+ optional signature files)
5. Compared original vs rerun:
   - `<summary of mismatches, if any>`

## Artifacts (rerun)

- `snapshot_index.json`: `<sha256>`
- `replication_report.json`: `<sha256>`
- `validation_pack_manifest.json`: `<sha256>`
- `validation_pack_manifest.json.sig` (if signed): `<sha256>`
- `validation_pack_manifest.pub.pem` (if published): `<sha256>`

## Notes

- `<any deviations, known differences, or why mismatches are expected>`

