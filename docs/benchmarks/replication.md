---
title: "Third-Party Replication Runbook (Signed Reports)"
status: draft
last_updated: 2026-02-08
---

# Third-Party Replication Runbook (Signed Reports)

This page is a **step-by-step template** for replicating a published NextStat benchmark snapshot and producing a **publishable, signed** replication artifact.

If you are new to the idea, start with the blog post: [/blog/third-party-replication-signed-report](/blog/third-party-replication-signed-report).

## Inputs (what you need from the original snapshot)

From the published snapshot artifacts, download at minimum:

- `snapshot_index.json` (hash index of published artifacts)
- `validation_pack_manifest.json` (hashes + sizes for core validation pack files)
- `validation_report.json` (+ optional PDF)

If the snapshot ships signatures, also download:

- `validation_pack_manifest.sha256`
- `validation_pack_manifest.sha256.bin`
- `validation_pack_manifest.json.sig`
- `validation_pack_manifest.pub.pem`

## Step 1: Verify the published manifest signature (optional)

If the original snapshot includes OpenSSL signature files:

```bash
openssl pkeyutl -verify -pubin -inkey validation_pack_manifest.pub.pem -rawin \
  -in validation_pack_manifest.sha256.bin \
  -sigfile validation_pack_manifest.json.sig
```

## Step 2: Rerun the suite (produce your own artifacts)

Your rerun must use the **same suite definition** and **same dataset IDs** as the original snapshot.

In this repo, the most common “publishable” correctness gate is the validation pack entrypoint:

```bash
make validation-pack
```

If you want a minimal, fixture-driven rerun (no Apex2 master execution), use:

```bash
bash validation-pack/render_validation_pack.sh \
  --out-dir tmp/validation_pack_rerun \
  --workspace tests/fixtures/simple_workspace.json \
  --apex2-master tests/fixtures/apex2_master_min_plus.json \
  --deterministic
```

## Step 3: Write your snapshot index (artifact hashes)

Create `snapshot_index.json` for your rerun artifact directory:

```bash
python3 scripts/benchmarks/write_snapshot_index.py \
  --suite apex2-nightly-slow \
  --artifacts-dir tmp/validation_pack_rerun \
  --out tmp/validation_pack_rerun/snapshot_index.json \
  --snapshot-id external-YYYY-MM-DD-your-org
```

Schema: `docs/schemas/benchmarks/snapshot_index_v1.schema.json`.

## Step 4: Compare against the original snapshot index

You can compare `snapshot_index.json` files using the replication report tool:

```bash
python3 scripts/benchmarks/write_replication_report.py \
  --original-index /path/to/original/snapshot_index.json \
  --replica-index tmp/validation_pack_rerun/snapshot_index.json \
  --out tmp/validation_pack_rerun/replication_report.json \
  --notes "CPU-only, deterministic. See env manifest in README."
```

Schema: `docs/schemas/benchmarks/replication_report_v1.schema.json`.

## Optional: One-command bundle helper

To generate a `.replication/` bundle folder (replication report + digest + optional signature) in one command:

```bash
bash scripts/benchmarks/make_replication_bundle.sh \
  --original-index /path/to/original/snapshot_index.json \
  --artifacts-dir tmp/validation_pack_rerun \
  --suite apex2-nightly-slow \
  --snapshot-id external-YYYY-MM-DD-your-org \
  --notes "CPU-only, deterministic rerun"
```

## Step 5: Sign your rerun (publishable artifact)

Recommended: sign the SHA-256 digest bytes of `validation_pack_manifest.json` with OpenSSL.

```bash
openssl genpkey -algorithm ed25519 -out tmp/manifest_priv.pem
openssl pkey -in tmp/manifest_priv.pem -pubout -out tmp/manifest_pub.pem

bash validation-pack/render_validation_pack.sh \
  --out-dir tmp/validation_pack_rerun \
  --workspace tests/fixtures/complex_workspace.json \
  --deterministic \
  --sign-openssl-key tmp/manifest_priv.pem \
  --sign-openssl-pub tmp/manifest_pub.pem
```

Verify:

```bash
openssl pkeyutl -verify -pubin -inkey tmp/manifest_pub.pem -rawin \
  -in tmp/validation_pack_rerun/validation_pack_manifest.sha256.bin \
  -sigfile tmp/validation_pack_rerun/validation_pack_manifest.json.sig
```

## Step 6: Publish your replication bundle

Publish (as a GitHub release, Zenodo upload, or a public object store) at minimum:

- your `snapshot_index.json`
- your `replication_report.json`
- your `validation_pack_manifest.json` + signature files
- your raw results and environment manifest (hardware + toolchains)

Example (production): replication bundle record `10.5281/zenodo.18543606` links to the published snapshot `10.5281/zenodo.18542624`.

Live registry pages on nextstat.io:

- [/docs/benchmark-results](/docs/benchmark-results)
- [/docs/snapshot-registry](/docs/snapshot-registry)

## Templates

- `docs/benchmarks/replication_template.md` (human-readable report template)
- `docs/specs/replication_report_v1.example.json` (machine-readable example)

## Public Benchmarks Seed Repo (Bootstrap)

For the standalone public benchmarks harness seed (directory: `benchmarks/nextstat-public-benchmarks/`), replication templates and lightweight tooling live under:

- `benchmarks/nextstat-public-benchmarks/replication/`

## CI drift detection (NextStat nightly)

For internal drift detection (CI), we recommend using "yesterday's nightly" as the baseline rather than a DOI-published snapshot:

- nightly run uploads `snapshot_index.json` + validation pack artifacts
- next nightly downloads the **previous successful run** artifact and generates a `replication_report.json`

This catches regressions between consecutive commits without pinning to a "frozen" DOI artifact.

Important nuance:

- `replication_report.json` is **hash-level** and may change between commits for non-semantic reasons (provenance fields, noisy perf outputs).
- For `validation_report.json`, CI should gate on a **semantic comparator** (suite/overall status regressions), emitting a `validation_drift_summary.json` alongside the replication bundle.
