# Benchmark Artifact Policy

This policy defines what benchmark and release-evidence material belongs in:

- git
- CI artifacts
- GitHub Release assets
- external storage

The goal is simple: keep the public repository small, reproducible, and auditable.

## Class A: committed canonical contracts

These belong in git.

Allowed examples:
- schemas
- example JSON/spec files
- tiny deterministic fixtures
- support matrices
- acceptance criteria
- promotion decisions
- one minimal accepted evidence bundle per stable surface when it is part of the public contract

Typical locations:
- [docs/schemas](docs/schemas)
- [docs/specs](docs/specs)
- [docs/benchmarks](docs/benchmarks)
- [tests/fixtures](tests/fixtures)

## Class B: published release evidence

These should be attached to GitHub Release artifacts, not stored as growing git history.

Examples:
- validation-pack bundles
- M15 bundles
- release-grade benchmark evidence
- signed manifests
- release PDFs

Canonical surface:
- GitHub Release assets produced by [release.yml](.github/workflows/release.yml)

## Class C: CI-transient artifacts

These do not belong in git.

Examples:
- raw benchmark runs
- compare reports
- transient rerender proofs
- temporary report bundles
- promotion drafts

Canonical surface:
- GitHub Actions artifact retention
- local `tmp/`

Canonical local location:
- [tmp](tmp)

## Class D: research/raw history

These must stay out of git.

Examples:
- timestamp forests
- repeated machine-specific reruns
- exploratory campaigns
- per-seed raw trees
- long benchmark histories

Use:
- CI artifacts
- release attachments
- external object storage / registry backends

## Hard rules

1. Never commit `tmp` outputs as public evidence.
2. Never commit raw timestamped benchmark trees unless they are the canonical accepted public bundle.
3. Never keep more history in git than is required for the public contract.
4. Every committed benchmark artifact in public repo surfaces must have an exact public reference.
5. Public benchmark docs must not reference local absolute filesystem paths or `tmp` outputs.

## Current repository interpretation

- [benchmarks/artifacts](benchmarks/artifacts): only minimal committed public evidence should remain
- [bench_results](bench_results): only explicitly referenced public benchmark outputs may remain
- [tmp](tmp): local/CI transient only

## Release policy tie-in

Release evidence should be emitted by:
- [scripts/apex2/pre_release_gate.sh](scripts/apex2/pre_release_gate.sh)
- [release.yml](.github/workflows/release.yml)

and then published as release assets, not accumulated as repo baggage.
