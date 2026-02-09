# Open-Core Boundaries (NextStat) - Draft

This document is a working draft and requires review by legal counsel. It is not legal advice.

Goal: define practical OSS vs Commercial boundaries early, so the roadmap and repository structure remain enforceable and do not undermine the licensing model.

## 1) Principle

- OSS (AGPL): everything required for a correct statistical inference engine, reproducibility, and baseline workflows (fit / scan / ranking).
- Pro (Commercial): enterprise value around the engine: audit/compliance, orchestration, collaboration, governance, and UI.

## 2) Proposed Module Split (Initial)

### 2.1 OSS (AGPL)

- `ns-core`: types, model interfaces, shared primitives
- `ns-compute`: NLL / expected data kernels (CPU reference + performance modes)
- `ns-inference`: minimizers, fits, scans, ranking (no enterprise orchestration)
- `ns-translate`: ingestion/conversion (pyhf JSON, HistFactory XML import)
- `ns-cli`: CLI
- `ns-py`: Python bindings (PyO3)

### 2.2 Pro (Commercial)

- `ns-audit`: audit trail, e-signatures, validation packs
- `ns-compliance`: domain-specific reporting (e.g., regulatory formats)
- `ns-scale`: distributed execution / orchestration primitives
- `ns-hub`: model registry, versioning, governance
- `ns-dashboard`: UI / monitoring

Note: GPU backends can be OSS or Pro depending on strategy. If GPU acceleration is meant to be a commercial differentiator, this should be explicitly reflected in the plans and reviewed with counsel.

## 3) Repository Layout Decision (Decide Early)

This should be decided before taking external contributors or commercial customers.

Option A (simplest):  
- Public repo: OSS crates (AGPL)  
- Separate private repo: Pro crates (Commercial)

Option B (single monorepo):  
- OSS crates public, Pro crates as a private submodule / split tooling

Default recommendation: Option A.

Rationale: lower risk of license mixing and accidental publication of proprietary code.

## 4) Contributions Policy (Decide Early)

Options:

- DCO: lightweight, commit-based sign-off
- CLA: separate agreement (sometimes preferred for open-core)

Default recommendation: start with DCO for OSS, introduce a CLA only if counsel strongly recommends it.

## 5) Trademark / Branding (Baseline Policy)

Draft policy (needs counsel review):

- Allow descriptive use: "compatible with NextStat"
- Disallow using "NextStat" as the name of a product or fork without permission
- Logos / trademarks governed by a separate policy and license

## 6) Release Policy (Artifact Boundaries)

Define:

- which binaries/wheels are published under AGPL,
- which builds are Pro-only,
- what telemetry / update checks are allowed (default should be opt-in).

