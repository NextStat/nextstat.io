# NextStat - Business Strategy Analysis (Draft)

Status: draft (strategy memo, not a commitment).  
Audience: founders/maintainers; early design decisions; planning input.  
Goal: document why the plan is shaped as it is (P0 correctness + CPU-first), and which business risks this mitigates.

## 1) Executive Summary

NextStat should win primarily through a correctness contract and a maintainable architecture, not through "GPU-first" positioning:

- Parity-first: deterministic CPU mode matches pyhf/TRExFitter outputs within the Phase 1 contract.
- CPU-first: a large fraction of real scientific workloads run on CPU clusters and batch systems.
- Open-core: OSS covers scientific workflows, Pro covers compliance/audit/enterprise value.

Recommended focus for the first 6-9 months:

1. pyhf parity + a usable CLI/Python API
2. CPU parallelism + AD (gradients/Hessian)
3. GPU as an optional accelerator later

## 2) Target Segments (initial priority)

### 2.1 HEP / scientific computing (primary wedge)

Why: clear early-adopter market, strong pain (fit time + complex toolchains), and a high-quality reference implementation (pyhf) for validation.

Value props:

- faster fits (CPU parallelism + AD),
- explicit validation (parity suite),
- easier installation (pip + wheels) and "no ROOT by default".

### 2.2 Finance / model risk (secondary)

Why: higher willingness to pay and strong requirements for reproducibility and auditability.

Value props (Pro):

- audit trail, model registry, governance,
- reproducible runs, signed artifacts, approvals.

### 2.3 Medical / 21 CFR Part 11 (later)

Why: high compliance and sales cost; do not pursue before the core is stable and widely validated.

## 3) Monetization (open-core)

### 3.1 OSS (AGPL)

Goal: become a best-in-class inference engine and a standard interface layer.

Keep in OSS:

- core likelihood engine + inference (fit/scan/ranking baseline),
- parsers/translators (pyhf JSON, HistFactory XML import),
- CLI and Python API,
- CPU performance path (Rayon/SIMD) and a deterministic reference mode.

### 3.2 Pro (Commercial)

Pro sells:

- Audit and compliance: 21 CFR Part 11 audit trail, e-signatures, validation packs
- Scale: distributed orchestration (K8s/Ray) + job management UX
- Hub/Dashboard: model registry, RBAC, collaboration
- Support/SLA and services

The OSS/Pro boundary should be defined in `docs/legal/open-core-boundaries.md` and reviewed by counsel.

## 4) Packaging and Pricing (working hypotheses)

- OSS: free under AGPL (creates commercial leverage)
- Pro: annual subscription per team/org plus support
- Enterprise: custom contract (SLA, on-prem, compliance bundle)

Candidate value metrics:

- number of projects/models in the registry,
- number of runs/fit-hours orchestrated,
- seats (often the noisiest proxy).

## 5) Timeline Guidance (aligned with plans)

### Months 0-4 (Phase 0-1)

Goal: first working fit plus parity on small models.

Critical:

- determinism and numeric contract (`docs/plans/standards.md`),
- validators and fixtures,
- minimal CLI/Python API.

### Months 4-9 (Phase 2)

Goal: make the product useful on real models (50-200 nuisance parameters).

Critical:

- AD (gradients/HVP) for stable uncertainties/ranking,
- CPU parallelism + batching,
- GPU optional and non-blocking.

### Months 9-15 (Phase 3)

Goal: production readiness (documentation, visualization, validation, stable releases).

## 6) Key Risks and Mitigations

### 6.1 Parity drift (math diverges from pyhf)

Mitigation: a single `standards.md`, golden tests, deterministic mode as the baseline contract.

### 6.2 Over-engineering (GPU too early)

Mitigation: GPU as an optional phase; never block the core.

### 6.3 Legal ambiguity (AGPL + commercial)

Mitigation: explicit boundaries + counsel review + contribution policy (DCO/CLA) before external contributions scale.

### 6.4 Adoption friction (hard installation)

Mitigation: wheels, minimal dependencies, fast "hello fit" path, examples.

## 7) Decisions Needed (to keep the plan executable)

1. Contribution policy: DCO vs CLA.
2. Repo layout: split OSS and Pro repos vs single repo + private modules.
3. Release policy: which artifacts are published under AGPL and in what form.
4. Minimum supported platforms: Linux x86_64 + macOS arm64 (initially).

