<!--
Plan
Created: 2026-02-08
Owner: nextstat.io
Purpose: Align BMCP epics/tasks to an execution plan (trust + adoption).
-->

# Trust + Agentic Bridge + Vertical Expansion Plan (Feb 2026)

This plan turns the strategy into **shippable trust artifacts** and a **contract-grade tool layer**.

Status: active planning (BMCP epics/tasks updated on 2026-02-08).

## Why This Plan Exists

NextStat’s bottleneck (Feb 2026) is not core inference features. It is **trust + adoption**:

1. People need proof they can reproduce.
2. Teams need machine-readable artifacts and human-readable documents.
3. Vertical users (pharma/econ) need connectors (R/Python) and familiar framing.
4. “Agentic” demos only work if tool semantics are correct and outputs are stable.

## Ground Truth From the Repo (What We Already Have)

1. **Spec vs Speed** architecture already exists:
   - `ns_compute::EvalMode::{Parity,Fast}` and parity contracts (`docs/internal/plans/standards.md`, `docs/pyhf-parity-contract.md`).
2. **Apex2** already produces deterministic JSON artifacts (`--deterministic`) for validation (`tests/apex2_master_report.py` + runners).
3. A first **LLM tools surface** already exists:
   - `nextstat.tools` provides OpenAI-compatible tool definitions + `execute_tool(...)`.
   - But: outputs are not versioned, determinism controls are not exposed, and at least one tool’s semantics is currently incorrect (CLs treated as p-value).

## What We Will Ship (Deliverables)

### D1. Contract-Grade Tool API (JSON)

Goal: an LLM (or any automation) can call tools and get **versioned, deterministic, validated** outputs.

Deliverables:
- Tool input+output schemas (versioned) committed to repo.
- Unified response envelope for every tool call:
  - `schema_version`, `ok`, `result`, `error`, `meta`.
- Deterministic execution controls:
  - `deterministic`, `eval_mode` (`parity|fast`), `threads`, `device`, `seed` where relevant.
- Correct semantics:
  - CLs is CLs (not p-value). Discovery p-values/Z0 must be explicitly defined and computed.

Definition of done:
- Golden tool outputs are stable in deterministic mode.
- Schema validation passes in CI.

### D2. Public Benchmarks Distribution + Publishing

Goal: a third party can reproduce the numbers.

Deliverables:
- `nextstat-benchmarks` repo (or an equivalent packaged suite) with pinned environments.
- Suites:
  - HEP: pyhf parity + ROOT workflow comparisons where meaningful.
  - Bayesian: ESS/sec vs Stan/PyMC on selected models.
  - Pharma: PK/NLME suite with publicly reproducible baselines:
    - prioritize **nlmixr2** (R) and **Torsten/Stan** first; NONMEM/Monolix as methodology/optional runs.
  - ML: JAX compile vs steady-state throughput microbench.
- Publishing:
  - baseline manifests + artifacts + DOI snapshots.

Definition of done:
- “one command” run on a fresh machine produces JSON reports matching published baselines within tolerances.

### D3. Unified Validation Report Pack (JSON + PDF)

Goal: an auditable “Validation Report” artifact that can be handed to reviewers (GxP/CSA style).

Deliverables:
- `validation_report.json` with `schema_version=validation_report_v1`.
- Optional `validation_report.pdf` generated deterministically from JSON.
- CLI entrypoint `nextstat validation-report ...`.

Definition of done:
- On the reference suite, report generation is deterministic and CI uploads artifacts.

### D4. Demos (AI Bridge + Verticals)

Goal: one killer demo per wedge, built on the tool contract.

Deliverables:
- Physics Assistant demo (ROOT/HS3/pyhf -> fit/scan/limits/p-values + plots).
- Pharma: “Population PK modeling in the browser” tutorial (WASM) using the same reporting artifacts.
- Econometrics: real-time volatility forecasting demo (Kalman + optional GARCH) with report export.

## Execution Order (Dependencies)

1) **D1 Tool API contract** (must come first)
   - Everything else depends on stable semantics + stable outputs.
2) **D3 Validation report v1**
   - Consumes Apex2 and workspace audit; becomes the primary “trust PDF”.
3) **D2 Benchmarks packaging**
   - Reuses the report schemas and publishing pipeline.
4) **D4 Demos + Docs**
   - Built on the same artifacts; no bespoke one-off outputs.
5) **Hardening**
   - Remove crash vectors (e.g., panic paths for ROOT ZSTD) and clarify server GPU modes.

## Key Design Decisions

### Tools: Deterministic by Default (for trust)

- Default for tools intended for reporting: `deterministic=true`, `eval_mode=parity`, `threads=1`.
- Allow opt-in fast mode for interactive exploration only.

Rationale:
- Fast mode can be nondeterministic due to thread scheduling and reductions.
- Trust artifacts must be reproducible.

### Tools: Versioned Schemas and Envelopes

Every tool output includes:
- `schema_version` (tool-specific)
- `meta` with:
  - `nextstat_version`, `git_commit`
  - `eval_mode`, `threads`, `device`, `deterministic`
  - `seed` only if the tool is stochastic

Rationale:
- Enables regression testing, long-term compatibility, and auditability.

### Correctness: Don’t Conflate CLs and p-values

- `hypotest` returns **CLs** (pyhf-compatible).
- Discovery p-values / Z0 require explicit tail-prob computation and must be separate tools.

Rationale:
- Wrong semantics is worse than missing features; it destroys trust.

## BMCP Mapping (Epics)

Primary epics:
- Trust Benchmarks: `ee9d3b45-e62d-49c2-8134-d2352c7378cc`
- Validation Report PDF: `3ee435da-d41d-441d-9c9d-0e6aef3da570`
- LLM Oracle / Tools + Agent Demo: `c74becb9-afd7-4008-b791-62b5d6dcdb8d`
- Pharma: `1a3ad381-eb69-46b3-8a66-8f93f751de33`
- Docs personas: `797853dc-4e21-4c24-8458-ee8512acbcba`
- Hardening: `52d6cd0d-76b7-4047-a2b0-18c66eb30077`

## 2-Week Sprint Proposal (Starting 2026-02-09)

### Week 1 (2026-02-09 .. 2026-02-15): Tool Contract + Correctness

Ship:
- tool envelope + schema_version
- deterministic controls
- fix CLs vs p-value semantics
- golden tool regression harness

### Week 2 (2026-02-16 .. 2026-02-22): Validation Report v1 (JSON + PDF)

Ship:
- `validation_report_v1` generator + CLI
- PDF generator MVP
- CI artifact upload

After that: benchmarks repo packaging + vertical demo(s) using the same artifacts.

## Risks / Watch Items

1. Tool semantic bugs (CLs vs p-value) are trust-critical.
2. Determinism “leaks” (threads, timestamps, nondeterministic dict ordering) must be systematically prevented.
3. Public pharma baselines must start with open-source tools (nlmixr2/Torsten) to keep results reproducible for outsiders.
4. Server GPU matrix must be consistent (do not advertise Metal single-fit if endpoint rejects it).

