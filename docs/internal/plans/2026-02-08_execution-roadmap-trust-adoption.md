<!--
Plan
Created: 2026-02-08
Owner: nextstat.io
Purpose: Execution roadmap for trust + adoption (benchmarks + tools + server + connectors + demos).
-->

# Execution Roadmap: Trust + Adoption (Feb 2026)

Date: 2026-02-08

This is a *shipping plan*, not a vision doc. It connects:
- what already exists in the repo,
- what is missing for adoption,
- what we will ship next,
- and how this maps into BMCP (epics + tasks).

## 1) Snapshot (What Exists Today)

Tooling + determinism:
- Tool API contract v1 is implemented for Python tools:
  - Envelope `schema_version=nextstat.tool_result.v1`
  - Deterministic execution knobs (`deterministic`, `eval_mode`, `threads`)
  - Seed policy for stochastic tools (toys)
  - JSON Schemas + golden regression harness
  - Docs: `docs/internal/plans/2026-02-08_tool-api-contract-v1.md`
- Tool API references:
  - `docs/references/tool-api.md`
  - `docs/references/python-api.md`

Trust artifacts:
- Validation Report pack (JSON + deterministic PDF) exists and is CI-gated under the “Validation Report” epic.

Benchmarks (methodology present, packaging not):
- Benchmark suite runbooks exist under `docs/benchmarks/suites/` (HEP/Bayesian/ML/Pharma/Econometrics/Time series).
- Public distribution (`nextstat-benchmarks` repo or equivalent) is not yet shipped end-to-end.

Server:
- A server exists: `crates/ns-server` (`nextstat-server`) with endpoints:
  - `POST /v1/fit`
  - `POST /v1/ranking`
  - `POST /v1/batch/fit`
  - `POST /v1/batch/toys`
  - `POST /v1/models` (upload/list/delete cached models)
  - `GET  /v1/health`
- Server does not yet expose hypotest/upper-limit/scan/discovery or tool-execution endpoints.

WASM:
- Browser/WASM playground exists via `bindings/ns-wasm` and `playground/` (the wedge for “in-browser tutorials”).

## 2) What From The Strategy List Is актуально (Feb 2026)

Trust offensive:
- актуально and still the top adoption blocker.
- Required framing: “spec vs speed”, reproducible runs, versioned artifacts.
- Note: “vs ROOT/RooFit” is only meaningful for *specific* comparable workflows; treat as optional, not as the headline.

AI for Science bridge:
- актуально only if treated as “tool contract + reproducible demo”, not “agent magic”.
- Prereq: stable tool semantics (already fixed in Python) and a server story (still missing).

Vertical expansion (Pharma/Econ):
- актуально, but blocked mostly by:
  - trust artifacts (validation/benchmarks that outsiders can rerun),
  - connectors (R for pharma/econ).

Documentation translation:
- актуально and the cheapest leverage to unlock non-HEP adoption.

Ecosystem connectors:
- актуально. Python exists; R bindings remain the critical missing piece.

Community building:
- useful later. Without public benchmark+validation artifacts, it risks becoming noise.

## 3) What We Add (Missing Workstreams)

1) “Server mode” for tools: same contract, same semantics, reproducible outputs.
2) Public benchmark distribution: pinned env, one-command run, DOI snapshots, `CITATION.cff`.
3) R integration: minimal “thin” binding surface for pharma/econ adoption.

## 4) Milestones (6-Week Execution)

### M1 (Weeks 1-2): Server Tool Runtime v1

Goal:
- Run the same tool calls in server mode and get `nextstat.tool_result.v1` responses.

Deliverables:
- `POST /v1/tools/execute` on `nextstat-server` returning tool envelope.
- `GET /v1/tools/schema` exposing tool registry and schema versions/refs.
- Determinism policy in server mode:
  - `deterministic=true` forces parity mode and requests `threads=1` where feasible.
  - Stochastic tools require explicit `seed` and echo it back.
- Hard limits:
  - request body limits already exist; add timeouts and ROOT input limits for any file ingestion endpoints.

Acceptance criteria:
- A deterministic tool call (same inputs, same seed) round-trips via server mode with stable output.
- CI has at least one server-mode golden tool test (normalized like local goldens).

BMCP mapping:
- Epic: `AI for Science Bridge: NextStat as LLM Oracle (Tools + Agent Demo)` (`c74becb9-afd7-4008-b791-62b5d6dcdb8d`)
- Tasks:
  - `ns-server: implement /v1/tools/execute (tool_result.v1 envelope)` (`4e236c06-554c-48ab-8e7d-10d9770798ba`)
  - `ns-server: add /v1/tools/schema (OpenAI tool defs + JSON Schema refs)` (`7ee4f46b-199e-4d89-8705-e3ddb3e7b6b1`)
  - `ns-server: implement hypotest/upper-limit/scan/discovery endpoints (or tool parity)` (`9da5e63f-d2dd-4c26-ac93-6f4ebc1a298f`)
  - `Python tools: add server transport (execute_tool via HTTP) + fallback to local` (`091995c5-8f16-4004-bd47-bb520bf0bd24`)
  - `Tests: server tool goldens + schema validation (match local tools)` (`71f5d102-4e93-4425-bb43-4b7a91201397`)
  - `Docs: nextstat-server API surface vs tool API (what exists, what doesn't)` (`e6f24259-75e3-4958-a7c5-8f0cefdc1919`)

### M2 (Weeks 2-3): Public Benchmarks Distribution (first publishable snapshot)

Goal:
- Outsiders can rerun at least one suite (HEP minimal + 1 other) with pinned env and get signed artifacts.

Deliverables:
- Choose distribution:
  - `nextstat-benchmarks` repo, or
  - `benchmarks/` folder in this repo with pinned lockfiles and runners.
- Produce “Snapshot v1” artifacts:
  - environment manifest
  - raw outputs
  - summary table JSON
  - report PDF (optional) driven by existing validation/benchmark artifacts
- Publish:
  - `CITATION.cff`
  - DOI (Zenodo) for immutable snapshots

Acceptance criteria:
- “fresh machine” run is one command, produces the same summary within declared tolerances.

BMCP mapping:
- Epic: `Trust Offensive: Public Benchmarks (nextstat-benchmarks)` (`ee9d3b45-e62d-49c2-8134-d2352c7378cc`)

### M3 (Weeks 3-4): Physics Assistant Demo (reproducible)

Goal:
- A demo that proves the “NextStat as Oracle” claim with real data ingestion and statistical answers.

Deliverables:
- Minimal agent demo:
  - `.root` histogram ingest
  - anomaly scan proposal
  - NextStat fit / discovery p-values / upper limits
  - plot export artifact (stable JSON + optional PNG)
- A single “runbook command” that produces:
  - tool call log
  - result JSON artifacts
  - plots

Acceptance criteria:
- Demo can run in deterministic mode end-to-end.
- Outputs validate against JSON schemas.

BMCP mapping:
- Epic: `AI for Science Bridge: NextStat as LLM Oracle (Tools + Agent Demo)` (`c74becb9-afd7-4008-b791-62b5d6dcdb8d`)
- Task: `Physics Assistant demo: .root -> anomaly scan -> p-values + plots` (`2b2eda6b-54df-4020-97d0-4b1f7d2384c9`)

### M4 (Weeks 4-6): Pharma Wedge (R + nlmixr2 parity harness)

Goal:
- Make NextStat usable in a pharma-native stack (R) and benchmark/validate against publicly reproducible baselines.

Deliverables:
- R bindings skeleton (tech choice and first exported functions):
  - minimal API for PK/NLME workflow data shapes
  - packaging + CI smoke
- Benchmark/validation parity:
  - runner for nlmixr2
  - optional Torsten runner or recorded reference outputs
- Browser tutorial:
  - “Population PK modeling in Rust/WASM” built on existing playground.

Acceptance criteria:
- R package builds on CI and can run at least one minimal PopPK example.
- Public benchmark runner produces comparable outputs and a reproducible manifest.

BMCP mapping:
- Epic: `Vertical Expansion: Pharma (PK/PD + NLME)` (`1a3ad381-eb69-46b3-8a66-8f93f751de33`)
- Epic: `Ecosystem Connectors: Python (sklearn), R bindings, Data Formats` (`18487268-19fa-4668-8c4d-e4a0007190d8`)

### M5 (Parallel): Documentation Translation (Personas + Glossary + Quickstarts)

Goal:
- Reduce cognitive friction for non-HEP users.

Deliverables:
- Glossary mapping table (HEP terms to DS/Quant/Bio)
- 3 persona quickstarts (minimal runnable examples)
- Doc navigation entry points (so users find the right surface fast)

Acceptance criteria:
- A new user can get to a working example in under 10 minutes with no HEP vocabulary.

BMCP mapping:
- Epic: `Documentation Translation: Personas + Glossary + Entry Points` (`797853dc-4e21-4c24-8458-ee8512acbcba`)

## 5) Key Constraints (Non-Negotiables)

- Determinism is a contract for trust artifacts: parity mode must be available and explicitly invokable.
- Tool semantics must be correct (CLs vs p-values, discovery definitions).
- Avoid one-off demo outputs. Everything produces versioned JSON artifacts and validates via schema.

## 6) Risks / Watch Items

- Server thread pool configuration is one-shot (Rayon). “threads=1” must be treated as best-effort and documented.
- ROOT ingestion is an attack surface. Keep strict size/time limits, avoid panics, and prefer “fail closed”.
- R bindings can become a time sink. Keep v0 scope minimal: a few core entrypoints plus examples.

## 7) References (Existing Docs)

- Market research note: `audit/2026-02-08_18-30_market-research.md`
- High-level trust/bridge/vertical plan: `docs/internal/plans/2026-02-08_trust-bridge-vertical-plan.md`
- Tool API contract: `docs/internal/plans/2026-02-08_tool-api-contract-v1.md`
- Benchmark suite runbooks: `docs/benchmarks/suites/`

