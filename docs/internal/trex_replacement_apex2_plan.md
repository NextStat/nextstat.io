# TRExFitter Replacement (NextStat) — Apex2 Plan

Methodology: **Apex2 = Planning → Exploration → Execution → Verification**.

NOTE: This doc is kept for history. The current active plan is:
- `docs/internal/trex-replacement-apex2-plan.md`

Goal: **full TRExFitter replacement** with:
- **identical numbers** (parity contracts + baselines),
- **publication-ready PDF/SVG** exports,
- **better UX** for configuration and analysis orchestration,
- **high-fidelity plots** (binning/labels/ratio/pulls/corr),
- robust **NTUP/HIST** workflows.

This doc is the living “what’s left?” and “what’s next?” plan.

## Planning (what / why / success criteria)

### Success criteria (replacement-ready)
1) **Config parity**: legacy TREx configs run with no manual rewrite (or importer gives actionable diffs).
2) **Numeric parity**: same workspace content, same test statistics and fits within declared tolerances.
3) **Artifact parity**: TREx-style report surfaces (pulls, corr, distributions, yields) export to PDF/SVG.
4) **Reproducibility**: deterministic mode works everywhere (`--threads 1`, stable outputs).

### Canonical validation sources
- **pyhf validation triad** (HistFactory XML import):
  - `xmlimport_input` (OverallSys + StatError + NormFactor)
  - `multichannel_histfactory` (ShapeSys)
  - `multichan_coupledhistosys` (coupled HistoSys)
- **Real-world TREx-like NTUP configs** (expression corpus):
  - FCCFitter configs extracted into `tests/fixtures/trex_expr_corpus/` (offline, testable).

## Exploration (what we already have in code + BMCP)

### Already implemented (high level)
- HistFactory XML/ROOT ingest + workspace build.
- ROOT-native NTUP/HIST pipeline.
- Region/sample override composition rules (Selection/Weight/Variable).
- Systematics preprocessing pipeline (smooth/symmetrize/prune).
- Reports + publication-ready PDF/SVG export surfaces.
- Parity contracts + baselines framework (incl. baseline schemas + record/compare tooling).

### Remaining “replacement gaps” (as of today)
1) **External baselines from realistic TREx export dirs** (needed to lock “identical numbers” vs real TREx outputs).
2) **Expression compatibility hardening** driven by real corpuses:
   - alias coverage (ROOT/TMath spellings),
   - vector branches / indexing variants,
   - any real `TTreeFormula`-specific constructs that appear in the wild.

## Execution (step-by-step implementation plan, TDD-first)

### Step A — Expression coverage reporting (NTUP)
Purpose: fast feedback on whether a legacy config’s `Selection/Weight/Variable` expressions compile and what branches they need.

TDD plan:
1) Unit tests: parse raw TREx config text; extract expression-bearing keys; compile with `ns-root` engine; capture required branches.
2) Derived expressions: ensure suffix-expansion systematics are also checked (`WeightBase + _up/_down`).
3) CLI wiring: add `--expr-coverage-json` to:
   - `nextstat import trex-config`
   - `nextstat build-hists`
4) Docs: document the flag and schema versioning.

Deliverables:
- JSON report schema `trex_expr_coverage_v0` with per-expression records.

### Step B — Expand expression corpus + close gaps
TDD plan:
1) Add failing corpus examples as tests (compile-only first).
2) Implement the minimal missing parsing/eval support in `crates/ns-root/src/expr.rs`.
3) Add integration test(s) that go through `ns-translate` ntuple histogram filling with a small ROOT fixture.

### Step C — External TREx export-dir baselines (postponed / external env)
We need **1–3 “realistic” TREx export dirs** (with `combination.xml` + referenced ROOT hists) to run the ROOT-suite baseline recorder in an environment with ROOT installed.

Inputs expected:
- export dir(s) + (optionally) a small JSON “cases” manifest (user-provided).

Process:
- run `record_baseline.py` following instructions in `README.md`
- check in resulting baseline artifacts (versioned schema)
- wire into CI gates (optional, may be nightly/manual)

## Verification (how we prove replacement readiness)

1) Unit tests pass:
   - expression corpus compiles
   - expression coverage report schema stable
2) End-to-end parity:
   - pyhf triad fixtures: workspace + fit stats + q(mu) comparisons
3) Artifact verification:
   - distributions/pulls/corr export to PDF/SVG with stable rendering
4) External baseline compare:
   - realistic TREx export dirs recorded + compared

## BMCP pointers
- Expression compat epic: `53b3d3fc-ef1f-42b2-b067-b4df90c1044e`
- Parity/baselines epic: `f4ead082-aa4a-49f0-8468-6201df649039`
  - External export-dir baselines task: `23711a70-dfb5-48e3-a96c-27abaa1f8fdc`
