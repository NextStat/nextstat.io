<!--
Reference
Created: 2026-02-08
Updated: 2026-02-24
Scope: nextstat.tools contract usage
-->

# Tool API (nextstat.tools)

This is the user-facing reference for NextStat's **agent/tool-calling** surface.

Determinism semantics: see `tests/python/_tolerances.py` (7-tier tolerance hierarchy).
Tool result schema: `docs/schemas/tools/nextstat_tool_result_v1.schema.json`

## What You Get

1. `nextstat.tools.get_toolkit()` returns OpenAI-compatible tool definitions (JSON Schema input).
2. `nextstat.tools.get_toolkit_descriptor()` returns the full machine-readable discovery descriptor (`tools` + `capabilities`).
3. `nextstat.tools.execute_tool(name, arguments)` executes a tool call and returns a stable envelope.

## Transport Modes

`nextstat.tools` currently exposes **two different capability surfaces** depending on transport:

- `transport="local"`: the in-process Python registry. This is the broadest surface.
- `transport="server"`: the `nextstat-server` registry fetched from `GET /v1/tools/schema`. This is a server-safe subset and is intentionally narrower.

Do not assume that server mode exposes the full local Python tool registry.

Most JSON-only stable Python helpers that satisfy the server-safe policy are
also promoted into `nextstat.tools`. Local-only capabilities remain explicit,
but file-ingest helpers can graduate when they use a bounded server-safe
contract such as uploaded in-memory bytes instead of arbitrary server paths.

For ads variance reduction, the canonical tool names are:

- `nextstat_ads_cuped_adjust`
- `nextstat_ads_cure_adjust`

Canonical manifest: `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`
Manifest schema: `docs/schemas/tools/nextstat_tool_manifest_v1.schema.json`
Tool discovery schema: `docs/schemas/tools/nextstat_tool_schema_v1.schema.json`
Manifest validator: `scripts/validate_tool_manifest.py`
Descriptor validator: `scripts/validate_tool_schema_descriptor.py --transport local`
  - Server mode: `scripts/validate_tool_schema_descriptor.py --transport server --server-url ... --api-key ...`
Descriptor examples: `docs/specs/nextstat_tool_schema_local_v1.example.json`, `docs/specs/nextstat_tool_schema_server_v1.example.json`
Descriptor example generator: `scripts/generate_tool_schema_examples.py`
Server policy defaults/overrides live under `manifest.policies.server`.
Reference doc sync/check: `scripts/generate_tool_reference_docs.py`
Golden drift check: `scripts/generate_tool_goldens.py --check`
Runner report schema: `docs/schemas/tools/nextstat_tool_contract_runner_report_v1.schema.json`
Dashboard generator: `scripts/summarize_tool_contract_reports.py`
Dashboard schema: `docs/schemas/tools/nextstat_tool_contract_dashboard_v1.schema.json`
Artifact manifest: `scripts/tool_contract_artifact_manifest_v1.json`
Artifact manifest schema: `docs/schemas/tools/nextstat_tool_contract_artifact_manifest_v1.schema.json`
Artifact manifest helper: `scripts/tool_contract_artifact_manifest.py`
Performance budget manifest: `scripts/tool_contract_performance_budget_v1.json`
Performance budget schema: `docs/schemas/tools/nextstat_tool_contract_performance_budget_v1.schema.json`
Performance budget helper: `scripts/tool_contract_performance_budget.py`
Agent bootstrap packs: `docs/references/agent-bootstrap.md`
Canonical runner:
  - Fast lane: `./.venv/bin/python scripts/check_tool_contracts.py --mode fast`
  - Live lane: `./.venv/bin/python scripts/check_tool_contracts.py --mode live`
  - End-to-end: `./.venv/bin/python scripts/check_tool_contracts.py --mode all`
  - Fast lane auto-syncs `bindings/ns-py` into the active Python environment via `python -m maturin develop -m bindings/ns-py/Cargo.toml` before the Python contract suite
  - JSON report: add `--report-json tmp/reports/tool_contracts_fast_report.json`
  - Dashboard: `./.venv/bin/python scripts/summarize_tool_contract_reports.py --report tmp/reports/tool_contracts_fast_report.json --out-json tmp/reports/tool_contract_dashboard.json --out-md tmp/reports/tool_contract_dashboard.md`
  - Synthetic step `Validate tool-contract performance budgets` enforces runner duration budgets and, in live mode, live-server metrics budgets
  - Uses isolated cargo target dir by default: `.nextstat-cargo-target/tool-contracts` (override via `NEXTSTAT_TOOL_CONTRACT_CARGO_TARGET_DIR`)
  - Bindings sync uses its own isolated cargo target dir by default: `.nextstat-cargo-target/tool-contracts-bindings` (override via `NEXTSTAT_TOOL_CONTRACT_BINDINGS_CARGO_TARGET_DIR`)
  - CI dashboard job is configured to run even when upstream `tool-contracts` or `live-server-contracts` jobs fail, so failure summaries are not lost
  - Artifact names/paths between producer jobs and dashboard consumer now come from the canonical artifact manifest and are regression-tested in `tests/python/test_tool_contract_workflow.py`

## Discovery Descriptor

`get_toolkit_descriptor()` returns a versioned discovery envelope:

```json
{
  "schema_version": "nextstat.tool_schema.v1",
  "transport": "local",
  "tools": [],
  "capabilities": [],
  "guidance": {
    "hints": [],
    "recipes": []
  }
}
```

Rules:
- `tools` is the callable subset for the selected transport.
- `capabilities` is the broader discovery map with `local_available`, `server_available`, and `server_policy`.
- `guidance` is the transport-aware prompt/bootstrap layer for agents and IDE assistants (`hints` + curated workflow recipes).
- `get_toolkit()` is the compatibility helper that returns only `descriptor["tools"]`.

## Available Local Python Tools

Simplified-likelihood support class in March 2026:

- `stable`: `nextstat_workspace_audit`, `nextstat_fit`, `nextstat_hypotest`, `nextstat_upper_limit`, `nextstat_scan`
- `research-grade`: `nextstat_discovery_asymptotic`, `nextstat_ranking`, `nextstat_hypotest_toys` for simplified-likelihood inputs
- companion docs: `docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md`, `docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md`

<!-- BEGIN GENERATED TOOL CAPABILITY MATRIX -->
Generated from `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`.

Local tools: 45. Server-safe subset: 45.

Server policy codes:
- `server_safe_subset`: Exposed on the restricted nextstat-server tool surface.

| Tool | Server | Policy | Summary |
|------|:------:|--------|---------|
| `nextstat_fit` | Yes | `server_safe_subset` | Run Maximum Likelihood Estimation (MLE) on a HistFactory statistical model; simplified-likelihood input is inside the promoted March 2026 stable subset. |
| `nextstat_hypotest` | Yes | `server_safe_subset` | Run an asymptotic CLs hypothesis test at a given signal strength mu (qtilde); simplified-likelihood input is inside the promoted March 2026 stable subset. |
| `nextstat_hypotest_toys` | Yes | `server_safe_subset` | Run a toy-based CLs hypothesis test at a given signal strength mu (qtilde); simplified-likelihood input is compatibility-tested but outside the promoted March 2026 stable subset. |
| `nextstat_upper_limit` | Yes | `server_safe_subset` | Compute the 95% CL upper limit on signal strength via CLs; simplified-likelihood input is inside the promoted March 2026 stable subset. |
| `nextstat_ranking` | Yes | `server_safe_subset` | Compute nuisance parameter ranking (systematic impact on signal strength); simplified-likelihood input is compatibility-tested but outside the promoted March 2026 stable subset. |
| `nextstat_discovery_asymptotic` | Yes | `server_safe_subset` | Compute an asymptotic discovery-style statistic at mu=0 from a profiled likelihood scan; simplified-likelihood input is compatibility-tested but outside the promoted March 2026 stable subset. |
| `nextstat_scan` | Yes | `server_safe_subset` | Run a profile likelihood scan over signal strength values; simplified-likelihood input is inside the promoted March 2026 stable subset. |
| `nextstat_workspace_audit` | Yes | `server_safe_subset` | Audit a pyhf or simplified-likelihood workspace; simplified-likelihood audit is inside the promoted March 2026 stable subset. |
| `nextstat_read_root_histogram` | Yes | `server_safe_subset` | Read a TH1 histogram from a ROOT file, including sumw2 and under/overflow bins. |
| `nextstat_glm_fit` | Yes | `server_safe_subset` | Fit a Generalized Linear Model (GLM). |
| `nextstat_bayesian_sample` | Yes | `server_safe_subset` | Run Bayesian NUTS sampling on supported NextStat models and return a bounded summary: parameter names, convergence diagnostics, and posterior mean summaries rather than raw posterior draws. |
| `nextstat_survival_fit` | Yes | `server_safe_subset` | Fit a survival model via MLE. |
| `nextstat_kaplan_meier` | Yes | `server_safe_subset` | Compute the Kaplan-Meier survival curve and optionally a log-rank test. |
| `nextstat_log_rank_test` | Yes | `server_safe_subset` | Run a direct log-rank (Mantel-Cox) test across two or more survival groups. |
| `nextstat_panel_fe` | Yes | `server_safe_subset` | Fit a panel data model with entity fixed effects (within estimator). |
| `nextstat_did` | Yes | `server_safe_subset` | Estimate a Difference-in-Differences (DiD) model via two-way fixed effects (TWFE). |
| `nextstat_iv_2sls` | Yes | `server_safe_subset` | Estimate an Instrumental Variables (IV) model via Two-Stage Least Squares (2SLS). |
| `nextstat_aipw` | Yes | `server_safe_subset` | Estimate a doubly-robust Average Treatment Effect (ATE or ATT) via Augmented Inverse Probability Weighting (AIPW). |
| `nextstat_meta_analysis` | Yes | `server_safe_subset` | Run a meta-analysis (fixed or random effects). |
| `nextstat_kalman` | Yes | `server_safe_subset` | Run Kalman filtering, smoothing, forecasting, simulation, or bounded EM re-estimation on a linear state-space model. |
| `nextstat_ads_cuped_adjust` | Yes | `server_safe_subset` | Apply one-covariate CUPED adjustment as the one-covariate case of the shared CURE layer. |
| `nextstat_ads_cure_adjust` | Yes | `server_safe_subset` | Apply multivariate CURE adjustment with SVD/ridge collinearity guardrails. |
| `nextstat_churn_generate_data` | Yes | `server_safe_subset` | Generate a seeded synthetic churn dataset with customer-level times, events, groups, treatments, covariates, and segment metadata. |
| `nextstat_churn_risk_model` | Yes | `server_safe_subset` | Fit a Cox PH churn risk model and return log-hazard coefficients, standard errors, hazard ratios, and confidence intervals for the requested covariates. |
| `nextstat_churn_retention` | Yes | `server_safe_subset` | Compute churn retention using stratified Kaplan-Meier summaries and a log-rank comparison. |
| `nextstat_churn_compare` | Yes | `server_safe_subset` | Compare churn segments with pairwise log-rank tests, hazard-ratio proxies, and multiple-comparisons correction. |
| `nextstat_churn_diagnostics` | Yes | `server_safe_subset` | Run churn-analysis trust-gate diagnostics: censoring rates by segment, covariate balance, optional propensity-overlap checks, and warning summaries for downstream retention or uplift analysis. |
| `nextstat_churn_cohort_matrix` | Yes | `server_safe_subset` | Build a churn cohort life-table matrix. |
| `nextstat_churn_bootstrap_hr` | Yes | `server_safe_subset` | Bootstrap hazard ratios from a churn Cox PH model. |
| `nextstat_churn_ingest` | Yes | `server_safe_subset` | Validate and ingest raw churn arrays into a clean bounded dataset. |
| `nextstat_churn_uplift` | Yes | `server_safe_subset` | Estimate bounded churn uplift with the promoted AIPW workflow: treatment effect on churn within a time horizon, standard error, confidence interval, and sensitivity summary. |
| `nextstat_churn_uplift_survival` | Yes | `server_safe_subset` | Estimate survival-native churn uplift with RMST, IPW-weighted survival differences at specified horizons, and overlap diagnostics. |
| `nextstat_chain_ladder` | Yes | `server_safe_subset` | Run chain ladder or Mack chain ladder reserving on a cumulative claims triangle. |
| `nextstat_pharma_fit` | Yes | `server_safe_subset` | Fit a nonlinear mixed-effects (NLME) population PK model using FOCE, FOCEI, FO, or SAEM. |
| `nextstat_pharma_vpc` | Yes | `server_safe_subset` | Run a Visual Predictive Check (VPC) for a fitted population PK model. |
| `nextstat_pk_gof` | Yes | `server_safe_subset` | Run per-observation population PK goodness-of-fit diagnostics. |
| `nextstat_pk_npde` | Yes | `server_safe_subset` | Run normalized prediction distribution error (NPDE) diagnostics for a population PK model. |
| `nextstat_trial_simulate` | Yes | `server_safe_subset` | Simulate a single clinical PK trial: generate concentration profiles for n_subjects with inter-individual variability and return concentrations, individual parameters, AUC, Cmax, Tmax, and Ctrough for each subject. |
| `nextstat_bioequivalence` | Yes | `server_safe_subset` | Run average bioequivalence (ABE) analysis on a 2x2 crossover study. |
| `nextstat_fault_tree_mc` | Yes | `server_safe_subset` | Run Monte Carlo simulation on a fault tree (FTA). |
| `nextstat_fault_tree_ce_is` | Yes | `server_safe_subset` | Run Cross-Entropy Importance Sampling (CE-IS) on a fault tree for rare-event probability estimation. |
| `nextstat_dose_response` | Yes | `server_safe_subset` | Evaluate an Emax or Sigmoid-Emax dose-response model. |
| `nextstat_competing_risks` | Yes | `server_safe_subset` | Analyze competing risks data: compute cumulative incidence functions (CIF), Gray's test for group comparison, or Fine-Gray subdistribution hazard regression. |
| `nextstat_event_study` | Yes | `server_safe_subset` | Run an event study (dynamic DiD) with leads and lags around treatment. |
| `nextstat_garch_fit` | Yes | `server_safe_subset` | Fit a GARCH(1,1), EGARCH(1,1), or GJR-GARCH(1,1) volatility model to a return series. |
<!-- END GENERATED TOOL CAPABILITY MATRIX -->

### Envelope format

```json
{
  "schema_version": "nextstat.tool_result.v1",
  "ok": true,
  "result": {},
  "error": null,
  "meta": {
    "tool_name": "nextstat_fit",
    "nextstat_version": "0.9.7",
    "deterministic": true,
    "eval_mode": "parity",
    "threads_requested": 1
  }
}
```

Schema files:
- `docs/schemas/tools/nextstat_tool_result_v1.schema.json` (envelope)
- `docs/schemas/tools/nextstat_tool_result_strict_v1.schema.json` (envelope + strict result shapes)
- `docs/schemas/tools/nextstat_tool_result_server_strict_v1.schema.json` (server-safe strict subset)
- Sync/check generator: `scripts/generate_tool_contract_schemas.py`

Notes:
- `nextstat_tool_result_v1` is the canonical transport-agnostic envelope.
- `nextstat_tool_result_strict_v1` is a strict validation helper for current core/common tool payloads; it is **not** an exhaustive discovery registry across transports.

## Execution Controls

Most tools accept an `execution` object:

```json
{
  "execution": {
    "deterministic": true,
    "eval_mode": "parity",
    "threads": 1
  }
}
```

Rules:
- `deterministic=true` forces `eval_mode="parity"` and requests `threads=1`.
- Thread control is request-scoped on the server when possible. Check `meta.threads_applied` when present.

## Semantics (Avoid Trust-Killers)

- `nextstat_hypotest` returns **CLs** (+ `clsb`, `clb`). It is not a p-value.
- Discovery-style outputs use `nextstat_discovery_asymptotic` which returns `{q0, z0, p0}`.

## Toy Tools and Seeds

Stochastic tools require a `seed` (default 42, but agents should set explicitly):
- `nextstat_hypotest_toys`: returns toy-based CLs as a `"raw"` payload plus the seed metadata.
- `nextstat_pharma_vpc`: simulates n_sim replicate datasets for VPC.
- `nextstat_trial_simulate`: simulates virtual PK trial with IIV.
- `nextstat_fault_tree_mc`: Monte Carlo scenarios for fault tree.
- `nextstat_fault_tree_ce_is`: CE-IS rare event estimation.

## Example: Tool Calling (OpenAI)

```python
import json
from openai import OpenAI
from nextstat.tools import get_toolkit, execute_tool

client = OpenAI()
tools = get_toolkit()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "Fit the workspace and report the POI."}],
    tools=tools,
)

for call in resp.choices[0].message.tool_calls:
    out = execute_tool(call.function.name, json.loads(call.function.arguments))
    if not out["ok"]:
        raise RuntimeError(out["error"])
    print(out["result"])
```

## Agent and IDE Guidance (Local)

<!-- BEGIN GENERATED LOCAL GUIDANCE RECIPES -->
Generated from `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`.

Transport hints:
- Use local transport when the agent or IDE assistant needs the full Python tool registry, including ROOT ingest and non-HEP verticals.
- Call get_toolkit_descriptor(transport="local") first and treat descriptor.tools as the callable contract instead of hard-coding tool names.
- For reproducible reviews, parity checks, and IDE quick-fixes, set execution.deterministic=true explicitly.

### `local_hep_workspace_inference` — Local HEP workspace inference
Summary: Audit a workspace, fit it, and produce CLs, upper-limit, and scan outputs in-process. For simplified-likelihood inputs, discovery, ranking, and toy CLs remain compatibility-tested rather than promoted stable.
Tools: `nextstat_workspace_audit`, `nextstat_fit`, `nextstat_discovery_asymptotic`, `nextstat_hypotest`, `nextstat_upper_limit`, `nextstat_scan`, `nextstat_ranking`, `nextstat_hypotest_toys`
Starter prompt: "Audit this workspace, then fit it and report the POI value and uncertainty, a short asymptotic CLs summary, observed and expected 95% CL upper limit, and a short profile scan summary. If the workspace is pyhf or HS3, also include discovery significance and top nuisance impacts by default. If it is simplified-likelihood, treat audit + fit + asymptotic CLs/upper-limit + scan as the promoted stable subset and use discovery, ranking, or toy CLs only when the user explicitly asks for the compatibility-tested surface. Use deterministic mode."
Docs: `docs/references/tool-api.md`, `docs/references/python-api.md`, `docs/tutorials/e2e-pipeline.md`

### `local_root_ingest_and_workspace_triage` — Local ROOT ingest and workspace triage
Summary: Read TH1 histograms locally, then use the regular HEP inference tools on the derived workspace.
Tools: `nextstat_read_root_histogram`, `nextstat_workspace_audit`, `nextstat_fit`, `nextstat_scan`
Starter prompt: "Read the requested ROOT histograms locally, summarize suspicious bins and under or overflow content, then continue with deterministic workspace validation and fit checks."
Docs: `docs/references/tool-api.md`, `docs/demos/physics-assistant.md`, `docs/references/python-api.md`

### `local_bayesian_and_glm_analysis` — Local Bayesian and GLM analysis
Summary: Use the local Python surface for generalized linear models and Bayesian posterior sampling.
Tools: `nextstat_glm_fit`, `nextstat_bayesian_sample`
Starter prompt: "Fit the requested generalized linear model, then if posterior uncertainty is important run Bayesian sampling locally and summarize posterior means plus convergence diagnostics rather than dumping raw chains by default."
Docs: `docs/references/tool-api.md`, `docs/references/python-api.md`, `docs/tutorials/phase-6-regression.md`

### `local_survival_and_risk_analysis` — Local survival and risk analysis
Summary: Run survival, Kaplan-Meier, direct log-rank, competing-risks, synthetic churn generation, churn-risk modeling, ingest, bootstrap HR, diagnostics, cohort-matrix, and retention/comparison workflows locally through the Python tool registry.
Tools: `nextstat_survival_fit`, `nextstat_kaplan_meier`, `nextstat_log_rank_test`, `nextstat_competing_risks`, `nextstat_churn_generate_data`, `nextstat_churn_risk_model`, `nextstat_churn_retention`, `nextstat_churn_bootstrap_hr`, `nextstat_churn_ingest`, `nextstat_churn_diagnostics`, `nextstat_churn_cohort_matrix`, `nextstat_churn_compare`, `nextstat_churn_uplift`, `nextstat_churn_uplift_survival`
Starter prompt: "Fit the requested survival or competing-risks model, include Kaplan-Meier or direct log-rank summaries when appropriate, and report synthetic or validated churn datasets, Cox PH churn-risk outputs, bootstrap hazard-ratio uncertainty summaries, retention curves, cohort life-table matrices, trust-gate diagnostics, pairwise churn-comparison outputs, bounded uplift estimates, or survival-native uplift summaries with deterministic settings."
Docs: `docs/references/tool-api.md`, `docs/tutorials/phase-9-survival.md`, `docs/tutorials/churn-subscription.md`

### `local_ads_variance_reduction` — Local ads variance reduction
Summary: Run deterministic CUPED or CURE adjustment locally through the ads-native Python helpers with provenance-aware leakage validation.
Tools: `nextstat_ads_cuped_adjust`, `nextstat_ads_cure_adjust`
Starter prompt: "Run the requested ads variance-reduction adjustment deterministically, keep covariates strictly pre-treatment, and report adjusted means, treatment effect, variance-reduction diagnostics, selected covariates, provenance validation, and solver metadata."
Docs: `docs/references/tool-api.md`, `docs/references/python-api.md`, `docs/benchmarks/ads-variance-reduction-benchmark-2026-03-08.md`

### `local_econometrics_and_time_series` — Local econometrics and time-series workflows
Summary: Use the local surface for panel, DiD, IV, doubly robust causal inference, event studies, volatility, and Kalman workflows.
Tools: `nextstat_panel_fe`, `nextstat_did`, `nextstat_iv_2sls`, `nextstat_aipw`, `nextstat_event_study`, `nextstat_garch_fit`, `nextstat_kalman`
Starter prompt: "Pick the right econometric tool for the identification strategy, run the model deterministically, and summarize treatment effects, event-study dynamics, volatility behavior, or state-space forecasts as appropriate."
Docs: `docs/references/tool-api.md`, `docs/references/econometrics.md`, `docs/tutorials/phase-12-did-event-study.md`

### `local_meta_insurance_and_reliability` — Local meta-analysis, reserving, and reliability
Summary: Cover meta-analysis, chain-ladder reserving, and reliability or fault-tree Monte Carlo workflows locally.
Tools: `nextstat_meta_analysis`, `nextstat_chain_ladder`, `nextstat_fault_tree_mc`, `nextstat_fault_tree_ce_is`
Starter prompt: "Run the requested meta-analysis, reserving, or fault-tree simulation locally and return the key pooled estimate, reserve view, or rare-event risk metrics with deterministic settings where applicable."
Docs: `docs/references/tool-api.md`, `docs/references/python-api.md`

### `local_pharma_and_dose_response` — Local pharma, trial simulation, and dose response
Summary: Fit PopPK models, run GOF, NPDE, or VPC diagnostics and trial simulations, evaluate bioequivalence, and work with dose-response models locally.
Tools: `nextstat_pharma_fit`, `nextstat_pk_gof`, `nextstat_pk_npde`, `nextstat_pharma_vpc`, `nextstat_trial_simulate`, `nextstat_bioequivalence`, `nextstat_dose_response`
Starter prompt: "Fit the requested population PK or dose-response workflow locally, run the follow-up GOF, NPDE, VPC, simulation, or bioequivalence step when relevant, and summarize the main parameter estimates and diagnostics."
Docs: `docs/references/tool-api.md`, `docs/references/python-api.md`, `docs/tutorials/pharma-pk.md`
<!-- END GENERATED LOCAL GUIDANCE RECIPES -->

## Regression Harness

- Golden outputs: `tests/fixtures/tool_goldens/*.v1.json`
- Generator: `scripts/generate_tool_goldens.py`
  - Runs against the in-repo Python tool surface layered onto the installed core package.
  - Resolves tool calls from the canonical manifest instead of a hand-maintained call list.
  - Supports `--check` for CI/contract enforcement.
- Smoke runner: `scripts/tool_call_smoke.py`
- Canonical contract runner: `scripts/check_tool_contracts.py`
  - JSON audit artifact: `--report-json tmp/reports/tool_contracts_fast_report.json`
- Contract dashboard: `scripts/summarize_tool_contract_reports.py`
  - JSON + Markdown artifacts: `tmp/reports/tool_contract_dashboard.json`, `tmp/reports/tool_contract_dashboard.md`
  - Failure drilldown includes failed step command plus compact `stdout/stderr` tails when a step fails
  - Failure classes: `none`, `schema_drift`, `performance_budget_failure`, `rust_contract_failure`, `python_contract_failure`, `live_server_failure`, `unknown`
  - Runner report and dashboard now include a machine-readable `performance` block with runner budgets and live metrics budget status

## Server Mode (nextstat-server)

If you run `nextstat-server`, you can also fetch tools and execute them over HTTP:
- Tool registry: `GET /v1/tools/schema`
- Tool execution: `POST /v1/tools/execute`

See: `docs/references/server-api.md`

Current server-safe subset:
<!-- BEGIN GENERATED SERVER TOOL SUBSET -->
Generated from `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`.

Server-safe tools: 45.
Policy code: `server_safe_subset`.

| Tool | Summary |
|------|---------|
| `nextstat_fit` | Run Maximum Likelihood Estimation (MLE) on a HistFactory model; simplified-likelihood input is inside the promoted March 2026 stable subset. |
| `nextstat_hypotest` | Run an asymptotic CLs hypothesis test at a given mu (qtilde); simplified-likelihood input is inside the promoted March 2026 stable subset. |
| `nextstat_hypotest_toys` | Toy-based CLs hypotest at mu (qtilde); simplified-likelihood input is compatibility-tested but outside the promoted March 2026 stable subset. |
| `nextstat_upper_limit` | Compute the 95% CL upper limit on mu via asymptotic CLs (qtilde); simplified-likelihood input is inside the promoted March 2026 stable subset. |
| `nextstat_ranking` | Compute nuisance parameter ranking (impact on POI); simplified-likelihood input is compatibility-tested but outside the promoted March 2026 stable subset. |
| `nextstat_discovery_asymptotic` | Compute asymptotic discovery statistics at mu=0: q0, z0, p0 (one-sided); simplified-likelihood input is compatibility-tested but outside the promoted March 2026 stable subset. |
| `nextstat_scan` | Run a profile likelihood scan over signal strength values; simplified-likelihood input is inside the promoted March 2026 stable subset and the result returns points for plotting q(mu). |
| `nextstat_workspace_audit` | Audit a pyhf or simplified-likelihood workspace JSON; simplified-likelihood audit is inside the promoted March 2026 stable subset. |
| `nextstat_read_root_histogram` | Read a TH1 histogram from uploaded ROOT file bytes entirely in memory on nextstat-server, including sumw2 and under/overflow bins. |
| `nextstat_glm_fit` | Fit a Generalized Linear Model (GLM). |
| `nextstat_bayesian_sample` | Run bounded Bayesian NUTS sampling on the authenticated server-safe model subset. |
| `nextstat_survival_fit` | Fit a survival model via MLE. |
| `nextstat_kaplan_meier` | Compute the Kaplan-Meier survival curve and optionally a log-rank test. |
| `nextstat_log_rank_test` | Run a bounded direct log-rank (Mantel-Cox) test remotely across two or more survival groups. |
| `nextstat_panel_fe` | Fit a panel data model with entity fixed effects (within estimator). |
| `nextstat_did` | Estimate a Difference-in-Differences (DiD) model via two-way fixed effects (TWFE). |
| `nextstat_iv_2sls` | Estimate an Instrumental Variables (IV) model via Two-Stage Least Squares (2SLS). |
| `nextstat_aipw` | Estimate a doubly-robust Average Treatment Effect (ATE or ATT) via Augmented Inverse Probability Weighting (AIPW). |
| `nextstat_meta_analysis` | Run a meta-analysis (fixed or random effects). |
| `nextstat_kalman` | Run Kalman filtering, smoothing, forecasting, deterministic simulation, or bounded EM re-estimation on a linear state-space model over the authenticated server-safe subset. |
| `nextstat_ads_cuped_adjust` | Apply one-covariate CUPED adjustment as the one-covariate case of the shared CURE layer over the authenticated server-safe subset. |
| `nextstat_ads_cure_adjust` | Apply multivariate CURE adjustment with SVD/ridge collinearity guardrails over the authenticated server-safe subset. |
| `nextstat_churn_generate_data` | Generate a seeded synthetic churn dataset over the authenticated server-safe subset. |
| `nextstat_churn_risk_model` | Fit a bounded Cox PH churn risk model remotely and return coefficients, standard errors, hazard ratios, and confidence intervals for the requested covariates. |
| `nextstat_churn_retention` | Compute churn retention using stratified Kaplan-Meier summaries and a log-rank comparison. |
| `nextstat_churn_compare` | Compare churn segments with pairwise log-rank tests, hazard-ratio proxies, and multiple-comparisons correction over the authenticated server-safe subset. |
| `nextstat_churn_diagnostics` | Run bounded churn-analysis trust-gate diagnostics over the authenticated server-safe subset: censoring rates by segment, covariate balance, optional propensity-overlap checks, and warning summaries for downstream retention or uplift analysis. |
| `nextstat_churn_cohort_matrix` | Build a churn cohort life-table matrix over the authenticated server-safe subset. |
| `nextstat_churn_bootstrap_hr` | Bootstrap hazard ratios over the authenticated server-safe subset. |
| `nextstat_churn_ingest` | Validate and ingest raw churn arrays over the authenticated server-safe subset. |
| `nextstat_churn_uplift` | Estimate bounded churn uplift over the authenticated server-safe subset: treatment effect on churn within a time horizon, confidence interval, and sensitivity summary. |
| `nextstat_churn_uplift_survival` | Estimate survival-native churn uplift over the authenticated server-safe subset with RMST, survival differences, and overlap diagnostics. |
| `nextstat_chain_ladder` | Run chain ladder or Mack chain ladder reserving on a cumulative claims triangle. |
| `nextstat_pharma_fit` | Fit a nonlinear mixed-effects (NLME) population PK model using FOCE, FOCEI, FO, or SAEM. |
| `nextstat_pharma_vpc` | Run a bounded Visual Predictive Check (VPC) for a fitted population PK model through nextstat-server. |
| `nextstat_pk_gof` | Run bounded population PK goodness-of-fit diagnostics through nextstat-server using deterministic JSON-only inputs. |
| `nextstat_pk_npde` | Run bounded NPDE diagnostics for a population PK model through nextstat-server using deterministic JSON-only inputs and an explicit seed. |
| `nextstat_trial_simulate` | Simulate a bounded single clinical PK trial through nextstat-server using deterministic JSON-only inputs. |
| `nextstat_bioequivalence` | Run average bioequivalence (ABE) analysis on a 2x2 crossover study. |
| `nextstat_fault_tree_mc` | Run bounded CPU-only Monte Carlo simulation on a fault tree (FTA) remotely. |
| `nextstat_fault_tree_ce_is` | Run bounded CPU-only Cross-Entropy Importance Sampling (CE-IS) on a fault tree remotely for rare-event probability estimation. |
| `nextstat_dose_response` | Evaluate an Emax or Sigmoid-Emax dose-response model. |
| `nextstat_competing_risks` | Analyze competing risks data remotely: compute cumulative incidence functions (CIF), Gray's test for group comparison, or Fine-Gray subdistribution hazard regression. |
| `nextstat_event_study` | Run an event study (dynamic DiD) with leads and lags around treatment. |
| `nextstat_garch_fit` | Fit a GARCH(1,1), EGARCH(1,1), or GJR-GARCH(1,1) volatility model to a return series. |
<!-- END GENERATED SERVER TOOL SUBSET -->

## Agent and IDE Guidance (Server)

<!-- BEGIN GENERATED SERVER GUIDANCE RECIPES -->
Generated from `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`.

Transport hints:
- Use server transport when the agent needs remote, auth-enabled execution through nextstat-server.
- Treat descriptor.tools as the only callable server surface and use descriptor.capabilities to explain why local-only tools are unavailable remotely.
- When server auth is enabled, pass api_key explicitly or provide NEXTSTAT_TOOLS_API_KEY / NEXTSTAT_SERVER_API_KEY.

### `server_root_ingest_and_workspace_triage` — Server-safe ROOT ingest and workspace triage
Summary: Upload bounded ROOT bytes to nextstat-server, read TH1 histograms remotely in memory, then continue with the regular HEP inference tools on the derived workspace.
Tools: `nextstat_read_root_histogram`, `nextstat_workspace_audit`, `nextstat_fit`, `nextstat_scan`
Starter prompt: "Over the remote server, upload the requested ROOT file bytes, read the requested TH1 histogram in memory without assuming server-side filesystem access, summarize suspicious bins and under or overflow content, then continue with deterministic workspace validation and fit checks."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/references/python-api.md`

### `server_workspace_inference` — Server-safe workspace inference
Summary: Remote, auth-aware HEP inference chain over nextstat-server using the restricted server-safe subset. For simplified-likelihood inputs, discovery, ranking, and toy CLs remain compatibility-tested rather than promoted stable.
Tools: `nextstat_workspace_audit`, `nextstat_fit`, `nextstat_discovery_asymptotic`, `nextstat_hypotest`, `nextstat_upper_limit`, `nextstat_scan`, `nextstat_ranking`, `nextstat_hypotest_toys`
Starter prompt: "Over the remote server, audit this workspace, fit it, and report a short asymptotic CLs summary, observed and expected 95% CL upper limit, and a short profile scan summary. If the workspace is pyhf or HS3, also include discovery significance and top nuisance impacts by default. If it is simplified-likelihood, treat audit + fit + asymptotic CLs/upper-limit + scan as the promoted stable subset and use discovery, ranking, or toy CLs only when the user explicitly asks for the compatibility-tested surface. Use deterministic mode and do not assume local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/e2e-pipeline.md`

### `server_glm_analysis` — Server-safe GLM analysis
Summary: Run bounded generalized linear model fits remotely through nextstat-server without relying on local-only Python tooling.
Tools: `nextstat_glm_fit`
Starter prompt: "Over the remote server, fit the requested GLM deterministically, report the coefficient estimates and standard errors, and keep the analysis on the server-safe subset without assuming local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/phase-6-regression.md`

### `server_bayesian_sampling_summary` — Server-safe Bayesian summary sampling
Summary: Run bounded NUTS sampling remotely over the promoted server-safe model subset and return posterior mean summaries plus convergence diagnostics.
Tools: `nextstat_bayesian_sample`
Starter prompt: "Over the remote server, run bounded Bayesian NUTS sampling on the requested server-safe model, keep the request inside the authenticated sample-budget limits, and report posterior means plus convergence diagnostics rather than raw posterior draws."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/references/python-api.md`

### `server_survival_analysis` — Server-safe survival analysis
Summary: Run bounded survival-model, Kaplan-Meier, direct log-rank, and competing-risks analyses remotely through nextstat-server without relying on local-only Python tooling.
Tools: `nextstat_survival_fit`, `nextstat_kaplan_meier`, `nextstat_log_rank_test`, `nextstat_competing_risks`
Starter prompt: "Over the remote server, fit the requested survival, Kaplan-Meier, direct log-rank, or competing-risks analysis deterministically, report fitted parameters or incidence-curve outputs with any log-rank or Gray-test comparison when requested, and stay within the authenticated server-safe subset."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/phase-9-survival.md`

### `server_retention_analysis` — Server-safe churn retention analysis
Summary: Run bounded synthetic churn generation, churn-risk modeling, churn-retention, churn ingest, bootstrap HR, churn diagnostics, cohort-matrix, churn-comparison, and churn-uplift analyses remotely through nextstat-server using deterministic JSON-only inputs and authenticated server execution.
Tools: `nextstat_churn_generate_data`, `nextstat_churn_risk_model`, `nextstat_churn_retention`, `nextstat_churn_bootstrap_hr`, `nextstat_churn_ingest`, `nextstat_churn_diagnostics`, `nextstat_churn_cohort_matrix`, `nextstat_churn_compare`, `nextstat_churn_uplift`, `nextstat_churn_uplift_survival`
Starter prompt: "Over the remote server, run the requested seeded churn data generation, Cox PH churn-risk analysis, churn retention, raw-array ingest and validation, bounded bootstrap hazard-ratio analysis, trust-gate diagnostics, cohort life-table matrix, pairwise churn comparison, bounded uplift analysis, or survival-native uplift analysis deterministically, report synthetic or normalized churn arrays, Cox PH coefficients and hazard-ratio summaries, bootstrap HR summaries with explicit CI settings, overall survival, diagnostics warnings, cohort retention periods, pairwise segment-comparison outputs, causal uplift summaries, or RMST/survival-difference summaries with explicit statistical settings, and stay within the authenticated server-safe subset."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/churn-subscription.md`

### `server_ads_variance_reduction` — Server-safe ads variance reduction
Summary: Run deterministic CUPED or CURE adjustment remotely through nextstat-server using JSON-only ads inputs and provenance-aware leakage validation.
Tools: `nextstat_ads_cuped_adjust`, `nextstat_ads_cure_adjust`
Starter prompt: "Over the remote server, run the requested ads variance-reduction adjustment deterministically, keep covariates strictly pre-treatment, and report adjusted means, treatment effect, variance-reduction diagnostics, selected covariates, provenance validation, and solver metadata without assuming local-only helpers are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/benchmarks/ads-variance-reduction-benchmark-2026-03-08.md`

### `server_reserving_analysis` — Server-safe reserving analysis
Summary: Run deterministic chain ladder or Mack chain ladder reserving remotely through nextstat-server using JSON-only triangle inputs and authenticated server execution.
Tools: `nextstat_chain_ladder`
Starter prompt: "Over the remote server, run the requested chain ladder or Mack chain ladder analysis deterministically, normalize square triangles with null trailing cells when needed, and report development factors, ultimates, reserves, and uncertainty bands without assuming local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/references/python-api.md`

### `server_fault_tree_analysis` — Server-safe fault-tree Monte Carlo
Summary: Run bounded CPU-only fault-tree Monte Carlo or Bernoulli-only CE-IS analysis remotely through nextstat-server using deterministic JSON-only inputs.
Tools: `nextstat_fault_tree_mc`, `nextstat_fault_tree_ce_is`
Starter prompt: "Over the remote server, run the requested fault-tree Monte Carlo or CE-IS analysis deterministically on CPU, report the estimated top-event probability, uncertainty interval, failure counts, and proposal diagnostics when available, and stay within the authenticated server-safe subset without assuming GPU or local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/references/python-api.md`

### `server_meta_analysis` — Server-safe meta-analysis
Summary: Run bounded fixed-effects or random-effects meta-analysis remotely through nextstat-server using deterministic JSON-only inputs.
Tools: `nextstat_meta_analysis`
Starter prompt: "Over the remote server, run the requested fixed-effects or random-effects meta-analysis deterministically, report the pooled estimate, uncertainty, heterogeneity statistics, and forest rows, and stay within the authenticated server-safe subset."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/phase-9-cross-vertical.md`

### `server_panel_econometrics` — Server-safe panel econometrics
Summary: Run bounded panel fixed-effects regressions remotely through nextstat-server using deterministic JSON-only inputs and authenticated server execution.
Tools: `nextstat_panel_fe`
Starter prompt: "Over the remote server, run the requested panel fixed-effects regression deterministically, keep the workflow inside the authenticated server-safe subset, and report coefficients, clustered standard errors, entity counts, and the effective clustering mode without assuming local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/phase-12-did-event-study.md`

### `server_did_analysis` — Server-safe DiD analysis
Summary: Run bounded Difference-in-Differences TWFE analyses remotely through nextstat-server using deterministic JSON-only inputs and authenticated server execution.
Tools: `nextstat_did`
Starter prompt: "Over the remote server, run the requested Difference-in-Differences TWFE analysis deterministically, keep the workflow inside the authenticated server-safe subset, and report the ATT, its uncertainty, and the clustered coefficient surface without assuming local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/phase-12-did-event-study.md`

### `server_iv_analysis` — Server-safe IV / 2SLS analysis
Summary: Run bounded IV / 2SLS analyses remotely through nextstat-server using deterministic JSON-only inputs and authenticated server execution.
Tools: `nextstat_iv_2sls`
Starter prompt: "Over the remote server, run the requested IV / 2SLS analysis deterministically, keep the workflow inside the authenticated server-safe subset, and report structural coefficients, uncertainty, and first-stage F diagnostics without assuming local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/phase-12-did-event-study.md`

### `server_aipw_analysis` — Server-safe AIPW causal analysis
Summary: Run bounded doubly-robust AIPW causal analyses remotely through nextstat-server using deterministic JSON-only inputs and authenticated server execution.
Tools: `nextstat_aipw`
Starter prompt: "Over the remote server, run the requested AIPW causal analysis deterministically, keep the workflow inside the authenticated server-safe subset, and report the estimand, treatment effect estimate, uncertainty, and observation count without assuming local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/phase-12-did-event-study.md`

### `server_event_study_analysis` — Server-safe event-study analysis
Summary: Run bounded event-study TWFE analyses remotely through nextstat-server using deterministic JSON-only inputs and authenticated server execution.
Tools: `nextstat_event_study`
Starter prompt: "Over the remote server, run the requested event-study TWFE analysis deterministically, keep the workflow inside the authenticated server-safe subset, and report the relative-time coefficients, uncertainty, and reference period without assuming local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/phase-12-did-event-study.md`

### `server_volatility_analysis` — Server-safe volatility analysis
Summary: Run bounded GARCH-family volatility analyses remotely through nextstat-server using deterministic JSON-only inputs and authenticated server execution.
Tools: `nextstat_garch_fit`
Starter prompt: "Over the remote server, run the requested GARCH-family volatility analysis deterministically, keep the workflow inside the authenticated server-safe subset, and report fitted parameters, conditional variance paths, and convergence diagnostics without assuming local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/references/econometrics.md`

### `server_state_space_analysis` — Server-safe state-space analysis
Summary: Run bounded Kalman filtering, smoothing, or forecasting remotely through nextstat-server using deterministic JSON-only inputs and authenticated server execution.
Tools: `nextstat_kalman`
Starter prompt: "Over the remote server, run the requested Kalman filter, smoother, or forecast deterministically, keep the workflow inside the authenticated server-safe subset, and report log-likelihood, state trajectories, and forecast paths without assuming local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/references/econometrics.md`

### `server_bioequivalence_analysis` — Server-safe bioequivalence analysis
Summary: Run bounded average bioequivalence, power, or sample-size calculations remotely through nextstat-server using deterministic JSON-only inputs and authenticated server execution.
Tools: `nextstat_bioequivalence`
Starter prompt: "Over the remote server, run the requested bioequivalence test, power, or sample-size calculation deterministically, keep the workflow inside the authenticated server-safe subset, and report the geometric mean ratio, power, or achieved sample size without assuming local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/pharma-pk.md`

### `server_dose_response_analysis` — Server-safe dose-response analysis
Summary: Run bounded Emax or Sigmoid-Emax dose-response predictions and likelihood evaluations remotely through nextstat-server using deterministic JSON-only inputs and authenticated server execution.
Tools: `nextstat_dose_response`
Starter prompt: "Over the remote server, run the requested Emax or Sigmoid-Emax dose-response prediction or likelihood evaluation deterministically, keep the workflow inside the authenticated server-safe subset, and report predictions or NLL values without assuming local-only tools are available."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/references/python-api.md`

### `server_pharma_population_pk` — Server-safe population PK fitting
Summary: Run bounded FOCE, FOCEI, FO, or SAEM population PK fits remotely through nextstat-server using deterministic JSON-only inputs, and run promoted GOF, NPDE, or VPC follow-up diagnostics when requested.
Tools: `nextstat_pharma_fit`, `nextstat_pk_gof`, `nextstat_pk_npde`, `nextstat_pharma_vpc`
Starter prompt: "Over the remote server, fit the requested population PK model deterministically using the promoted nextstat_pharma_fit contract, report theta, omega, ETA summaries, OFV, convergence status, and any covariance-step or SAEM diagnostics that are returned, and when the user asks for bounded diagnostics, run nextstat_pk_gof, nextstat_pk_npde, or nextstat_pharma_vpc inside the authenticated server-safe subset with explicit deterministic inputs without assuming broader trial-simulation tools are available remotely."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/pharma-pk.md`

### `server_pharma_trial_simulation` — Server-safe PK trial simulation
Summary: Run bounded single-trial PK simulations remotely through nextstat-server using deterministic JSON-only inputs and explicit seeds.
Tools: `nextstat_trial_simulate`
Starter prompt: "Over the remote server, run the requested single-trial PK simulation deterministically using nextstat_trial_simulate, keep the workflow inside the authenticated server-safe subset, report concentration profiles, individual PK parameters, and derived endpoints (AUC, Cmax, Tmax, Ctrough), and do not assume Monte Carlo, dose-optimization, or other broader trial-simulation APIs are available remotely."
Docs: `docs/references/tool-api.md`, `docs/references/server-api.md`, `docs/tutorials/pharma-pk.md`
<!-- END GENERATED SERVER GUIDANCE RECIPES -->

Server mode does **not** currently expose the full local registry. ROOT file ingest, broader Monte Carlo / dose-optimization pharma workflows, and other heavier local workflows remain local-only; use the manifest-driven server tool table above as the canonical remote subset.

Python usage:

```python
from nextstat.tools import get_toolkit, execute_tool

server_url = "http://127.0.0.1:3742"
api_key = "secret-key"
tools = get_toolkit(transport="server", server_url=server_url, api_key=api_key)

out = execute_tool(
    "nextstat_fit",
    {"workspace_json": "{...}", "execution": {"deterministic": True}},
    transport="server",
    server_url=server_url,
    api_key=api_key,
)
```

Notes:
- `server_url` can also be provided via env vars: `NEXTSTAT_SERVER_URL` or `NEXTSTAT_TOOLS_SERVER_URL`.
- `api_key` can also be provided via env vars: `NEXTSTAT_SERVER_API_KEY` or `NEXTSTAT_TOOLS_API_KEY`.
- `execute_tool(..., transport="server")` falls back to local execution by default only on network/transport failures.
  - HTTP auth/rate-limit failures and invalid server envelopes do not silently fall back.
  - To disable fallback: `fallback_to_local=False`.
