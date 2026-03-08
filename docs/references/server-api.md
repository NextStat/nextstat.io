<!--
Reference
Created: 2026-02-08
Scope: nextstat-server endpoints + tool runtime
-->

# NextStat Server API (nextstat-server)

Server crate: `crates/ns-server`

The server is a self-hosted HTTP API for running NextStat inference centrally (CPU and optional CUDA/Metal).

## Endpoints (v1)

### Inference
- `POST /v1/fit` — MLE fit (HistFactory binned model)
- `POST /v1/ranking` — systematic ranking (impact plot)
- `POST /v1/unbinned/fit` — unbinned MLE fit (event-level likelihood)
- `POST /v1/nlme/fit` — NLME / PK population fit

### Batch
- `POST /v1/batch/fit` — batch MLE fit (up to 100 workspaces)
- `POST /v1/batch/toys` — batch toy fits (pseudo-experiments)

### Async Jobs
- `POST /v1/jobs/submit` — submit long-running task (returns `job_id`)
- `GET /v1/jobs/{id}` — poll job status
- `DELETE /v1/jobs/{id}` — cancel a job
- `GET /v1/jobs` — list all jobs

### Model Cache
- `POST /v1/models` — upload workspace to cache
- `GET /v1/models` — list cached models
- `DELETE /v1/models/{id}` — evict cached model

### Tool API (Agent Surface)
- `GET /v1/tools/schema` — server-safe tool registry for agents
- `POST /v1/tools/execute` — tool execution (`nextstat.tool_result.v1` envelope)

### Admin
- `GET /v1/health` — server status, version, GPU info
- `GET /v1/openapi.json` — OpenAPI 3.1 specification

## `POST /v1/fit` (MLE fit)

Request JSON:

- `workspace` (object, optional): pyhf/HS3/simplified-likelihood workspace JSON (full object, not a string)
- `model_id` (string, optional): cached model id (SHA-256) from `/v1/models`
- `gpu` (boolean or string, optional; default `true`):
  - `true` / `"auto"`: use GPU if server started with `--gpu cuda|metal` (else CPU)
  - `false` / `"cpu"`: force CPU
  - `"cuda"` / `"metal"`: require that specific server GPU device (case-insensitive)

Examples:

```json
{ "workspace": { "...": "..." }, "gpu": true }
```

```json
{ "workspace": { "...": "..." }, "gpu": "Metal" }
```

## `POST /v1/unbinned/fit` (Unbinned MLE fit)

Request JSON:

- `spec` (object, required): unbinned spec JSON (`nextstat_unbinned_spec_v0` schema)
- `data_root` (string, optional; default `"."`): server-side directory containing data files referenced by the spec. Relative paths in `spec.channels[].data.file` are resolved against this.

```json
{
  "spec": {
    "schema_version": "nextstat_unbinned_spec_v0",
    "model": { "poi": "mu", "parameters": [...] },
    "channels": [...]
  },
  "data_root": "/data/experiment_2026"
}
```

Response: same shape as `/v1/fit` — `parameter_names`, `bestfit`, `uncertainties`, `nll`, `converged`, etc.

## `POST /v1/nlme/fit` (NLME / PK fit)

Request JSON:

- `model_type` (string, required): `"pk_1cpt"` (individual 1-compartment oral PK) or `"nlme_1cpt"` (population NLME with log-normal random effects)
- `times` (array of numbers, required): observation times (≥ 0)
- `observations` (array of numbers, required): observed concentrations (≥ 0)
- `dose` (number, required): dose amount (> 0)
- `sigma` (number, required): observation noise std dev (> 0)
- `bioavailability` (number, optional; default `1.0`)
- `subject_idx` (array of integers): required for `nlme_1cpt`, maps each observation to a subject `[0, n_subjects)`
- `n_subjects` (integer): required for `nlme_1cpt`
- `lloq` (number, optional): lower limit of quantification
- `lloq_policy` (string, optional; default `"censored"`): `"ignore"`, `"replace_half"`, or `"censored"`

Individual PK example:

```json
{
  "model_type": "pk_1cpt",
  "times": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
  "observations": [1.2, 2.8, 4.1, 3.5, 1.8, 0.4],
  "dose": 100.0,
  "sigma": 0.05
}
```

Population NLME example:

```json
{
  "model_type": "nlme_1cpt",
  "times": [0.25, 0.5, 1.0, 2.0, 0.25, 0.5, 1.0, 2.0],
  "observations": [1.2, 2.8, 4.1, 3.5, 1.0, 2.5, 3.8, 3.2],
  "subject_idx": [0, 0, 0, 0, 1, 1, 1, 1],
  "n_subjects": 2,
  "dose": 100.0,
  "sigma": 0.05,
  "lloq": 0.1,
  "lloq_policy": "censored"
}
```

Response: `model_type`, `parameter_names`, `bestfit`, `uncertainties`, `nll`, `converged`, `covariance`, `wall_time_s`.

## Async Jobs

For long-running tasks (large toy studies, scans), use the async job API:

### `POST /v1/jobs/submit`

```json
{
  "task_type": "batch_toys",
  "payload": {
    "workspace": { "...": "..." },
    "n_toys": 10000,
    "seed": 42
  }
}
```

Response: `{ "job_id": "job-...", "status": "pending" }`

### `GET /v1/jobs/{id}`

Response:

```json
{
  "id": "job-...",
  "status": "running",
  "task_type": "batch_toys",
  "elapsed_s": 12.5,
  "result": null,
  "error": null
}
```

Status values: `pending`, `running`, `completed`, `failed`, `cancelled`.
When `status == "completed"`, `result` contains the full output (same schema as the sync endpoint).

### `DELETE /v1/jobs/{id}`

Requests cancellation. Response: `{ "cancelled": true, "job_id": "..." }`.

### `GET /v1/jobs`

Lists all jobs (including completed/failed within TTL). Response: `{ "jobs": [...] }`.

## Authentication

Controlled via `--api-keys <file>` CLI flag or `NS_API_KEYS` environment variable.

- **File mode**: one API key per line (blank lines and `#` comments ignored)
- **Env mode**: comma-separated keys in `NS_API_KEYS`
- **Disabled**: if neither is configured, all endpoints are open (dev mode)

When enabled, all endpoints except `GET /v1/health` require:

```
Authorization: Bearer <api-key>
```

Unauthorized requests receive `401` with `{ "error": "..." }`.

## Rate Limiting

Controlled via `--rate-limit <N>` CLI flag (requests per second per IP).

- Default: `0` (unlimited)
- Health endpoint is always exempt
- Returns `429 Too Many Requests` with `{ "error": "rate limit exceeded", "retry_after_s": 1 }` when exceeded

## OpenAPI Specification

`GET /v1/openapi.json` returns a complete OpenAPI 3.1 specification covering all endpoints, schemas, and security definitions. If auth is enabled on the server, this endpoint also requires a bearer token.

## Tool Runtime (Agent Surface)

The tool runtime is designed to mirror a **server-safe subset** of `nextstat.tools`, not the full local Python registry:
- Stable envelope: `schema_version = "nextstat.tool_result.v1"`
- Correct semantics: CLs is CLs; discovery p-values are separate
- Determinism controls: `execution.deterministic` with parity-friendly eval mode and request-scoped thread control when possible
- Canonical registry source: `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`
- Manifest schema: `docs/schemas/tools/nextstat_tool_manifest_v1.schema.json`
- Manifest validator: `scripts/validate_tool_manifest.py`
- Server policy defaults/overrides: `manifest.policies.server`

Boundary note:
- JSON-only analytics helpers can be promoted into the authenticated
  `nextstat-server` subset when they have manifest-backed tool contracts,
  strict result schemas, and parity coverage.
- ROOT ingest is also promoted when it stays server-safe: agents upload
  bounded ROOT file bytes and `nextstat-server` reads the histogram fully in
  memory instead of touching arbitrary server filesystem paths.
- ads variance-reduction helpers are promoted through
  `nextstat_ads_cuped_adjust` and `nextstat_ads_cure_adjust`
- they remain tool-runtime capabilities, not dedicated standalone REST
  endpoints outside `POST /v1/tools/execute`
- Local-only capabilities remain the exception only when they would require
  unsafe server-side file access or an unbounded runtime surface.

Schema references:
- Tool discovery descriptor: `docs/schemas/tools/nextstat_tool_schema_v1.schema.json`
- Descriptor examples: `docs/specs/nextstat_tool_schema_local_v1.example.json`, `docs/specs/nextstat_tool_schema_server_v1.example.json`
- Generic envelope: `docs/schemas/tools/nextstat_tool_result_v1.schema.json`
- Server-safe strict result shapes: `docs/schemas/tools/nextstat_tool_result_server_strict_v1.schema.json`
- Sync/check generator: `scripts/generate_tool_contract_schemas.py`
- Descriptor validator: `scripts/validate_tool_schema_descriptor.py --transport server --server-url ... --api-key ...`
- Descriptor example generator: `scripts/generate_tool_schema_examples.py`
- Reference doc sync/check: `scripts/generate_tool_reference_docs.py`
- Golden drift check: `scripts/generate_tool_goldens.py --check`
- Runner report schema: `docs/schemas/tools/nextstat_tool_contract_runner_report_v1.schema.json`
- Dashboard generator: `scripts/summarize_tool_contract_reports.py`
- Dashboard schema: `docs/schemas/tools/nextstat_tool_contract_dashboard_v1.schema.json`
- Artifact manifest: `scripts/tool_contract_artifact_manifest_v1.json`
- Artifact manifest schema: `docs/schemas/tools/nextstat_tool_contract_artifact_manifest_v1.schema.json`
- Artifact manifest helper: `scripts/tool_contract_artifact_manifest.py`
- Performance budget manifest: `scripts/tool_contract_performance_budget_v1.json`
- Performance budget schema: `docs/schemas/tools/nextstat_tool_contract_performance_budget_v1.schema.json`
- Performance budget helper: `scripts/tool_contract_performance_budget.py`
- Agent bootstrap packs: `docs/references/agent-bootstrap.md`
- Canonical runner:
  - Fast lane: `./.venv/bin/python scripts/check_tool_contracts.py --mode fast`
  - Live lane: `./.venv/bin/python scripts/check_tool_contracts.py --mode live`
  - End-to-end: `./.venv/bin/python scripts/check_tool_contracts.py --mode all`
  - Fast lane auto-syncs `bindings/ns-py` into the active Python environment via `python -m maturin develop -m bindings/ns-py/Cargo.toml` before the Python contract suite
  - JSON report: add `--report-json tmp/reports/tool_contracts_live_report.json`
  - Dashboard: `./.venv/bin/python scripts/summarize_tool_contract_reports.py --report tmp/reports/tool_contracts_fast_report.json --report tmp/reports/tool_contracts_live_report.json --out-json tmp/reports/tool_contract_dashboard.json --out-md tmp/reports/tool_contract_dashboard.md`
  - Failure drilldown surfaces the failed step command and compact `stdout/stderr` tails in the dashboard markdown/job summary
  - Failure classes: `none`, `schema_drift`, `performance_budget_failure`, `rust_contract_failure`, `python_contract_failure`, `live_server_failure`, `unknown`
  - Synthetic step `Validate tool-contract performance budgets` enforces runner duration budgets and live metrics budgets
  - Fast Rust lane uses isolated cargo target dir by default: `.nextstat-cargo-target/tool-contracts` (override via `NEXTSTAT_TOOL_CONTRACT_CARGO_TARGET_DIR`)
- Bindings sync uses its own isolated cargo target dir by default: `.nextstat-cargo-target/tool-contracts-bindings` (override via `NEXTSTAT_TOOL_CONTRACT_BINDINGS_CARGO_TARGET_DIR`)
- The CI dashboard job still runs when upstream contract jobs fail, so classified incident summaries remain available on red builds
- Artifact names/paths across runner jobs and the dashboard consumer now come from the canonical artifact manifest and are regression-tested in `tests/python/test_tool_contract_workflow.py`

Simplified-likelihood support class in March 2026:
- `stable`: `nextstat_workspace_audit`, `nextstat_fit`, `nextstat_hypotest`, `nextstat_upper_limit`, `nextstat_scan`
- `research-grade`: `nextstat_discovery_asymptotic`, `nextstat_ranking`, `nextstat_hypotest_toys` for simplified-likelihood inputs
- companion docs: `docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md`, `docs/benchmarks/simplified-likelihood-release-notes-2026-03-08.md`, `docs/benchmarks/simplified-likelihood-exporter-stable-source-semantics-boundary-2026-03-09.md`

The published future stable exporter boundary is intentionally narrower than the
current research-grade exporter runtime: `pyhf` source only, single-POI only,
and `constraint_covariance_source="source_model_constraints"` for
Gaussian-constrained source nuisances. Derived reduced artifacts remain
reduced-coordinate models rather than source-level nuisance-identity replicas.

Current server-exposed tools:
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

The server still does **not** expose the broader local Python tool registry wholesale. ROOT file ingest, broader Monte Carlo / dose-optimization pharma workflows, and other heavier local workflows remain local-only; the manifest-driven table above is the canonical current server-safe subset.

### Determinism Notes

`ns_compute::EvalMode` is process-wide. To avoid races, `nextstat-server` serializes inference requests behind a global compute lock. This means:
- per-request `execution.eval_mode` is safe (no cross-request bleed)
- total throughput is lower (one inference request at a time)

GPU policy in server tools:
- If `execution.deterministic=true` (default), tools run on CPU (parity-friendly).
- If `execution.deterministic=false` and the server is started with `--gpu cuda|metal`, some tools may use GPU (fit/ranking/scan).

### `GET /v1/tools/schema`

Returns:
- `schema_version = "nextstat.tool_schema.v1"`
- `transport = "server"`
- `tools`: OpenAI-compatible tool definitions
- `capabilities`: machine-readable transport/policy metadata for tool discovery
- `guidance`: transport-aware hints and curated workflow recipes for agents and IDE assistants

This is intended so an agent can bootstrap tool definitions without importing Python.
Smart clients should treat `tools` as the callable subset and `capabilities` as the broader policy/discovery map.
The same response now also includes transport-aware `guidance` so agents and IDE assistants can bootstrap prompt recipes without maintaining a separate manual list.
The committed server example fixture is also checked against the live route contract in server tests.
If auth is enabled on the server, `GET /v1/tools/schema` requires the same bearer token as other non-health endpoints. It is also subject to normal rate limiting.

Server guidance recipes:
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

### `POST /v1/tools/execute`

Request:

```json
{
  "name": "nextstat_fit",
  "arguments": {
    "workspace_json": "{...}",
    "execution": { "deterministic": true }
  }
}
```

Response is always a tool envelope:

```json
{
  "schema_version": "nextstat.tool_result.v1",
  "ok": true,
  "result": { },
  "error": null,
  "meta": { "tool_name": "nextstat_fit", "nextstat_version": "..." }
}
```

Operational semantics:
- If auth is enabled on the server, `POST /v1/tools/execute` requires the same bearer token as other non-health endpoints.
- The endpoint is subject to normal rate limiting.
- Malformed JSON bodies or invalid top-level request shape return HTTP `400` with a standard JSON error body: `{ "error": "..." }`.
- Missing or non-JSON `Content-Type` returns HTTP `415` with the same JSON error shape: `{ "error": "..." }`.
- Tool-level validation or domain failures still return HTTP `200` with `ok = false` inside the `nextstat.tool_result.v1` envelope.
- HTTP error statuses are reserved for transport/runtime concerns such as auth and rate limiting.

Python client usage (no extra deps; uses stdlib HTTP in `nextstat.tools`):

```python
from nextstat.tools import get_toolkit, execute_tool

server_url = "http://127.0.0.1:3742"
api_key = "secret-key"
tools = get_toolkit(transport="server", server_url=server_url, api_key=api_key)

out = execute_tool(
    "nextstat_fit",
    {"workspace_json": "...", "execution": {"deterministic": True}},
    transport="server",
    server_url=server_url,
    api_key=api_key,
)
print(out)
```

Notes:
- `server_url` can also be provided via `NEXTSTAT_SERVER_URL` or `NEXTSTAT_TOOLS_SERVER_URL`.
- `api_key` can also be provided via `NEXTSTAT_SERVER_API_KEY` or `NEXTSTAT_TOOLS_API_KEY`.
- `execute_tool(..., transport="server")` falls back to local execution by default only on network/transport errors.
  - HTTP auth/rate-limit failures and invalid server envelopes do not silently fall back.
  - Use `fallback_to_local=False` if you want server-only behavior.

## Security / Input Policy

Server mode does **not** expose file-ingest tools (like reading ROOT files from arbitrary paths) via `/v1/tools/*`.
If you need ROOT ingest for a demo agent, do it client-side (local Python) and send derived data to the server.
