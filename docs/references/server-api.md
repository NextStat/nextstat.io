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
- `GET /v1/tools/schema` — tool registry for agents
- `POST /v1/tools/execute` — tool execution (`nextstat.tool_result.v1` envelope)

### Admin
- `GET /v1/health` — server status, version, GPU info
- `GET /v1/openapi.json` — OpenAPI 3.1 specification

## `POST /v1/fit` (MLE fit)

Request JSON:

- `workspace` (object, optional): pyhf/HS3 workspace JSON (full object, not a string)
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

`GET /v1/openapi.json` returns a complete OpenAPI 3.1 specification covering all endpoints, schemas, and security definitions. No auth required.

## Tool Runtime (Agent Surface)

The tool runtime is designed to mirror `nextstat.tools`:
- Stable envelope: `schema_version = "nextstat.tool_result.v1"`
- Correct semantics: CLs is CLs; discovery p-values are separate
- Determinism controls: `execution.deterministic` (best-effort on server)

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
- `tools`: OpenAI-compatible tool definitions

This is intended so an agent can bootstrap tool definitions without importing Python.

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

Python client usage (no extra deps; uses stdlib HTTP in `nextstat.tools`):

```python
from nextstat.tools import get_toolkit, execute_tool

server_url = "http://127.0.0.1:3742"
tools = get_toolkit(transport="server", server_url=server_url)

out = execute_tool(
    "nextstat_fit",
    {"workspace_json": "...", "execution": {"deterministic": True}},
    transport="server",
    server_url=server_url,
)
print(out)
```

Notes:
- `server_url` can also be provided via `NEXTSTAT_SERVER_URL` or `NEXTSTAT_TOOLS_SERVER_URL`.
- `execute_tool(..., transport="server")` falls back to local execution by default on transport errors.
  - Use `fallback_to_local=False` if you want server-only behavior.

## Security / Input Policy

Server mode does **not** expose file-ingest tools (like reading ROOT files from arbitrary paths) via `/v1/tools/*`.
If you need ROOT ingest for a demo agent, do it client-side (local Python) and send derived data to the server.
