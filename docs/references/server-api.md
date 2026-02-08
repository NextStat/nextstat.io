<!--
Reference
Created: 2026-02-08
Scope: nextstat-server endpoints + tool runtime
-->

# NextStat Server API (nextstat-server)

Server crate: `/Users/andresvlc/WebDev/nextstat.io/crates/ns-server`

The server is a self-hosted HTTP API for running NextStat inference centrally (CPU and optional CUDA/Metal).

## Endpoints (v1)

- `GET /v1/health`
- `POST /v1/fit`
- `POST /v1/ranking`
- `POST /v1/batch/fit`
- `POST /v1/batch/toys`
- `POST /v1/models` (upload/cache)
- `GET /v1/models` (list cache)
- `DELETE /v1/models/{id}` (evict)
- `GET /v1/tools/schema` (tool registry for agents)
- `POST /v1/tools/execute` (tool execution, `nextstat.tool_result.v1` envelope)

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
