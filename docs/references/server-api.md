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

## Security / Input Policy

Server mode does **not** expose file-ingest tools (like reading ROOT files from arbitrary paths) via `/v1/tools/*`.
If you need ROOT ingest for a demo agent, do it client-side (local Python) and send derived data to the server.

