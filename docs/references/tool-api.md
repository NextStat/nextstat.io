<!--
Reference
Created: 2026-02-08
Scope: nextstat.tools contract usage
-->

# Tool API (nextstat.tools)

This is the user-facing reference for NextStat's **agent/tool-calling** surface.

Source of truth for determinism semantics: `/Users/andresvlc/WebDev/nextstat.io/docs/internal/plans/standards.md`  
Tool contract (schemas/seed policy): `/Users/andresvlc/WebDev/nextstat.io/docs/internal/plans/2026-02-08_tool-api-contract-v1.md`

## What You Get

1. `nextstat.tools.get_toolkit()` returns OpenAI-compatible tool definitions (JSON Schema input).
2. `nextstat.tools.execute_tool(name, arguments)` executes a tool call and returns a stable envelope:

```json
{
  "schema_version": "nextstat.tool_result.v1",
  "ok": true,
  "result": {},
  "error": null,
  "meta": {
    "tool_name": "nextstat_fit",
    "nextstat_version": "0.1.0",
    "deterministic": true,
    "eval_mode": "parity",
    "threads_requested": 1
  }
}
```

Schema files:
- `/Users/andresvlc/WebDev/nextstat.io/docs/schemas/tools/nextstat_tool_result_v1.schema.json` (envelope)
- `/Users/andresvlc/WebDev/nextstat.io/docs/schemas/tools/nextstat_tool_result_strict_v1.schema.json` (envelope + strict result shapes)

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
- Thread control is **best-effort** (Rayon global pool is one-shot). Check `meta.threads_applied` when present.

## Semantics (Avoid Trust-Killers)

- `nextstat_hypotest` returns **CLs** (+ `clsb`, `clb`). It is not a p-value.
- Discovery-style outputs use `nextstat_discovery_asymptotic` which returns `{q0, z0, p0}`.

## Toy Tools and Seeds

Stochastic tools require a `seed` (default 42, but agents should set explicitly):
- `nextstat_hypotest_toys`: returns toy-based CLs as a `"raw"` payload plus the seed metadata.

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

## Example: ROOT Ingest

Use `nextstat_read_root_histogram` to fetch TH1 content for downstream analysis.

## Prompt Templates (Copy/Paste)

- Fit + report POI:
  - "Fit this workspace and return the POI value and error. Use deterministic mode."
- Upper limit:
  - "Compute the observed and expected 95% CL upper limit (CLs) for this workspace. Use deterministic mode."
- Discovery summary:
  - "Compute the asymptotic discovery significance (q0, z0, p0) at mu=0. Use deterministic mode."
- ROOT quick look:
  - "Read histogram `hist1` from `tests/fixtures/simple_histos.root` and summarize its bin contents and under/overflow."

## Regression Harness

- Golden outputs: `/Users/andresvlc/WebDev/nextstat.io/tests/fixtures/tool_goldens/simple_workspace_deterministic.v1.json`
- Generator: `/Users/andresvlc/WebDev/nextstat.io/scripts/generate_tool_goldens.py`
- Smoke runner: `/Users/andresvlc/WebDev/nextstat.io/scripts/tool_call_smoke.py`

## Server Mode (nextstat-server)

If you run `nextstat-server`, you can also fetch tools and execute them over HTTP:
- Tool registry: `GET /v1/tools/schema`
- Tool execution: `POST /v1/tools/execute`

See: `/Users/andresvlc/WebDev/nextstat.io/docs/references/server-api.md`

Python usage:

```python
from nextstat.tools import get_toolkit, execute_tool

server_url = "http://127.0.0.1:3742"
tools = get_toolkit(transport="server", server_url=server_url)

out = execute_tool(
    "nextstat_hypotest",
    {"workspace_json": "...", "mu": 1.0, "execution": {"deterministic": True}},
    transport="server",
    server_url=server_url,
)
```

Notes:
- `server_url` can also be provided via env vars: `NEXTSTAT_SERVER_URL` or `NEXTSTAT_TOOLS_SERVER_URL`.
- `execute_tool(..., transport="server")` falls back to local execution by default when the server call fails.
  - To disable fallback: `fallback_to_local=False`.
