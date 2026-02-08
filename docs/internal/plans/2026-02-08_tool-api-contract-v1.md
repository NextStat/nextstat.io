<!--
Plan
Created: 2026-02-08
Owner: nextstat.io
Purpose: Contract-grade tool API for LLM/automation: schema + determinism + seed policy.
-->

# Tool API Contract v1 (Feb 2026)

Status: active

This document defines a **stable, deterministic, versioned** tool surface intended for:
- LLM tool-calling (OpenAI-style function calling, MCP servers)
- automation / pipelines (CI, benchmarks, validation runs)

The goal is trust: if someone reruns the same tool call, they should get the same answer (within the deterministic contract).

## Why This Is Needed

1. **Semantics must be correct.** Returning CLs but calling it “p-value/significance” destroys trust.
2. **Determinism must be explicit.** Fast mode is allowed to be non-bit-exact; trust artifacts must not be.
3. **Outputs must be parseable long-term.** That means a stable envelope + schema versioning.

## Scope (v1)

This v1 contract standardizes:
- a response envelope (`nextstat.tool_result.v1`)
- deterministic execution knobs (parity mode, threads best-effort)
- seed policy for stochastic tools
- correct naming of inference quantities

It does **not** (yet) guarantee:
- strict per-tool output schemas for every nested field (we start with the envelope schema first)
- process-wide thread control in all environments (Rayon can only be configured once per process)

## Determinism Policy (Source of Truth)

See `/Users/andresvlc/WebDev/nextstat.io/docs/internal/plans/standards.md` for EvalMode semantics.

For tool calls:
- Default is `deterministic=true`, which forces `eval_mode="parity"` and requests `threads=1`.
- `threads` is best-effort. If Rayon was already initialized, the request cannot be applied.
- Parity mode is allowed to be slower but must be reproducible.

## Seed Policy

Any stochastic tool must:
- accept an explicit `seed` argument
- echo back the `seed` and `n_toys` (or similar) in the result

Defaults:
- `seed=42` (only as a fallback; agents should set it explicitly)

## Semantics Policy (Non-Negotiable)

- `hypotest(...)` returns **CLs** (and optionally CLs+b and CLb), not p-values.
- Discovery p-values/significance must be computed and named explicitly:
  - discovery tool returns `q0`, `z0`, `p0` (one-sided).

## Tool Set (Current)

Implemented in `/Users/andresvlc/WebDev/nextstat.io/bindings/ns-py/python/nextstat/tools.py`:

- `nextstat_fit`
- `nextstat_hypotest` (asymptotic CLs + tails)
- `nextstat_hypotest_toys` (toy CLs; returns raw payload to avoid losing information while the output schema stabilizes)
- `nextstat_upper_limit` (observed; optionally expected via `upper_limits_root`)
- `nextstat_ranking`
- `nextstat_scan` (profile scan artifact passthrough)
- `nextstat_discovery_asymptotic` (q0/Z0/p0 at mu=0 from a profiled scan)
- `nextstat_workspace_audit`
- `nextstat_read_root_histogram` (ROOT TH1 ingest for agent demos)

## Response Envelope (v1)

All tool calls return:

```json
{
  "schema_version": "nextstat.tool_result.v1",
  "ok": true,
  "result": {},
  "error": null,
  "meta": {
    "tool_name": "nextstat_fit",
    "nextstat_version": "..."
  }
}
```

Schema: `/Users/andresvlc/WebDev/nextstat.io/docs/schemas/tools/nextstat_tool_result_v1.schema.json`

## Next Steps (v1 -> v1.1)

1. Add per-tool output schemas (not just the envelope), and validate tool outputs in CI.
2. Replace `nextstat_hypotest_toys` result from `"raw"` to a stable structured output once the upstream binding return-shape is frozen.
3. Add a dedicated discovery API in Rust/Python (avoid relying on `profile_scan` shape).
