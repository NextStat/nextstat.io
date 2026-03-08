# Agent Bootstrap Packs

This page tracks the generated bootstrap packs for major agent/client surfaces
that teams actually use with NextStat: `Codex`, `Gemini`, `Ollama/local`,
`MCP` tool servers, and IDE assistants.

The goal is not another hand-maintained prompt page. These packs are generated
from:

- `scripts/agent_bootstrap_profile_manifest_v1.json`
- `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`

That means client-specific guidance stays aligned with:

- the real callable surface (`tools`)
- transport policy (`capabilities`)
- curated workflow prompts (`guidance`)

## IDE Workspace Files

<!-- BEGIN GENERATED IDE WORKSPACE FILES -->
Generated from the canonical workspace-output bootstrap profiles and transport-aware tool guidance.

| Profile | File | Purpose |
|---------|------|---------|
| `ide` | `.github/copilot-instructions.md` | Repo-native Copilot instructions generated from the canonical workspace-output bootstrap profile. |
| `ide` | `.cursor/rules/nextstat-tools.mdc` | Cursor rule file generated from the same workspace-output bootstrap profile and transport-aware guidance. |
<!-- END GENERATED IDE WORKSPACE FILES -->

## Runnable Integration Examples

<!-- BEGIN GENERATED PROVIDER EXAMPLES -->
Generated from the canonical `codex`, `gemini`, `claude`, `ollama`, `mcp` bootstrap profiles plus transport-aware guidance.

| Script | Purpose |
|--------|---------|
| `docs/specs/agent_bootstrap/examples/nextstat_codex_local_example.py` | Runnable Codex/OpenAI local bootstrap that prints an OpenAI-compatible tool-calling request payload from the canonical local descriptor. |
| `docs/specs/agent_bootstrap/examples/nextstat_codex_server_example.py` | Runnable Codex/OpenAI server bootstrap that prints an OpenAI-compatible tool-calling request payload from the authenticated server descriptor. |
| `docs/specs/agent_bootstrap/examples/nextstat_gemini_local_example.py` | Runnable Gemini local bootstrap that prints a function-calling request payload from the canonical local descriptor. |
| `docs/specs/agent_bootstrap/examples/nextstat_gemini_server_example.py` | Runnable Gemini server bootstrap that prints a function-calling request payload from the authenticated server descriptor. |
| `docs/specs/agent_bootstrap/examples/nextstat_claude_local_example.py` | Runnable Claude local bootstrap that prints an Anthropic-native messages plus tools request payload from the canonical local descriptor. |
| `docs/specs/agent_bootstrap/examples/nextstat_claude_server_example.py` | Runnable Claude server bootstrap that prints an Anthropic-native messages plus tools request payload from the authenticated server descriptor. |
| `docs/specs/agent_bootstrap/examples/nextstat_ollama_local_example.py` | Runnable Ollama local bootstrap that prints a manual-tool-loop request payload from the canonical local descriptor. |
| `docs/specs/agent_bootstrap/examples/nextstat_ollama_server_example.py` | Runnable Ollama server bootstrap that prints a manual-tool-loop request payload from the authenticated server descriptor. |
| `docs/specs/agent_bootstrap/examples/nextstat_mcp_local_example.py` | Runnable MCP local bootstrap that prints an MCP tool-server config from the canonical local MCP mirror plus descriptor guidance. |
| `docs/specs/agent_bootstrap/examples/nextstat_mcp_server_example.py` | Runnable MCP server bootstrap that prints an MCP tool-server config derived from the authenticated server descriptor. |
<!-- END GENERATED PROVIDER EXAMPLES -->

## Profile Matrix

<!-- BEGIN GENERATED AGENT PROFILE MATRIX -->
Generated from `scripts/agent_bootstrap_profile_manifest_v1.json` and the canonical tool manifest guidance.

| Client | Summary | Local Pack | Server Pack |
|--------|---------|------------|-------------|
| `codex` | Bootstrap pack for OpenAI-compatible tool calling, including Codex-style agent loops. | `docs/specs/agent_bootstrap/nextstat_codex_local_bootstrap_v1.json` | `docs/specs/agent_bootstrap/nextstat_codex_server_bootstrap_v1.json` |
| `gemini` | Bootstrap pack for Gemini-style function calling using the same canonical NextStat descriptor. | `docs/specs/agent_bootstrap/nextstat_gemini_local_bootstrap_v1.json` | `docs/specs/agent_bootstrap/nextstat_gemini_server_bootstrap_v1.json` |
| `claude` | Bootstrap pack for Anthropic Claude tool use using a provider-native messages plus tools request shape. | `docs/specs/agent_bootstrap/nextstat_claude_local_bootstrap_v1.json` | `docs/specs/agent_bootstrap/nextstat_claude_server_bootstrap_v1.json` |
| `ollama` | Bootstrap pack for local models that may use MCP or a manual validated tool loop instead of first-class function calling. | `docs/specs/agent_bootstrap/nextstat_ollama_local_bootstrap_v1.json` | `docs/specs/agent_bootstrap/nextstat_ollama_server_bootstrap_v1.json` |
| `mcp` | Bootstrap pack for Model Context Protocol tool servers using the canonical NextStat tool and guidance surfaces. | `docs/specs/agent_bootstrap/nextstat_mcp_local_bootstrap_v1.json` | `docs/specs/agent_bootstrap/nextstat_mcp_server_bootstrap_v1.json` |
| `ide` | Bootstrap pack for editor-integrated assistants that need short workspace-safe rules and transport-aware recipes. | `docs/specs/agent_bootstrap/nextstat_ide_local_bootstrap_v1.json` | `docs/specs/agent_bootstrap/nextstat_ide_server_bootstrap_v1.json` |
<!-- END GENERATED AGENT PROFILE MATRIX -->

## Profile Details

<!-- BEGIN GENERATED AGENT PROFILE DETAILS -->
Generated from `scripts/agent_bootstrap_profile_manifest_v1.json` plus transport-aware guidance from `bindings/ns-py/python/nextstat/_tool_manifest_v1.json`.

## `codex` — Codex / OpenAI tool-calling
Bootstrap pack for OpenAI-compatible tool calling, including Codex-style agent loops.

Global instructions:
- Use descriptor.tools as the callable contract and do not invent tools outside the NextStat descriptor.
- Keep deterministic execution on by default for reproducible scientific workflows unless the user explicitly asks for stochastic runs.

### `local`
Pack: `docs/specs/agent_bootstrap/nextstat_codex_local_bootstrap_v1.json`
Recipes: 8
Transport instructions:
- Prefer local transport when the analysis needs the full Python registry, including ROOT ingest and non-HEP verticals.
Bootstrap snippet:
```text
from nextstat.tools import get_toolkit_descriptor\ndescriptor = get_toolkit_descriptor(transport="local")\ntools = descriptor["tools"]\nguidance = descriptor["guidance"]
```
Execution-loop snippet:
```text
Use the OpenAI-compatible descriptor.tools list directly. When the model emits a tool call, validate the arguments against the chosen function schema and execute with nextstat.tools.execute_tool(..., transport="local").
```

### `server`
Pack: `docs/specs/agent_bootstrap/nextstat_codex_server_bootstrap_v1.json`
Recipes: 21
Transport instructions:
- Prefer server transport when the workflow should stay auth-enabled and remote through nextstat-server.
Bootstrap snippet:
```text
from nextstat.tools import get_toolkit_descriptor\ndescriptor = get_toolkit_descriptor(transport="server", server_url=server_url, api_key=api_key)\ntools = descriptor["tools"]\nguidance = descriptor["guidance"]
```
Execution-loop snippet:
```text
Use descriptor.tools as the only callable server-safe subset. Execute tool calls with nextstat.tools.execute_tool(..., transport="server", server_url=..., api_key=..., fallback_to_local=False) when you need strict remote behavior.
```

## `gemini` — Gemini function-calling
Bootstrap pack for Gemini-style function calling using the same canonical NextStat descriptor.

Global instructions:
- Map descriptor.tools into Gemini function declarations without changing tool names or argument schemas.
- Use descriptor.guidance.recipes as prompt seeds so the model stays aligned with the intended transport surface.

### `local`
Pack: `docs/specs/agent_bootstrap/nextstat_gemini_local_bootstrap_v1.json`
Recipes: 8
Transport instructions:
- Local transport is the broadest surface and should be the default when the Gemini workflow runs inside a trusted Python environment.
Bootstrap snippet:
```text
descriptor = get_toolkit_descriptor(transport="local")\nfunction_declarations = descriptor["tools"]\nrecipe_seeds = descriptor["guidance"]["recipes"]
```
Execution-loop snippet:
```text
Present function_declarations to Gemini, then execute approved calls with nextstat.tools.execute_tool(..., transport="local"). Use descriptor.guidance.hints to explain local-only capabilities when the workflow needs broader coverage.
```

### `server`
Pack: `docs/specs/agent_bootstrap/nextstat_gemini_server_bootstrap_v1.json`
Recipes: 21
Transport instructions:
- Server transport keeps Gemini on the server-safe subset and is the right default for remote/shared deployments.
Bootstrap snippet:
```text
descriptor = get_toolkit_descriptor(transport="server", server_url=server_url, api_key=api_key)\nfunction_declarations = descriptor["tools"]\nrecipe_seeds = descriptor["guidance"]["recipes"]
```
Execution-loop snippet:
```text
Only expose descriptor.tools to Gemini. When a call is selected, execute it remotely with nextstat.tools.execute_tool(..., transport="server", server_url=..., api_key=..., fallback_to_local=False).
```

## `claude` — Anthropic / Claude tool use
Bootstrap pack for Anthropic Claude tool use using a provider-native messages plus tools request shape.

Global instructions:
- Map descriptor.tools into Anthropic tool definitions without renaming tools or mutating argument schemas.
- Keep the system prompt concise and use descriptor.guidance.recipes as the workflow layer instead of embedding long procedural prompts.

### `local`
Pack: `docs/specs/agent_bootstrap/nextstat_claude_local_bootstrap_v1.json`
Recipes: 8
Transport instructions:
- Use local transport when the Claude workflow runs inside a trusted Python environment and needs the full local NextStat descriptor.
Bootstrap snippet:
```text
descriptor = get_toolkit_descriptor(transport="local")\ntools = descriptor["tools"]\nrecipe_seeds = descriptor["guidance"]["recipes"]
```
Execution-loop snippet:
```text
Expose descriptor.tools as Anthropic tool definitions, keep descriptor.guidance.recipes as the workflow layer, and execute approved calls with nextstat.tools.execute_tool(..., transport="local").
```

### `server`
Pack: `docs/specs/agent_bootstrap/nextstat_claude_server_bootstrap_v1.json`
Recipes: 21
Transport instructions:
- Use server transport for remote or shared Claude workflows so the callable surface stays aligned with the authenticated server-safe subset.
Bootstrap snippet:
```text
descriptor = get_toolkit_descriptor(transport="server", server_url=server_url, api_key=api_key)\ntools = descriptor["tools"]\nrecipe_seeds = descriptor["guidance"]["recipes"]
```
Execution-loop snippet:
```text
Only expose descriptor.tools to Claude, then execute approved calls with nextstat.tools.execute_tool(..., transport="server", server_url=..., api_key=..., fallback_to_local=False).
```

## `ollama` — Ollama / local open-weight models
Bootstrap pack for local models that may use MCP or a manual validated tool loop instead of first-class function calling.

Global instructions:
- Do not assume the model can emit perfect function calls; keep a validated manual loop available.
- Use descriptor.capabilities to explain unavailable tools instead of silently substituting local-only functionality.

### `local`
Pack: `docs/specs/agent_bootstrap/nextstat_ollama_local_bootstrap_v1.json`
Recipes: 8
Transport instructions:
- For trusted local automation, use the full local descriptor and, if needed, get_mcp_tools() as the MCP-facing mirror.
Bootstrap snippet:
```text
descriptor = get_toolkit_descriptor(transport="local")\nmcp_tools = get_mcp_tools()\nmanual_loop_contract = descriptor["tools"]
```
Execution-loop snippet:
```text
If the local model lacks reliable function calling, ask it to choose a tool name plus JSON arguments, validate that pair against descriptor.tools, then execute with nextstat.tools.execute_tool(..., transport="local").
```

### `server`
Pack: `docs/specs/agent_bootstrap/nextstat_ollama_server_bootstrap_v1.json`
Recipes: 21
Transport instructions:
- For remote/shared use, keep Ollama on the server-safe subset and disable local fallback when strict server execution matters.
Bootstrap snippet:
```text
descriptor = get_toolkit_descriptor(transport="server", server_url=server_url, api_key=api_key)\nmanual_loop_contract = descriptor["tools"]
```
Execution-loop snippet:
```text
Run a validated manual tool loop over descriptor.tools and execute with nextstat.tools.execute_tool(..., transport="server", server_url=..., api_key=..., fallback_to_local=False). Treat descriptor.guidance.recipes as the workflow prompt layer.
```

## `mcp` — MCP tool server
Bootstrap pack for Model Context Protocol tool servers using the canonical NextStat tool and guidance surfaces.

Global instructions:
- Use MCP tool definitions as the protocol surface, but keep descriptor.guidance.recipes as workflow and prompt seeds.
- Keep MCP tool names and input schemas aligned with the canonical NextStat tool manifest; do not invent aliases.

### `local`
Pack: `docs/specs/agent_bootstrap/nextstat_mcp_local_bootstrap_v1.json`
Recipes: 8
Transport instructions:
- For trusted local automation, build the MCP tool list from nextstat.tools.get_mcp_tools() and dispatch through nextstat.tools.handle_mcp_call().
Bootstrap snippet:
```text
from nextstat.tools import get_mcp_tools, get_toolkit_descriptor\nmcp_tools = get_mcp_tools()\ndescriptor = get_toolkit_descriptor(transport="local")
```
Execution-loop snippet:
```text
Expose mcp_tools through your MCP server or bridge, then dispatch validated calls with nextstat.tools.handle_mcp_call(name, arguments). Use descriptor.guidance.recipes as the workflow layer.
```

### `server`
Pack: `docs/specs/agent_bootstrap/nextstat_mcp_server_bootstrap_v1.json`
Recipes: 21
Transport instructions:
- For remote/shared use, derive MCP tool definitions from get_toolkit_descriptor(transport="server", ...) so the protocol surface stays restricted to the authenticated server-safe subset.
Bootstrap snippet:
```text
descriptor = get_toolkit_descriptor(transport="server", server_url=server_url, api_key=api_key)\nmcp_tools = [{"name": tool["function"]["name"], "description": tool["function"]["description"], "inputSchema": tool["function"]["parameters"]} for tool in descriptor["tools"]]
```
Execution-loop snippet:
```text
Expose the derived mcp_tools through your MCP bridge, then dispatch approved calls with nextstat.tools.execute_tool(..., transport="server", server_url=..., api_key=..., fallback_to_local=False).
```

## `ide` — IDE assistant / copilot rules
Bootstrap pack for editor-integrated assistants that need short workspace-safe rules and transport-aware recipes.

Global instructions:
- Treat the descriptor as the canonical source when authoring IDE rules, snippets, or assistant instructions.
- Prefer short, transport-aware workspace rules instead of long narrative prompts.

### `local`
Pack: `docs/specs/agent_bootstrap/nextstat_ide_local_bootstrap_v1.json`
Recipes: 8
Transport instructions:
- Local IDE rules may mention the full Python registry, including ROOT ingest and non-HEP verticals.
Bootstrap snippet:
```text
Workspace rule seed: 'Load the local NextStat descriptor first, use descriptor.tools as the callable surface, and use descriptor.guidance.recipes as task templates.'
```
Execution-loop snippet:
```text
If the IDE assistant proposes a tool call, validate it against descriptor.tools and prefer deterministic execution for code review, notebooks, and reproducible analyses.
```

### `server`
Pack: `docs/specs/agent_bootstrap/nextstat_ide_server_bootstrap_v1.json`
Recipes: 21
Transport instructions:
- Server IDE rules must describe the restricted subset explicitly so the assistant does not hallucinate local-only tools.
Bootstrap snippet:
```text
Workspace rule seed: 'For remote NextStat usage, only use descriptor.tools from get_toolkit_descriptor(transport="server", ...); use descriptor.capabilities to explain why local-only tools are unavailable.'
```
Execution-loop snippet:
```text
Keep IDE instructions short: audit with descriptor.guidance.recipes, execute through nextstat.tools.execute_tool(..., transport="server", server_url=..., api_key=..., fallback_to_local=False), and avoid local fallback when strict remote behavior is required.
```
<!-- END GENERATED AGENT PROFILE DETAILS -->
