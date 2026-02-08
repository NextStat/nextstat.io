"""LLM Tool Definitions for Agentic Analysis.

Provides OpenAI-compatible function-calling schemas so that AI agents
(GPT-4o, Llama 4, Mistral-Next, Claude, local Ollama models) can
discover and invoke NextStat operations programmatically.

Usage with OpenAI::

    from nextstat.tools import get_toolkit, execute_tool
    import openai

    tools = get_toolkit()  # list of OpenAI function-calling dicts
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Fit this workspace and show me the POI"}],
        tools=tools,
    )

    # When the agent calls a tool:
    for call in response.choices[0].message.tool_calls:
        result = execute_tool(call.function.name, json.loads(call.function.arguments))
        # result is a JSON-serialisable dict

Usage with LangChain / LlamaIndex::

    from nextstat.tools import get_langchain_tools
    tools = get_langchain_tools()  # list of langchain BaseTool

Usage standalone (MCP server)::

    from nextstat.tools import get_mcp_tools
    tools = get_mcp_tools()  # list of MCP tool dicts
"""

from __future__ import annotations

import json
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "nextstat_fit",
            "description": (
                "Run Maximum Likelihood Estimation (MLE) on a HistFactory statistical model. "
                "Returns best-fit parameters, uncertainties, NLL at minimum, and convergence info. "
                "The workspace_json must be a pyhf-style or HS3-style JSON workspace string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace. Auto-detected.",
                    },
                },
                "required": ["workspace_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_hypotest",
            "description": (
                "Run an asymptotic CLs hypothesis test at a given signal strength mu. "
                "Returns CLs, CLs+b, CLb values and optionally the expected band (±1σ, ±2σ)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace.",
                    },
                    "mu": {
                        "type": "number",
                        "description": "Signal strength hypothesis to test (e.g. 1.0 for SM).",
                    },
                    "expected_set": {
                        "type": "boolean",
                        "description": "If true, return expected CLs band (±1σ, ±2σ). Default false.",
                        "default": False,
                    },
                },
                "required": ["workspace_json", "mu"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_upper_limit",
            "description": (
                "Compute the 95% CL upper limit on signal strength via CLs. "
                "Returns observed limit and optionally expected limits (±1σ, ±2σ)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace.",
                    },
                    "expected": {
                        "type": "boolean",
                        "description": "If true, return expected limits with ±1σ/±2σ bands.",
                        "default": False,
                    },
                },
                "required": ["workspace_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_ranking",
            "description": (
                "Compute nuisance parameter ranking (systematic impact on signal strength). "
                "Returns a sorted list of systematics with their impact (delta_mu_up, delta_mu_down), "
                "pull values, and constraints. This is the physics equivalent of Feature Importance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Return only the top N most impactful systematics. Default: all.",
                    },
                },
                "required": ["workspace_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_significance",
            "description": (
                "Compute the discovery significance (Z₀) for the signal in the workspace. "
                "Runs a hypothesis test at mu=0 (background-only) and returns the p-value "
                "and significance in standard deviations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace.",
                    },
                },
                "required": ["workspace_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_scan",
            "description": (
                "Run a profile likelihood scan over signal strength values. "
                "Returns (mu, q_mu, 2*delta_NLL) arrays for plotting the likelihood curve."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf or HS3 workspace.",
                    },
                    "start": {
                        "type": "number",
                        "description": "Start of mu scan range. Default 0.0.",
                        "default": 0.0,
                    },
                    "stop": {
                        "type": "number",
                        "description": "End of mu scan range. Default 5.0.",
                        "default": 5.0,
                    },
                    "points": {
                        "type": "integer",
                        "description": "Number of scan points. Default 21.",
                        "default": 21,
                    },
                },
                "required": ["workspace_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "nextstat_workspace_audit",
            "description": (
                "Audit a pyhf workspace for compatibility. Reports channel count, sample count, "
                "modifier types, parameter count, and any unsupported features."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace_json": {
                        "type": "string",
                        "description": "JSON string of the pyhf workspace to audit.",
                    },
                },
                "required": ["workspace_json"],
            },
        },
    },
]


def get_toolkit() -> list[dict[str, Any]]:
    """Return OpenAI-compatible function-calling tool definitions.

    These can be passed directly to ``openai.chat.completions.create(tools=...)``,
    or adapted for any agent framework that uses the OpenAI tool schema.

    Returns:
        ``list[dict]`` — each dict has ``type: "function"`` and a ``function``
        key with ``name``, ``description``, and ``parameters`` (JSON Schema).

    Example::

        import openai
        from nextstat.tools import get_toolkit

        tools = get_toolkit()
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Fit this workspace"}],
            tools=tools,
        )
    """
    return [dict(t) for t in _TOOLS]


def get_tool_names() -> list[str]:
    """Return the list of available tool names."""
    return [t["function"]["name"] for t in _TOOLS]


def get_tool_schema(name: str) -> Optional[dict[str, Any]]:
    """Return the JSON Schema for a specific tool by name.

    Returns ``None`` if the tool is not found.
    """
    for t in _TOOLS:
        if t["function"]["name"] == name:
            return dict(t)
    return None


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _load_model(workspace_json: str):
    """Load a HistFactoryModel from a JSON string (auto-detects pyhf vs HS3)."""
    import nextstat

    model = nextstat.HistFactoryModel.from_workspace(workspace_json)
    return model


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a NextStat tool by name with the given arguments.

    This is the bridge between an LLM agent's tool call and NextStat's
    Python API. The return value is always a JSON-serialisable dict.

    Args:
        name: tool name (e.g. ``"nextstat_fit"``).
        arguments: dict of arguments matching the tool's JSON Schema.

    Returns:
        ``dict`` — result payload (JSON-serialisable).

    Raises:
        ValueError: if the tool name is unknown.
        Exception: if the underlying NextStat call fails (propagated).

    Example::

        result = execute_tool("nextstat_fit", {"workspace_json": ws_str})
        print(result["nll"], result["converged"])
    """
    import nextstat

    if name == "nextstat_fit":
        model = _load_model(arguments["workspace_json"])
        result = nextstat.fit(model)
        params = result.parameters
        names = model.parameter_names()
        poi_idx = model.poi_index()
        return {
            "nll": result.nll,
            "converged": result.converged,
            "n_iter": result.n_iter,
            "poi_index": poi_idx,
            "poi_value": params[poi_idx] if poi_idx is not None else None,
            "poi_error": result.uncertainties[poi_idx] if poi_idx is not None else None,
            "parameters": {n: {"value": v, "error": e}
                          for n, v, e in zip(names, params, result.uncertainties)},
        }

    elif name == "nextstat_hypotest":
        model = _load_model(arguments["workspace_json"])
        mu = float(arguments["mu"])
        expected_set = arguments.get("expected_set", False)
        result = nextstat.hypotest(mu, model, expected_set=expected_set)
        if expected_set:
            return {"cls_obs": result[0], "cls_exp": result[1]}
        return {"cls_obs": float(result)}

    elif name == "nextstat_upper_limit":
        model = _load_model(arguments["workspace_json"])
        expected = arguments.get("expected", False)
        result = nextstat.upper_limit(model, expected=expected)
        if expected and isinstance(result, tuple):
            return {"obs_limit": result[0], "exp_limits": list(result[1])}
        return {"obs_limit": float(result)}

    elif name == "nextstat_ranking":
        model = _load_model(arguments["workspace_json"])
        from nextstat.interpret import rank_impact
        top_n = arguments.get("top_n")
        table = rank_impact(model, top_n=top_n)
        return {"ranking": table}

    elif name == "nextstat_significance":
        model = _load_model(arguments["workspace_json"])
        result = nextstat.hypotest(0.0, model)
        cls_val = float(result)
        import math
        if cls_val > 0 and cls_val < 1:
            from nextstat._core import _normal_quantile  # type: ignore
            try:
                z0 = _normal_quantile(1.0 - cls_val)
            except Exception:
                z0 = math.sqrt(2) * math.erfc(2 * cls_val) if cls_val > 0 else 0.0
        else:
            z0 = 0.0
        return {"cls_obs": cls_val, "significance_sigma": z0}

    elif name == "nextstat_scan":
        model = _load_model(arguments["workspace_json"])
        start = arguments.get("start", 0.0)
        stop = arguments.get("stop", 5.0)
        points = arguments.get("points", 21)
        step = (stop - start) / max(points - 1, 1)
        mu_values = [start + i * step for i in range(points)]
        result = nextstat.profile_scan(model, mu_values)
        return {
            "mu_values": result.get("mu_values", mu_values),
            "twice_delta_nll": result.get("twice_delta_nll", []),
            "mu_hat": result.get("mu_hat"),
            "nll_hat": result.get("nll_hat"),
        }

    elif name == "nextstat_workspace_audit":
        ws_json = arguments["workspace_json"]
        result = nextstat.workspace_audit(ws_json)
        return dict(result)

    else:
        raise ValueError(
            f"Unknown tool: {name!r}. Available: {get_tool_names()}"
        )


# ---------------------------------------------------------------------------
# LangChain integration (optional)
# ---------------------------------------------------------------------------


def get_langchain_tools():
    """Return NextStat tools as LangChain ``StructuredTool`` instances.

    Requires ``langchain-core`` to be installed.

    Example::

        from nextstat.tools import get_langchain_tools
        tools = get_langchain_tools()
        agent = create_tool_calling_agent(llm, tools, prompt)
    """
    try:
        from langchain_core.tools import StructuredTool  # type: ignore
    except ImportError:
        raise ImportError(
            "langchain-core is required for get_langchain_tools(). "
            "Install: pip install langchain-core"
        )

    lc_tools = []
    for tool_def in _TOOLS:
        fn_def = tool_def["function"]
        name = fn_def["name"]

        def _make_fn(tool_name):
            def fn(**kwargs):
                return execute_tool(tool_name, kwargs)
            fn.__name__ = tool_name
            fn.__doc__ = fn_def["description"]
            return fn

        lc_tools.append(
            StructuredTool.from_function(
                func=_make_fn(name),
                name=name,
                description=fn_def["description"],
            )
        )

    return lc_tools


# ---------------------------------------------------------------------------
# MCP (Model Context Protocol) integration
# ---------------------------------------------------------------------------


def get_mcp_tools() -> list[dict[str, Any]]:
    """Return NextStat tools as MCP tool definitions.

    Compatible with the Model Context Protocol standard for AI tool servers.

    Returns:
        ``list[dict]`` — each dict has ``name``, ``description``, ``inputSchema``.
    """
    mcp_tools = []
    for tool_def in _TOOLS:
        fn_def = tool_def["function"]
        mcp_tools.append({
            "name": fn_def["name"],
            "description": fn_def["description"],
            "inputSchema": fn_def["parameters"],
        })
    return mcp_tools


def handle_mcp_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle an MCP tool call. Alias for :func:`execute_tool`."""
    return execute_tool(name, arguments)
