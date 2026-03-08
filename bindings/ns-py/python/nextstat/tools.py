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

import base64
import copy
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Optional

from ._tool_manifest import build_toolkit_descriptor, get_transport_tools


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, Any]] = get_transport_tools("local")


def _build_local_toolkit_descriptor() -> dict[str, Any]:
    return build_toolkit_descriptor("local")


def _validate_toolkit_descriptor_schema(
    descriptor: Any,
    *,
    expected_transport: str,
) -> dict[str, Any]:
    if not isinstance(descriptor, dict):
        raise RuntimeError("Invalid tool schema response: descriptor must be an object")
    if descriptor.get("schema_version") != "nextstat.tool_schema.v1":
        raise RuntimeError("Invalid tool schema response: unexpected schema_version")
    if descriptor.get("transport") != expected_transport:
        raise RuntimeError(
            f"Invalid tool schema response: expected transport={expected_transport!r}"
        )

    tools = descriptor.get("tools")
    if not isinstance(tools, list):
        raise RuntimeError("Invalid tool schema response: missing 'tools' list")
    capabilities = descriptor.get("capabilities")
    if not isinstance(capabilities, list):
        raise RuntimeError("Invalid tool schema response: missing 'capabilities' list")
    guidance = descriptor.get("guidance")
    if not isinstance(guidance, dict):
        raise RuntimeError("Invalid tool schema response: missing 'guidance' object")
    hints = guidance.get("hints")
    if not isinstance(hints, list) or not hints:
        raise RuntimeError("Invalid tool schema response: guidance.hints must be a non-empty list")
    if not all(isinstance(hint, str) and hint.strip() for hint in hints):
        raise RuntimeError(
            "Invalid tool schema response: guidance.hints must contain only non-empty strings"
        )
    recipes = guidance.get("recipes")
    if not isinstance(recipes, list):
        raise RuntimeError("Invalid tool schema response: guidance.recipes must be a list")

    tool_names: list[str] = []
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise RuntimeError(f"Invalid tool schema response: tools[{idx}] must be a function tool")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise RuntimeError(f"Invalid tool schema response: tools[{idx}].function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name.startswith("nextstat_"):
            raise RuntimeError(f"Invalid tool schema response: tools[{idx}].function.name is invalid")
        if name in tool_names:
            raise RuntimeError(f"Invalid tool schema response: duplicate tool {name!r}")
        if not isinstance(function.get("description"), str) or not function["description"].strip():
            raise RuntimeError(
                f"Invalid tool schema response: tools[{idx}].function.description is invalid"
            )
        if not isinstance(function.get("parameters"), dict):
            raise RuntimeError(
                f"Invalid tool schema response: tools[{idx}].function.parameters must be an object"
            )
        tool_names.append(name)

    capability_map: dict[str, dict[str, Any]] = {}
    for idx, capability in enumerate(capabilities):
        if not isinstance(capability, dict):
            raise RuntimeError(f"Invalid tool schema response: capabilities[{idx}] must be an object")
        name = capability.get("name")
        if not isinstance(name, str) or not name.startswith("nextstat_"):
            raise RuntimeError(f"Invalid tool schema response: capabilities[{idx}].name is invalid")
        if name in capability_map:
            raise RuntimeError(f"Invalid tool schema response: duplicate capability {name!r}")
        if not isinstance(capability.get("local_available"), bool):
            raise RuntimeError(
                f"Invalid tool schema response: capabilities[{idx}].local_available must be boolean"
            )
        if not isinstance(capability.get("server_available"), bool):
            raise RuntimeError(
                f"Invalid tool schema response: capabilities[{idx}].server_available must be boolean"
            )
        policy = capability.get("server_policy")
        if not isinstance(policy, dict):
            raise RuntimeError(
                f"Invalid tool schema response: capabilities[{idx}].server_policy must be an object"
            )
        availability = policy.get("availability")
        if availability not in ("exposed", "local_only"):
            raise RuntimeError(
                f"Invalid tool schema response: capabilities[{idx}].server_policy.availability is invalid"
            )
        if not isinstance(policy.get("reason_code"), str) or not policy["reason_code"].strip():
            raise RuntimeError(
                f"Invalid tool schema response: capabilities[{idx}].server_policy.reason_code is invalid"
            )
        if not isinstance(policy.get("reason"), str) or not policy["reason"].strip():
            raise RuntimeError(
                f"Invalid tool schema response: capabilities[{idx}].server_policy.reason is invalid"
            )
        if capability["server_available"] != (availability == "exposed"):
            raise RuntimeError(
                "Invalid tool schema response: server_available and server_policy.availability disagree"
            )
        capability_map[name] = capability

    availability_key = "local_available" if expected_transport == "local" else "server_available"
    for name in tool_names:
        capability = capability_map.get(name)
        if capability is None:
            raise RuntimeError(f"Invalid tool schema response: missing capability entry for {name!r}")
        if capability.get(availability_key) is not True:
            raise RuntimeError(
                f"Invalid tool schema response: capability {name!r} is not available on transport {expected_transport!r}"
            )

    seen_recipe_ids: set[str] = set()
    for idx, recipe in enumerate(recipes):
        if not isinstance(recipe, dict):
            raise RuntimeError(f"Invalid tool schema response: guidance.recipes[{idx}] must be an object")
        recipe_id = recipe.get("id")
        if not isinstance(recipe_id, str) or not recipe_id.strip():
            raise RuntimeError(f"Invalid tool schema response: guidance.recipes[{idx}].id is invalid")
        if recipe_id in seen_recipe_ids:
            raise RuntimeError(
                f"Invalid tool schema response: duplicate guidance recipe {recipe_id!r}"
            )
        seen_recipe_ids.add(recipe_id)
        if recipe.get("transport") != expected_transport:
            raise RuntimeError(
                f"Invalid tool schema response: guidance.recipes[{idx}].transport must equal {expected_transport!r}"
            )
        for field in ("title", "summary", "prompt"):
            value = recipe.get(field)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(
                    f"Invalid tool schema response: guidance.recipes[{idx}].{field} is invalid"
                )
        recipe_tools = recipe.get("tools")
        if not isinstance(recipe_tools, list) or not recipe_tools:
            raise RuntimeError(
                f"Invalid tool schema response: guidance.recipes[{idx}].tools must be a non-empty list"
            )
        for tool_name in recipe_tools:
            if tool_name not in tool_names:
                raise RuntimeError(
                    f"Invalid tool schema response: guidance recipe {recipe_id!r} references unavailable tool {tool_name!r}"
                )
        recipe_docs = recipe.get("docs")
        if not isinstance(recipe_docs, list) or not recipe_docs:
            raise RuntimeError(
                f"Invalid tool schema response: guidance.recipes[{idx}].docs must be a non-empty list"
            )
        if not all(isinstance(doc, str) and doc.strip() for doc in recipe_docs):
            raise RuntimeError(
                f"Invalid tool schema response: guidance.recipes[{idx}].docs must contain only non-empty strings"
            )

    return descriptor

class ServerTransportError(RuntimeError):
    """Network-level failure while contacting nextstat-server."""


class ServerHTTPError(RuntimeError):
    """HTTP error returned by nextstat-server."""

    def __init__(self, method: str, url: str, status_code: int, detail: str) -> None:
        self.method = method
        self.url = url
        self.status_code = int(status_code)
        self.detail = detail
        super().__init__(f"HTTP {status_code} from {url}: {detail}")


class InvalidServerResponseError(RuntimeError):
    """The server responded, but the payload violated the declared contract."""


def _resolve_server_api_key(api_key: Optional[str]) -> Optional[str]:
    if api_key and api_key.strip():
        return api_key.strip()
    for k in ("NEXTSTAT_TOOLS_API_KEY", "NEXTSTAT_SERVER_API_KEY"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    return None


def _server_headers(*, api_key: Optional[str], content_type_json: bool = False) -> dict[str, str]:
    headers: dict[str, str] = {}
    if content_type_json:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _http_json_get(url: str, *, timeout_s: float, api_key: Optional[str] = None) -> Any:
    req = urllib.request.Request(url, headers=_server_headers(api_key=api_key), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = resp.read().decode("utf-8")
            try:
                return json.loads(data)
            except Exception as e:
                raise InvalidServerResponseError(
                    f"Invalid JSON response from {url}: {e}"
                ) from e
    except urllib.error.HTTPError as e:
        body = None
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = None
        raise ServerHTTPError("GET", url, e.code, body or e.reason)
    except Exception as e:
        if isinstance(e, InvalidServerResponseError):
            raise
        raise ServerTransportError(f"Failed GET {url}: {e}") from e


def _http_json_post(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    api_key: Optional[str] = None,
) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=_server_headers(api_key=api_key, content_type_json=True),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            out = resp.read().decode("utf-8")
            try:
                return json.loads(out)
            except Exception as e:
                raise InvalidServerResponseError(
                    f"Invalid JSON response from {url}: {e}"
                ) from e
    except urllib.error.HTTPError as e:
        body = None
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = None
        raise ServerHTTPError("POST", url, e.code, body or e.reason)
    except Exception as e:
        if isinstance(e, InvalidServerResponseError):
            raise
        raise ServerTransportError(f"Failed POST {url}: {e}") from e

def _resolve_server_url(server_url: Optional[str]) -> Optional[str]:
    if server_url:
        return server_url.strip()
    for k in ("NEXTSTAT_TOOLS_SERVER_URL", "NEXTSTAT_SERVER_URL"):
        v = os.environ.get(k)
        if v and v.strip():
            return v.strip()
    return None


def get_toolkit(
    *,
    transport: str = "local",
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 10.0,
) -> list[dict[str, Any]]:
    """Return OpenAI-compatible function-calling tool definitions.

    These can be passed directly to ``openai.chat.completions.create(tools=...)``,
    or adapted for any agent framework that uses the OpenAI tool schema.

    Args:
        transport:
            - ``"local"`` (default): return the in-process Python tool registry.
            - ``"server"``: fetch the registry from ``nextstat-server`` at ``GET /v1/tools/schema``.
        server_url: Base URL for server mode, e.g. ``"http://127.0.0.1:3742"``.
        api_key: Optional bearer token for auth-enabled server mode. If omitted, the
            client will also check ``NEXTSTAT_TOOLS_API_KEY`` and ``NEXTSTAT_SERVER_API_KEY``.
        timeout_s: HTTP timeout (server mode only).

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
    if transport not in ("local", "server"):
        raise ValueError(f"Unknown transport: {transport!r}. Use 'local' or 'server'.")
    schema = get_toolkit_descriptor(
        transport=transport,
        server_url=server_url,
        api_key=api_key,
        timeout_s=timeout_s,
    )
    tools = schema.get("tools")
    if not isinstance(tools, list):
        raise RuntimeError("Invalid tool schema response: missing 'tools' list")
    return copy.deepcopy(tools)


def get_toolkit_descriptor(
    *,
    transport: str = "local",
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Return the full tool-discovery descriptor for a transport.

    The descriptor is a versioned machine-readable envelope that contains:
    - ``tools``: callable OpenAI-compatible function definitions
    - ``capabilities``: transport/policy metadata for tool discovery

    ``get_toolkit()`` remains the compatibility helper when only the callable
    ``tools`` list is needed.
    """
    if transport == "local":
        return _validate_toolkit_descriptor_schema(
            _build_local_toolkit_descriptor(),
            expected_transport="local",
        )
    if transport != "server":
        raise ValueError(f"Unknown transport: {transport!r}. Use 'local' or 'server'.")
    server_url = _resolve_server_url(server_url)
    if not server_url:
        raise ValueError("server_url is required for transport='server'")
    api_key = _resolve_server_api_key(api_key)
    schema = _http_json_get(
        f"{server_url.rstrip('/')}/v1/tools/schema",
        timeout_s=timeout_s,
        api_key=api_key,
    )
    return _validate_toolkit_descriptor_schema(schema, expected_transport="server")


def get_tool_names() -> list[str]:
    """Return the list of available tool names."""
    return [t["function"]["name"] for t in _TOOLS]


def get_tool_schema(name: str) -> Optional[dict[str, Any]]:
    """Return the JSON Schema for a specific tool by name.

    Returns ``None`` if the tool is not found.
    """
    for t in _TOOLS:
        if t["function"]["name"] == name:
            return copy.deepcopy(t)
    return None


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _load_model(workspace_json: str):
    """Load a HistFactoryModel from a JSON string (auto-detects pyhf vs HS3)."""
    import nextstat

    model = nextstat.HistFactoryModel.from_workspace(workspace_json)
    return model


def _normal_sf(z: float) -> float:
    """Standard normal survival function (1 - CDF) without SciPy."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _normalize_chain_ladder_triangle(triangle: list[list[Any]]) -> list[list[float]]:
    if not isinstance(triangle, list) or not triangle:
        raise ValueError("triangle must be a non-empty 2D array")
    if not all(isinstance(row, list) for row in triangle):
        raise ValueError("triangle must be a 2D array")

    n = len(triangle)
    row_lengths = [len(row) for row in triangle]
    is_square = all(length == n for length in row_lengths)

    normalized: list[list[float]] = []
    for i, row in enumerate(triangle):
        expected = n - i
        if is_square:
            values = row[:expected]
        else:
            if len(row) != expected:
                raise ValueError(
                    f"triangle row {i} has length {len(row)}, expected {expected} for ragged upper-left triangle"
                )
            values = row

        parsed_row: list[float] = []
        for j, value in enumerate(values):
            if value is None:
                raise ValueError(f"triangle[{i}][{j}] must be a finite non-negative number")
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"triangle[{i}][{j}] must be a finite non-negative number")
            parsed_row.append(number)
        normalized.append(parsed_row)

    return normalized


def _strict_event_bools(values: list[Any], field_name: str) -> list[bool]:
    out: list[bool] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool):
            out.append(value)
        elif isinstance(value, int) and value in (0, 1):
            out.append(bool(value))
        else:
            raise ValueError(
                f"{field_name}[{idx}] must be boolean or 0/1 integer (got {value!r})"
            )
    return out


def _strict_binary_u8(values: list[Any], field_name: str) -> list[int]:
    out: list[int] = []
    for idx, value in enumerate(values):
        if isinstance(value, bool):
            out.append(int(value))
        elif isinstance(value, int) and value in (0, 1):
            out.append(value)
        else:
            raise ValueError(
                f"{field_name}[{idx}] must be boolean or 0/1 integer (got {value!r})"
            )
    return out


def _apply_execution(nextstat, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Apply best-effort execution controls. Returns meta additions."""
    exec_cfg = arguments.get("execution") or {}
    deterministic = bool(exec_cfg.get("deterministic", True))

    requested_eval_mode = exec_cfg.get("eval_mode")
    if deterministic:
        eval_mode = "parity"
    else:
        eval_mode = requested_eval_mode

    prev_eval_mode = None
    if eval_mode in ("fast", "parity"):
        try:
            prev_eval_mode = nextstat.get_eval_mode()
            if prev_eval_mode != eval_mode:
                nextstat.set_eval_mode(eval_mode)
        except Exception:
            prev_eval_mode = None

    requested_threads = exec_cfg.get("threads")
    if requested_threads is None and deterministic:
        requested_threads = 1

    threads_applied = None
    if isinstance(requested_threads, int):
        try:
            set_threads = getattr(nextstat, "set_threads", None)
            if callable(set_threads):
                threads_applied = bool(set_threads(int(requested_threads)))
        except Exception:
            threads_applied = False

    return {
        "tool_name": tool_name,
        "deterministic": deterministic,
        "eval_mode_effective": eval_mode if eval_mode in ("fast", "parity") else None,
        "eval_mode_prev": prev_eval_mode,
        "threads_requested": requested_threads,
        "threads_applied": threads_applied,
    }


def _restore_execution(nextstat, meta: dict[str, Any]) -> None:
    prev = meta.get("eval_mode_prev")
    if prev in ("fast", "parity"):
        try:
            if nextstat.get_eval_mode() != prev:
                nextstat.set_eval_mode(prev)
        except Exception:
            pass


def _execute_tool_impl(nextstat, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Implementation body for tools, assuming execution controls are already applied."""
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
            "parameters": {
                n: {"value": v, "error": e}
                for n, v, e in zip(names, params, result.uncertainties)
            },
        }

    if name == "nextstat_hypotest":
        model = _load_model(arguments["workspace_json"])
        mu = float(arguments["mu"])
        cls_val, tails = nextstat.hypotest(mu, model, return_tail_probs=True)
        clsb, clb = tails
        return {"mu": mu, "cls": float(cls_val), "clsb": float(clsb), "clb": float(clb)}

    if name == "nextstat_hypotest_toys":
        model = _load_model(arguments["workspace_json"])
        mu = float(arguments["mu"])
        n_toys = int(arguments.get("n_toys", 1000))
        seed = int(arguments.get("seed", 42))
        expected_set = bool(arguments.get("expected_set", False))
        return_meta = bool(arguments.get("return_meta", False))
        r = nextstat.hypotest_toys(
            mu,
            model,
            n_toys=n_toys,
            seed=seed,
            expected_set=expected_set,
            return_tail_probs=True,
            return_meta=return_meta,
        )
        # Return shape depends on expected_set/return_meta; keep it explicit and lossless.
        return {
            "mu": mu,
            "n_toys": n_toys,
            "seed": seed,
            "expected_set": expected_set,
            "raw": r,
        }

    if name == "nextstat_upper_limit":
        model = _load_model(arguments["workspace_json"])
        expected = bool(arguments.get("expected", False))
        alpha = float(arguments.get("alpha", 0.05))
        lo = float(arguments.get("lo", 0.0))
        hi = arguments.get("hi", None)
        hi_val = None if hi is None else float(hi)
        rtol = float(arguments.get("rtol", 1e-4))
        max_iter = int(arguments.get("max_iter", 80))
        if expected:
            obs, exp = nextstat.upper_limit(
                model, method="root", alpha=alpha, lo=lo, hi=hi_val, rtol=rtol, max_iter=max_iter
            )
            return {
                "alpha": alpha,
                "obs_limit": float(obs),
                "exp_limits": [float(x) for x in exp],
            }
        obs = nextstat.upper_limit(model, alpha=alpha, lo=lo, hi=hi_val, rtol=rtol, max_iter=max_iter)
        return {"alpha": alpha, "obs_limit": float(obs)}

    if name == "nextstat_ranking":
        model = _load_model(arguments["workspace_json"])
        from nextstat.interpret import rank_impact

        top_n = arguments.get("top_n")
        table = rank_impact(model, top_n=top_n)
        return {"ranking": table}

    if name == "nextstat_discovery_asymptotic":
        model = _load_model(arguments["workspace_json"])
        fit_res = nextstat.fit(model)
        poi_idx = model.poi_index()
        mu_hat = None
        if poi_idx is not None:
            try:
                mu_hat = float(fit_res.parameters[poi_idx])
            except Exception:
                mu_hat = None
        scan = nextstat.profile_scan(model, [0.0])
        pts = scan.get("points") or []
        if not pts:
            raise RuntimeError("profile_scan returned no points for mu=0")
        nll0 = float(pts[0].get("nll_mu"))
        nll_hat = float(fit_res.nll)
        q0_raw = 2.0 * (nll0 - nll_hat)
        q0 = max(0.0, q0_raw)
        if mu_hat is not None and mu_hat <= 0.0:
            q0 = 0.0
        z0 = math.sqrt(q0)
        p0 = _normal_sf(z0)
        return {
            "mu_hat": mu_hat,
            "nll_hat": nll_hat,
            "nll_mu0": nll0,
            "q0": q0,
            "z0": z0,
            "p0": p0,
        }

    if name == "nextstat_scan":
        model = _load_model(arguments["workspace_json"])
        start = float(arguments.get("start", 0.0))
        stop = float(arguments.get("stop", 5.0))
        points = int(arguments.get("points", 21))
        step = (stop - start) / max(points - 1, 1)
        mu_values = [start + i * step for i in range(points)]
        artifact = nextstat.profile_scan(model, mu_values)
        return dict(artifact)

    if name == "nextstat_workspace_audit":
        ws_json = arguments["workspace_json"]
        result = nextstat.workspace_audit(ws_json)
        return dict(result)

    if name == "nextstat_read_root_histogram":
        hist_path = arguments["hist_path"]
        root_path = arguments.get("root_path")
        root_bytes_base64 = arguments.get("root_bytes_base64")
        filename_hint = arguments.get("filename_hint")

        if bool(root_path) == bool(root_bytes_base64):
            raise ValueError(
                "nextstat_read_root_histogram requires exactly one of root_path or root_bytes_base64"
            )
        if root_bytes_base64 is not None:
            root_bytes = base64.b64decode(str(root_bytes_base64), validate=True)
            result = nextstat._read_root_histogram_bytes(
                root_bytes,
                hist_path,
                filename_hint=(str(filename_hint) if filename_hint is not None else None),
            )
        else:
            result = nextstat.read_root_histogram(str(root_path), hist_path)
        return dict(result)

    # -------------------------------------------------------------------
    # Cross-vertical tools
    # -------------------------------------------------------------------

    if name == "nextstat_glm_fit":
        import nextstat.glm as glm

        x = arguments["x"]
        y = arguments["y"]
        family = arguments.get("family", "linear")
        intercept = arguments.get("include_intercept", True)
        l2 = arguments.get("l2")

        if family == "linear":
            fit = glm.linear.fit(x, y, include_intercept=intercept, l2=l2)
            return {
                "family": "linear",
                "coef": list(fit.coef),
                "standard_errors": list(fit.standard_errors),
                "sigma2_hat": fit.sigma2_hat,
            }
        elif family == "logistic":
            fit = glm.logistic.fit(x, y, include_intercept=intercept, l2=l2)
            return {
                "family": "logistic",
                "coef": list(fit.coef),
                "standard_errors": list(fit.standard_errors),
            }
        elif family == "poisson":
            fit = glm.poisson.fit(x, y, include_intercept=intercept, l2=l2)
            return {
                "family": "poisson",
                "coef": list(fit.coef),
                "standard_errors": list(fit.standard_errors),
            }
        elif family == "negbin":
            fit = glm.negbin.fit(x, y, include_intercept=intercept, l2=l2)
            return {
                "family": "negbin",
                "coef": list(fit.coef),
                "standard_errors": list(fit.standard_errors),
                "alpha": fit.alpha,
            }
        else:
            raise ValueError(f"Unknown GLM family: {family!r}")

    if name == "nextstat_bayesian_sample":
        import nextstat.bayes

        mt = arguments["model_type"]
        x = arguments.get("x")
        y = arguments.get("y")
        time_arr = arguments.get("time")
        event_arr = arguments.get("event")
        n_levels = arguments.get("n_levels")
        ws_json = arguments.get("workspace_json")
        n_chains = int(arguments.get("n_chains", 4))
        n_warmup = int(arguments.get("n_warmup", 500))
        n_samples = int(arguments.get("n_samples", 1000))
        seed = int(arguments.get("seed", 42))
        target_accept = float(arguments.get("target_accept", 0.8))
        y_int = [int(v) for v in y] if y is not None else None
        event_bool = [bool(v) for v in event_arr] if event_arr is not None else None

        model_map = {
            "linear_regression": lambda: nextstat.LinearRegressionModel(x, y),
            "logistic_regression": lambda: nextstat.LogisticRegressionModel(x, y_int),
            "poisson_regression": lambda: nextstat.PoissonRegressionModel(x, y_int),
            "negbin_regression": lambda: nextstat.NegativeBinomialRegressionModel(x, y_int),
            "cox_ph": lambda: nextstat.CoxPhModel(time_arr, event_bool, x, ties="efron"),
            "weibull_survival": lambda: nextstat.WeibullSurvivalModel(time_arr, event_bool),
            "lognormal_aft": lambda: nextstat.LogNormalAftModel(time_arr, event_bool),
            "ordered_logit": lambda: nextstat.OrderedLogitModel(x, y_int, n_levels),
            "ordered_probit": lambda: nextstat.OrderedProbitModel(x, y_int, n_levels),
            "histfactory": lambda: nextstat.HistFactoryModel.from_workspace(ws_json),
        }
        if mt not in model_map:
            raise ValueError(f"Unknown model_type: {mt!r}")
        model = model_map[mt]()

        raw = nextstat.bayes.sample(
            model,
            n_chains=n_chains,
            n_warmup=n_warmup,
            n_samples=n_samples,
            seed=seed,
            target_accept=target_accept,
            return_idata=False,
        )
        diag = raw.get("diagnostics", {})
        return {
            "model_type": mt,
            "n_chains": n_chains,
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "param_names": raw.get("param_names", []),
            "diagnostics": diag,
            "posterior_summary": {
                k: {
                    "mean": sum(sum(ch) for ch in chains) / max(1, sum(len(ch) for ch in chains)),
                }
                for k, chains in (raw.get("posterior") or {}).items()
            },
        }

    if name == "nextstat_survival_fit":
        x = [[float(v) for v in row] for row in arguments["x"]]
        time_arr = [float(v) for v in arguments["time"]]
        event_arr = [bool(v) for v in arguments["event"]]
        model_name = arguments.get("model", "cox_ph")

        if len(x) != len(time_arr) or len(time_arr) != len(event_arr):
            raise ValueError("x, time, and event must have the same length")
        if x and any(len(row) != len(x[0]) for row in x):
            raise ValueError("x must be a rectangular 2D array")

        model_map = {
            "cox_ph": lambda: nextstat.CoxPhModel(time_arr, event_arr, x, ties="efron"),
            "weibull": lambda: nextstat.WeibullSurvivalModel(time_arr, event_arr),
            "lognormal_aft": lambda: nextstat.LogNormalAftModel(time_arr, event_arr),
            "exponential": lambda: nextstat.ExponentialSurvivalModel(time_arr, event_arr),
        }
        if model_name not in model_map:
            raise ValueError(f"Unknown survival model: {model_name!r}")
        model = model_map[model_name]()
        result = nextstat.fit(model)
        return {
            "model": model_name,
            "parameters": list(result.parameters),
            "uncertainties": list(result.uncertainties),
            "nll": result.nll,
            "converged": result.converged,
        }

    if name == "nextstat_kaplan_meier":
        time_arr = [float(v) for v in arguments["time"]]
        event_arr = [bool(v) for v in arguments["event"]]
        group_arr = arguments.get("group")
        if group_arr is not None:
            group_arr = [int(v) for v in group_arr]

        if len(time_arr) != len(event_arr):
            raise ValueError("time and event must have the same length")
        if group_arr is not None and len(group_arr) != len(time_arr):
            raise ValueError("group must have the same length as time and event")

        km = nextstat.kaplan_meier(time_arr, event_arr)
        out = dict(km)
        if group_arr is not None:
            lr = nextstat.log_rank_test(time_arr, event_arr, group_arr)
            out["log_rank"] = dict(lr)
        return out

    if name == "nextstat_log_rank_test":
        times = [float(v) for v in arguments["times"]]
        events = _strict_event_bools(arguments["events"], "events")
        groups = [int(v) for v in arguments["groups"]]
        if len(times) != len(events) or len(events) != len(groups):
            raise ValueError("times, events, and groups must have the same length")
        result = nextstat.log_rank_test(times, events, groups)
        return dict(result)

    if name == "nextstat_panel_fe":
        import nextstat.econometrics as econ

        x = arguments["x"]
        y = arguments["y"]
        entity = arguments["entity"]
        time_arr = arguments.get("time")
        cluster = arguments.get("cluster", "entity")

        fit = econ.panel_fe_fit(x, y, entity=entity, time=time_arr, cluster=cluster)
        if fit.cluster == "entity":
            n_clusters = len(set(entity))
        elif fit.cluster == "time":
            n_clusters = 0 if time_arr is None else len(set(time_arr))
        elif fit.cluster == "two_way":
            n_clusters = 0 if time_arr is None else len(set(zip(entity, time_arr)))
        else:
            n_clusters = 0
        return {
            "coef": list(fit.coef),
            "standard_errors": list(fit.standard_errors),
            "n_obs": fit.n_obs,
            "n_entities": fit.n_entities,
            "cluster_kind": fit.cluster,
            "n_clusters": int(n_clusters),
            "cluster": fit.cluster,
        }

    if name == "nextstat_did":
        import nextstat.econometrics as econ

        y = arguments["y"]
        treat = arguments["treat"]
        post = arguments["post"]
        entity = arguments["entity"]
        time_arr = arguments["time"]
        x = arguments.get("x")
        cluster = arguments.get("cluster", "entity")

        did = econ.did_twfe_fit(x, y, treat=treat, post=post, entity=entity, time=time_arr, cluster=cluster)
        return {
            "att": did.att,
            "att_se": did.att_se,
            "coef": list(did.twfe.coef),
            "standard_errors": list(did.twfe.standard_errors),
            "n_obs": did.twfe.n_obs,
            "cluster": did.twfe.cluster,
        }

    if name == "nextstat_iv_2sls":
        import nextstat.econometrics as econ

        y = arguments["y"]
        endog = arguments["endog"]
        instruments = arguments["instruments"]
        exog = arguments.get("exog")
        cov = arguments.get("cov", "hc1")
        cluster = arguments.get("cluster")
        time_index = arguments.get("time_index")
        max_lag = arguments.get("max_lag")

        iv = econ.iv_2sls_fit(
            y,
            endog=endog,
            instruments=instruments,
            exog=exog,
            cov=cov,
            cluster=cluster,
            time_index=time_index,
            max_lag=max_lag,
        )
        out = {
            "coef": list(iv.coef),
            "standard_errors": list(iv.standard_errors),
            "n_obs": iv.n_obs,
        }
        if hasattr(iv, "diagnostics") and iv.diagnostics is not None:
            out["diagnostics"] = {
                "first_stage_f": iv.diagnostics.first_stage_f,
            }
        return out

    if name == "nextstat_aipw":
        from nextstat.causal.aipw import aipw_fit

        x = arguments["x"]
        y = arguments["y"]
        treatment = arguments["treatment"]
        estimand = arguments.get("estimand", "ate")

        result = aipw_fit(x, y, treatment, estimand=estimand)
        return {
            "estimand": result.estimand,
            "estimate": result.estimate,
            "standard_error": result.standard_error,
            "n_obs": result.n_obs,
        }

    if name == "nextstat_meta_analysis":
        effects = arguments["effects"]
        ses = arguments["standard_errors"]
        method = arguments.get("method", "random")

        if method == "fixed":
            result = nextstat.meta_fixed(effects, ses)
        else:
            result = nextstat.meta_random(effects, ses)
        return dict(result)

    if name == "nextstat_ads_cuped_adjust":
        result = nextstat.ads.cuped_adjust(
            arguments["control_outcomes"],
            arguments["control_covariates"],
            arguments["variant_outcomes"],
            arguments["variant_covariates"],
            covariate_name=arguments.get("covariate_name"),
            covariate_provenance=arguments.get("covariate_provenance"),
            pre_treatment_only=bool(arguments.get("pre_treatment_only", True)),
        )
        return dict(result)

    if name == "nextstat_ads_cure_adjust":
        result = nextstat.ads.cure_adjust(
            arguments["control_outcomes"],
            arguments["control_covariates"],
            arguments["variant_outcomes"],
            arguments["variant_covariates"],
            covariate_names=arguments.get("covariate_names"),
            covariate_provenance=arguments.get("covariate_provenance"),
            pre_treatment_only=bool(arguments.get("pre_treatment_only", True)),
        )
        return dict(result)

    if name == "nextstat_kalman":
        F = arguments["F"]
        H = arguments["H"]
        Q = arguments["Q"]
        R = arguments["R"]
        x0 = arguments["x0"]
        P0 = arguments["P0"]
        operation = arguments.get("operation", "filter")
        n_ahead = int(arguments.get("n_ahead", 10))
        alpha = arguments.get("alpha")

        model = nextstat.KalmanModel(f=F, q=Q, h=H, r=R, m0=x0, p0=P0)

        if operation == "filter":
            y = arguments["y"]
            result = nextstat.kalman_filter(model, y)
        elif operation == "smooth":
            y = arguments["y"]
            result = nextstat.kalman_smooth(model, y)
        elif operation == "forecast":
            y = arguments["y"]
            forecast_kwargs = {"steps": n_ahead}
            if alpha is not None:
                forecast_kwargs["alpha"] = float(alpha)
            result = nextstat.kalman_forecast(model, y, **forecast_kwargs)
        elif operation == "simulate":
            simulate_kwargs = {
                "t_max": int(arguments["t_max"]),
                "seed": int(arguments.get("seed", 42)),
                "init": arguments.get("init", "sample"),
            }
            simulate_x0 = arguments.get("simulate_x0")
            if simulate_x0 is not None:
                simulate_kwargs["x0"] = [float(v) for v in simulate_x0]
            result = nextstat.kalman_simulate(model, **simulate_kwargs)
        elif operation == "em":
            y = arguments["y"]
            result = nextstat.kalman_em(
                model,
                y,
                max_iter=int(arguments.get("max_iter", 50)),
                tol=float(arguments.get("tol", 1e-6)),
                estimate_q=bool(arguments.get("estimate_q", True)),
                estimate_r=bool(arguments.get("estimate_r", True)),
                estimate_f=bool(arguments.get("estimate_f", False)),
                estimate_h=bool(arguments.get("estimate_h", False)),
                min_diag=float(arguments.get("min_diag", 1e-12)),
            )
            result = {
                "converged": bool(result["converged"]),
                "n_iter": int(result["n_iter"]),
                "loglik_trace": [float(v) for v in result["loglik_trace"]],
                "f": result["f"],
                "h": result["h"],
                "q": result["q"],
                "r": result["r"],
            }
        else:
            raise ValueError(f"Unknown kalman operation: {operation!r}")
        return dict(result)

    if name == "nextstat_churn_generate_data":
        result = nextstat.churn_generate_data(
            n_customers=int(arguments.get("n_customers", 2000)),
            n_cohorts=int(arguments.get("n_cohorts", 6)),
            max_time=float(arguments.get("max_time", 24.0)),
            treatment_fraction=float(arguments.get("treatment_fraction", 0.3)),
            seed=int(arguments.get("seed", 42)),
        )
        return dict(result)

    if name == "nextstat_churn_risk_model":
        times = [float(v) for v in arguments["times"]]
        events = _strict_event_bools(arguments["events"], "events")
        covariates = [[float(v) for v in row] for row in arguments["covariates"]]
        names = [str(v) for v in arguments["names"]]
        conf_level = float(arguments.get("conf_level", 0.95))
        if len(times) != len(events):
            raise ValueError("times and events must have the same length")
        if len(covariates) != len(times):
            raise ValueError("covariates must have the same length as times and events")
        if covariates and any(len(row) != len(covariates[0]) for row in covariates):
            raise ValueError("covariates must be a rectangular 2D array")
        if not covariates or len(names) != len(covariates[0]):
            raise ValueError("names must match the covariate column count")
        result = nextstat.churn_risk_model(
            times,
            events,
            covariates,
            names,
            conf_level=conf_level,
        )
        return dict(result)

    if name == "nextstat_churn_retention":
        times = arguments.get("times", arguments.get("tenure"))
        if times is None:
            raise ValueError("nextstat_churn_retention requires times (or legacy tenure)")
        raw_events = arguments.get("events", arguments.get("event"))
        if raw_events is None:
            raise ValueError("nextstat_churn_retention requires events (or legacy event)")
        groups = arguments.get("groups")
        if groups is None:
            raise ValueError("nextstat_churn_retention requires groups")
        conf_level = float(arguments.get("conf_level", 0.95))
        events = _strict_event_bools(raw_events, "events")

        result = nextstat.churn_retention(times, events, groups, conf_level=conf_level)
        return dict(result)

    if name == "nextstat_churn_compare":
        times = arguments["times"]
        groups = arguments["groups"]
        events = _strict_event_bools(arguments["events"], "events")
        conf_level = float(arguments.get("conf_level", 0.95))
        correction = arguments.get("correction", "benjamini_hochberg")
        if correction == "bh":
            correction = "benjamini_hochberg"
        alpha = float(arguments.get("alpha", 0.05))
        result = nextstat.churn_compare(
            times,
            events,
            groups,
            conf_level=conf_level,
            correction=correction,
            alpha=alpha,
        )
        return dict(result)

    if name == "nextstat_churn_diagnostics":
        times = [float(v) for v in arguments["times"]]
        events = _strict_event_bools(arguments["events"], "events")
        groups = [int(v) for v in arguments["groups"]]
        treated = _strict_binary_u8(arguments.get("treated", []), "treated")
        covariates = [[float(v) for v in row] for row in arguments.get("covariates", [])]
        covariate_names = [str(v) for v in arguments.get("covariate_names", [])]
        trim = float(arguments.get("trim", 0.01))

        if len(times) != len(events) or len(events) != len(groups):
            raise ValueError("times, events, and groups must have the same length")
        if treated and len(treated) != len(times):
            raise ValueError("treated must have the same length as times, events, and groups")
        if covariates and len(covariates) != len(times):
            raise ValueError("covariates must have the same length as times, events, and groups")
        if covariates and any(len(row) != len(covariates[0]) for row in covariates):
            raise ValueError("covariates must be a rectangular 2D array")
        if covariate_names and (not covariates or len(covariate_names) != len(covariates[0])):
            raise ValueError("covariate_names must match the covariate column count")

        result = nextstat.churn_diagnostics(
            times,
            events,
            groups,
            treated=treated,
            covariates=covariates,
            covariate_names=covariate_names,
            trim=trim,
        )
        return dict(result)

    if name == "nextstat_churn_cohort_matrix":
        times = [float(v) for v in arguments["times"]]
        events = _strict_event_bools(arguments["events"], "events")
        groups = [int(v) for v in arguments["groups"]]
        period_boundaries = [float(v) for v in arguments["period_boundaries"]]
        if len(times) != len(events) or len(events) != len(groups):
            raise ValueError("times, events, and groups must have the same length")
        if not period_boundaries:
            raise ValueError("period_boundaries must be non-empty")
        result = nextstat.churn_cohort_matrix(times, events, groups, period_boundaries)
        return dict(result)

    if name == "nextstat_churn_bootstrap_hr":
        times = [float(v) for v in arguments["times"]]
        events = _strict_event_bools(arguments["events"], "events")
        covariates = [[float(v) for v in row] for row in arguments["covariates"]]
        names = [str(v) for v in arguments["names"]]
        if len(times) != len(events):
            raise ValueError("times and events must have the same length")
        if len(covariates) != len(times):
            raise ValueError("covariates must have the same length as times and events")
        if covariates and any(len(row) != len(covariates[0]) for row in covariates):
            raise ValueError("covariates must be a rectangular 2D array")
        if not covariates or len(names) != len(covariates[0]):
            raise ValueError("names must match the covariate column count")
        result = nextstat.churn_bootstrap_hr(
            times,
            events,
            covariates,
            names,
            n_bootstrap=int(arguments.get("n_bootstrap", 1000)),
            seed=int(arguments.get("seed", 42)),
            conf_level=float(arguments.get("conf_level", 0.95)),
            ci_method=str(arguments.get("ci_method", "percentile")),
            n_jackknife=int(arguments.get("n_jackknife", 200)),
        )
        return dict(result)

    if name == "nextstat_churn_ingest":
        times = [float(v) for v in arguments["times"]]
        events = _strict_event_bools(arguments["events"], "events")
        groups_arg = arguments.get("groups")
        treated_arg = arguments.get("treated")
        covariates = [[float(v) for v in row] for row in arguments.get("covariates", [])]
        covariate_names = [str(v) for v in arguments.get("covariate_names", [])]
        observation_end = arguments.get("observation_end")
        groups = [int(v) for v in groups_arg] if groups_arg is not None else None
        treated = (
            _strict_binary_u8(treated_arg, "treated") if treated_arg is not None else None
        )

        if len(times) != len(events):
            raise ValueError("times and events must have the same length")
        if groups is not None and len(groups) != len(times):
            raise ValueError("groups must have the same length as times and events")
        if treated is not None and len(treated) != len(times):
            raise ValueError("treated must have the same length as times and events")
        if covariates and len(covariates) != len(times):
            raise ValueError("covariates must have the same length as times and events")
        if covariates and any(len(row) != len(covariates[0]) for row in covariates):
            raise ValueError("covariates must be a rectangular 2D array")
        if covariate_names and (not covariates or len(covariate_names) != len(covariates[0])):
            raise ValueError("covariate_names must match the covariate column count")

        result = nextstat.churn_ingest(
            times,
            events,
            groups=groups,
            treated=treated,
            covariates=covariates,
            covariate_names=covariate_names,
            observation_end=float(observation_end) if observation_end is not None else None,
        )
        return dict(result)

    if name == "nextstat_churn_uplift":
        times = [float(v) for v in arguments["times"]]
        events = _strict_event_bools(arguments["events"], "events")
        treated = _strict_binary_u8(arguments["treated"], "treated")
        covariates = [[float(v) for v in row] for row in arguments.get("covariates", [])]
        if len(times) != len(events) or len(events) != len(treated) or len(treated) != len(covariates):
            raise ValueError("times, events, treated, and covariates must have the same length")
        if covariates and any(len(row) != len(covariates[0]) for row in covariates):
            raise ValueError("covariates must be a rectangular 2D array")
        horizon = float(arguments.get("horizon", 12.0))
        result = nextstat.churn_uplift(times, events, treated, covariates, horizon=horizon)
        return dict(result)

    if name == "nextstat_churn_uplift_survival":
        times = [float(v) for v in arguments["times"]]
        events = _strict_event_bools(arguments["events"], "events")
        treated = _strict_binary_u8(arguments["treated"], "treated")
        covariates = [[float(v) for v in row] for row in arguments.get("covariates", [])]
        if len(times) != len(events) or len(events) != len(treated):
            raise ValueError("times, events, and treated must have the same length")
        if covariates and len(covariates) != len(times):
            raise ValueError("covariates must have the same length as times, events, and treated")
        if covariates and any(len(row) != len(covariates[0]) for row in covariates):
            raise ValueError("covariates must be a rectangular 2D array")
        horizon = float(arguments.get("horizon", 12.0))
        eval_horizons = [float(v) for v in arguments.get("eval_horizons", [3.0, 6.0, 12.0, 24.0])]
        trim = float(arguments.get("trim", 0.01))
        result = nextstat.churn_uplift_survival(
            times,
            events,
            treated,
            covariates=covariates,
            horizon=horizon,
            eval_horizons=eval_horizons,
            trim=trim,
        )
        return dict(result)

    if name == "nextstat_chain_ladder":
        triangle = _normalize_chain_ladder_triangle(arguments["triangle"])
        method = arguments.get("method", "mack")
        conf_level = float(arguments.get("conf_level", 0.95))

        if method == "mack":
            result = nextstat.mack_chain_ladder(triangle, conf_level=conf_level)
        elif method in ("basic", "chain_ladder"):
            result = nextstat.chain_ladder(triangle)
        else:
            raise ValueError(f"Unknown chain ladder method: {method!r}")
        return dict(result)

    # -------------------------------------------------------------------
    # Pharma / NLME tools
    # -------------------------------------------------------------------

    if name == "nextstat_pharma_fit":
        method = arguments.get("method", "focei")
        model_type = arguments.get("model", "1cpt_oral")
        error_model = arguments.get("error_model", "proportional")
        sigma = float(arguments.get("sigma", 0.1))
        sigma_add = arguments.get("sigma_add")
        bioavailability = float(arguments.get("bioavailability", 1.0))
        theta_init = arguments["theta_init"]
        omega_init = arguments["omega_init"]

        common = dict(
            times=arguments["times"],
            y=arguments["y"],
            subject_idx=arguments["subject_idx"],
            n_subjects=int(arguments["n_subjects"]),
            model=model_type,
            doses=arguments["doses"],
            bioavailability=bioavailability,
            error_model=error_model,
            sigma=sigma,
            sigma_add=(float(sigma_add) if sigma_add is not None else None),
            theta_init=theta_init,
            omega_init=omega_init,
        )

        if method == "saem":
            result = nextstat.nlme_saem(**common)
        else:
            common["method"] = method
            result = nextstat.nlme_foce(**common)
        return dict(result)

    if name == "nextstat_pharma_vpc":
        sigma_add = arguments.get("sigma_add")
        result = nextstat.pk_vpc(
            times=arguments["times"],
            y=arguments["y"],
            subject_idx=arguments["subject_idx"],
            n_subjects=int(arguments["n_subjects"]),
            model=arguments["model"],
            doses=arguments["doses"],
            bioavailability=float(arguments.get("bioavailability", 1.0)),
            theta=arguments["theta"],
            omega_matrix=arguments["omega_matrix"],
            sigma=float(arguments["sigma"]),
            sigma_add=(float(sigma_add) if sigma_add is not None else None),
            error_model=arguments.get("error_model", "proportional"),
            n_sim=int(arguments.get("n_sim", 200)),
            quantiles=arguments.get("quantiles"),
            n_bins=int(arguments.get("n_bins", 10)),
            seed=int(arguments.get("seed", 42)),
            pi_level=float(arguments.get("pi_level", 0.90)),
        )
        return dict(result)

    if name == "nextstat_pk_gof":
        sigma_add = arguments.get("sigma_add")
        records = nextstat.pk_gof(
            times=arguments["times"],
            y=arguments["y"],
            subject_idx=arguments["subject_idx"],
            model=arguments.get("model", "1cpt_oral"),
            doses=arguments["doses"],
            bioavailability=float(arguments.get("bioavailability", 1.0)),
            theta=arguments["theta"],
            eta=arguments["eta"],
            error_model=arguments.get("error_model", "proportional"),
            sigma=float(arguments.get("sigma", 0.1)),
            sigma_add=(float(sigma_add) if sigma_add is not None else None),
        )
        return {
            "model": arguments.get("model", "1cpt_oral"),
            "n_subjects": len(arguments["eta"]),
            "n_records": len(records),
            "records": [dict(record) for record in records],
        }

    if name == "nextstat_pk_npde":
        sigma_add = arguments.get("sigma_add")
        result = nextstat.pk_npde(
            times=arguments["times"],
            y=arguments["y"],
            subject_idx=arguments["subject_idx"],
            n_subjects=int(arguments["n_subjects"]),
            model=arguments.get("model", "1cpt_oral"),
            doses=arguments["doses"],
            bioavailability=float(arguments.get("bioavailability", 1.0)),
            theta=arguments["theta"],
            omega_matrix=arguments["omega_matrix"],
            error_model=arguments.get("error_model", "proportional"),
            sigma=float(arguments.get("sigma", 0.1)),
            sigma_add=(float(sigma_add) if sigma_add is not None else None),
            n_sim=int(arguments.get("n_sim", 500)),
            seed=int(arguments.get("seed", 42)),
        )
        return {
            "model": arguments.get("model", "1cpt_oral"),
            "n_subjects": int(arguments["n_subjects"]),
            "n_records": len(result["records"]),
            "n_sim": int(arguments.get("n_sim", 500)),
            "seed": int(arguments.get("seed", 42)),
            "records": [dict(record) for record in result["records"]],
            "mean": float(result["mean"]),
            "variance": float(result["variance"]),
        }

    if name == "nextstat_trial_simulate":
        result = nextstat.simulate_trial(
            n_subjects=int(arguments["n_subjects"]),
            dose=float(arguments["dose"]),
            obs_times=arguments["obs_times"],
            pk_model=arguments.get("pk_model", "1cpt_oral"),
            theta=arguments["theta"],
            omega=arguments["omega"],
            sigma=float(arguments["sigma"]),
            error_model=arguments.get("error_model", "proportional"),
            bioavailability=float(arguments.get("bioavailability", 1.0)),
            seed=int(arguments.get("seed", 42)),
        )
        return dict(result)

    if name == "nextstat_bioequivalence":
        op = arguments.get("operation", "test")
        if op == "test":
            result = nextstat.average_be(
                arguments["test_values"],
                arguments["ref_values"],
            )
            return dict(result)
        elif op == "power":
            power = nextstat.be_power(
                int(arguments["n_total"]),
                cv=float(arguments.get("cv", 0.30)),
                gmr=float(arguments.get("gmr", 0.95)),
            )
            return {"power": float(power)}
        elif op == "sample_size":
            result = nextstat.be_sample_size(
                cv=float(arguments.get("cv", 0.30)),
                gmr=float(arguments.get("gmr", 0.95)),
                target_power=float(arguments.get("target_power", 0.80)),
            )
            return dict(result)
        else:
            raise ValueError(f"Unknown BE operation: {op!r}")

    # -------------------------------------------------------------------
    # Safety / Reliability tools
    # -------------------------------------------------------------------

    if name == "nextstat_fault_tree_mc":
        result = nextstat.fault_tree_mc(
            arguments["spec"],
            int(arguments.get("n_scenarios", 1_000_000)),
            seed=int(arguments.get("seed", 42)),
            device=arguments.get("device", "cpu"),
        )
        return dict(result)

    if name == "nextstat_fault_tree_ce_is":
        core = getattr(nextstat, "_core", None)
        if core is None:
            raise RuntimeError("nextstat._core is unavailable")
        result = core.fault_tree_mc_ce_is(
            arguments["spec"],
            n_per_level=int(arguments.get("n_per_level", 10_000)),
            elite_fraction=float(arguments.get("elite_fraction", 0.01)),
            max_levels=int(arguments.get("max_levels", 20)),
            q_max=float(arguments.get("q_max", 0.99)),
            seed=int(arguments.get("seed", 42)),
        )
        return dict(result)

    # -------------------------------------------------------------------
    # Dose-response tools
    # -------------------------------------------------------------------

    if name == "nextstat_dose_response":
        core = getattr(nextstat, "_core", None)
        if core is None:
            raise RuntimeError("nextstat._core is unavailable")
        model_type = arguments["model"]
        e0 = float(arguments["e0"])
        emax = float(arguments["emax"])
        ec50 = float(arguments["ec50"])
        conc = arguments.get("conc", arguments.get("dose"))
        obs = arguments.get("obs", arguments.get("response"))
        error_model = arguments.get("error_model", "additive")
        sigma = float(arguments.get("sigma", 0.05))
        sigma_add_raw = arguments.get("sigma_add")
        sigma_add = None if sigma_add_raw is None else float(sigma_add_raw)

        if conc is None:
            raise ValueError("nextstat_dose_response requires conc (or alias dose)")

        if model_type == "emax":
            if obs is not None:
                nll = core.emax_nll(
                    e0,
                    emax,
                    ec50,
                    conc,
                    obs,
                    error_model=error_model,
                    sigma=sigma,
                    sigma_add=sigma_add,
                )
                return {"model": "emax", "nll": float(nll)}
            pred = core.emax_predict(e0, emax, ec50, conc)
            return {"model": "emax", **dict(pred)}
        elif model_type == "sigmoid_emax":
            gamma = float(arguments.get("gamma", 1.0))
            if obs is not None:
                nll = core.sigmoid_emax_nll(
                    e0,
                    emax,
                    ec50,
                    gamma,
                    conc,
                    obs,
                    error_model=error_model,
                    sigma=sigma,
                    sigma_add=sigma_add,
                )
                return {"model": "sigmoid_emax", "nll": float(nll)}
            pred = core.sigmoid_emax_predict(e0, emax, ec50, gamma, conc)
            return {"model": "sigmoid_emax", **dict(pred)}
        else:
            raise ValueError(f"Unknown dose-response model: {model_type!r}")

    # -------------------------------------------------------------------
    # Competing risks tools
    # -------------------------------------------------------------------

    if name == "nextstat_competing_risks":
        core = getattr(nextstat, "_core", None)
        if core is None:
            raise RuntimeError("nextstat._core is unavailable")
        op = arguments["operation"]
        times = arguments["times"]
        events = arguments["events"]
        target_cause = int(arguments.get("target_cause", 1))
        conf_level = float(arguments.get("conf_level", 0.95))

        if op == "cif":
            result = core.cumulative_incidence(
                times,
                events,
                target_cause,
                conf_level=conf_level,
            )
            return dict(result)
        elif op == "gray_test":
            groups = arguments["groups"]
            result = core.gray_test(times, events, groups, target_cause)
            return dict(result)
        elif op == "fine_gray":
            x = arguments["x"]
            p = int(arguments["p"])
            result = core.fine_gray_fit(times, events, x, p, target_cause)
            return dict(result)
        else:
            raise ValueError(f"Unknown competing risks operation: {op!r}")

    # -------------------------------------------------------------------
    # Econometrics: event study
    # -------------------------------------------------------------------

    if name == "nextstat_event_study":
        import nextstat.econometrics as econ

        treat_time = arguments["treat_time"]
        treat = [0 if value is None else 1 for value in treat_time]
        event_time = [0 if value is None else int(value) for value in treat_time]

        result = econ.event_study_twfe_fit(
            y=arguments["y"],
            treat=treat,
            entity=arguments["entity"],
            time=arguments["time"],
            event_time=event_time,
            window=(-int(arguments.get("n_leads", 3)), int(arguments.get("n_lags", 3))),
            reference=-1,
            cluster=arguments.get("cluster", "entity"),
        )
        return {
            "rel_times": list(result.rel_times),
            "coef": list(result.coef),
            "standard_errors": list(result.standard_errors),
            "covariance": [list(row) for row in result.covariance],
            "reference": result.reference,
            "n_obs": result.n_obs,
            "n_entities": result.n_entities,
            "n_times": result.n_times,
            "cluster": result.cluster,
        }

    # -------------------------------------------------------------------
    # Volatility tools
    # -------------------------------------------------------------------

    if name == "nextstat_garch_fit":
        import nextstat.volatility as vol

        returns = arguments["returns"]
        model_type = arguments.get("model", "garch")

        if model_type == "garch":
            result = vol.garch(returns)
        elif model_type == "egarch":
            result = vol.egarch(returns)
        elif model_type == "gjr_garch":
            result = vol.gjr_garch(returns)
        else:
            raise ValueError(f"Unknown GARCH model: {model_type!r}")
        return dict(result)

    raise ValueError(f"Unknown tool: {name!r}. Available: {get_tool_names()}")


def execute_tool_raw(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a NextStat tool by name, returning the raw tool result (no envelope)."""
    import nextstat

    meta = _apply_execution(nextstat, name, arguments)
    try:
        return _execute_tool_impl(nextstat, name, arguments)
    finally:
        _restore_execution(nextstat, meta)


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    transport: str = "local",
    server_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout_s: float = 30.0,
    fallback_to_local: bool = True,
) -> dict[str, Any]:
    """Execute a NextStat tool call and return a stable response envelope.

    Args:
        transport:
            - ``"local"`` (default): execute in-process via Python bindings.
            - ``"server"``: execute over HTTP via ``nextstat-server`` at ``POST /v1/tools/execute``.
        server_url: Base URL for server mode, e.g. ``"http://127.0.0.1:3742"``.
        api_key: Optional bearer token for auth-enabled server mode. If omitted, the
            client will also check ``NEXTSTAT_TOOLS_API_KEY`` and ``NEXTSTAT_SERVER_API_KEY``.
        timeout_s: HTTP timeout (server mode only).
        fallback_to_local: If true (default), network/transport failures in server mode fall back
            to local execution (if available). HTTP auth/rate-limit failures and invalid server
            envelopes do not silently fall back.
    """
    if transport == "server":
        server_url = _resolve_server_url(server_url)
        if not server_url:
            raise ValueError(
                "server_url is required for transport='server' "
                "(or set NEXTSTAT_SERVER_URL / NEXTSTAT_TOOLS_SERVER_URL)"
            )
        api_key = _resolve_server_api_key(api_key)
        try:
            out = _http_json_post(
                f"{server_url.rstrip('/')}/v1/tools/execute",
                {"name": name, "arguments": arguments},
                timeout_s=timeout_s,
                api_key=api_key,
            )
            if not isinstance(out, dict) or out.get("schema_version") != "nextstat.tool_result.v1":
                raise InvalidServerResponseError(
                    "Invalid server tool response (missing tool_result.v1 envelope)"
                )
            return out
        except Exception as e:
            if fallback_to_local and isinstance(e, ServerTransportError):
                try:
                    local = execute_tool(name, arguments, transport="local")
                    meta = local.get("meta")
                    if isinstance(meta, dict):
                        warnings = meta.get("warnings")
                        if not isinstance(warnings, list):
                            warnings = []
                        warnings.append(
                            f"server transport failed ({server_url}): {e.__class__.__name__}: {e}; fell back to local"
                        )
                        meta["warnings"] = warnings
                    return local
                except Exception:
                    # Fall through to an error envelope if local execution is unavailable.
                    pass
            return {
                "schema_version": "nextstat.tool_result.v1",
                "ok": False,
                "result": None,
                "error": {"type": e.__class__.__name__, "message": str(e)},
                "meta": {"tool_name": name, "nextstat_version": None, "warnings": [f"server_url={server_url}"]},
            }

    if transport != "local":
        raise ValueError(f"Unknown transport: {transport!r}. Use 'local' or 'server'.")

    import nextstat

    envelope: dict[str, Any] = {
        "schema_version": "nextstat.tool_result.v1",
        "ok": False,
        "result": None,
        "error": None,
        "meta": {
            "tool_name": name,
            "nextstat_version": getattr(nextstat, "__version__", None),
        },
    }

    exec_meta: dict[str, Any] | None = None
    try:
        exec_meta = _apply_execution(nextstat, name, arguments)
        envelope["result"] = _execute_tool_impl(nextstat, name, arguments)
        envelope["ok"] = True
    except Exception as e:
        envelope["error"] = {"type": e.__class__.__name__, "message": str(e)}
    finally:
        if exec_meta is not None:
            _restore_execution(nextstat, exec_meta)

    if exec_meta is not None:
        envelope["meta"]["deterministic"] = exec_meta.get("deterministic")
        ev = exec_meta.get("eval_mode_effective")
        if ev is not None:
            envelope["meta"]["eval_mode"] = ev
        tr = exec_meta.get("threads_requested")
        if tr is not None:
            envelope["meta"]["threads_requested"] = tr
        ta = exec_meta.get("threads_applied")
        if ta is not None:
            envelope["meta"]["threads_applied"] = ta

    return envelope


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
