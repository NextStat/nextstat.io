from __future__ import annotations

from typing import Any


def _assert_workspace_output_renderer_copilot_instructions(content: str) -> None:
    assert "NextStat Copilot Instructions" in content
    assert "descriptor.tools" in content
    assert "fallback_to_local=False" in content
    assert "local_root_ingest_and_workspace_triage" in content
    assert "server_root_ingest_and_workspace_triage" in content


def _assert_workspace_output_renderer_cursor_rule(content: str) -> None:
    assert content.startswith("---\n")
    assert "NextStat Tool Surface" in content
    assert "descriptor.tools" in content
    assert "server_root_ingest_and_workspace_triage" in content


WORKSPACE_OUTPUT_RENDERER_ASSERTIONS = {
    "copilot_instructions": _assert_workspace_output_renderer_copilot_instructions,
    "cursor_rule": _assert_workspace_output_renderer_cursor_rule,
}


def assert_workspace_output_renderer(renderer: str, *, content: str) -> None:
    fn = WORKSPACE_OUTPUT_RENDERER_ASSERTIONS.get(renderer)
    assert callable(fn), f"missing workspace-output assertion handler for renderer={renderer}"
    fn(content)


def _assert_runnable_payload_renderer_provider_example(
    *, payload: dict[str, Any], client: str, transport: str, runnable: dict[str, Any], payload_key: str
) -> None:
    assert payload["provider"] == client
    assert payload["transport"] == transport
    assert isinstance(payload[payload_key], list) and payload[payload_key]
    assert isinstance(payload[runnable["instruction_key"]], str) and payload[runnable["instruction_key"]]
    assert isinstance(payload[runnable["prompt_key"]], str) and payload[runnable["prompt_key"]]


def _assert_runnable_payload_renderer_anthropic_example(
    *, payload: dict[str, Any], client: str, transport: str, runnable: dict[str, Any], payload_key: str
) -> None:
    assert payload["provider"] == client
    assert payload["transport"] == transport
    assert isinstance(payload[payload_key], list) and payload[payload_key]
    assert isinstance(payload[runnable["instruction_key"]], str) and payload[runnable["instruction_key"]]
    assert isinstance(payload[runnable["prompt_key"]], list) and payload[runnable["prompt_key"]]
    first_message = payload[runnable["prompt_key"]][0]
    assert first_message["role"] == "user"
    assert isinstance(first_message["content"], str) and first_message["content"]


def _assert_runnable_payload_renderer_mcp_example(
    *, payload: dict[str, Any], client: str, transport: str, runnable: dict[str, Any], payload_key: str
) -> None:
    assert payload["client"] == client
    assert payload["transport"] == transport
    assert isinstance(payload[payload_key], list) and payload[payload_key]
    assert isinstance(payload["dispatch"], dict) and payload["dispatch"]["mode"]
    assert payload["tool_server_name"]


RUNNABLE_PAYLOAD_RENDERER_ASSERTIONS = {
    "provider_example": _assert_runnable_payload_renderer_provider_example,
    "anthropic_example": _assert_runnable_payload_renderer_anthropic_example,
    "mcp_example": _assert_runnable_payload_renderer_mcp_example,
}


def assert_runnable_payload_renderer(
    renderer: str, *, payload: dict[str, Any], client: str, transport: str, runnable: dict[str, Any], payload_key: str
) -> None:
    fn = RUNNABLE_PAYLOAD_RENDERER_ASSERTIONS.get(renderer)
    assert callable(fn), f"missing runnable-payload assertion handler for renderer={renderer}"
    fn(payload=payload, client=client, transport=transport, runnable=runnable, payload_key=payload_key)


def _assert_live_provider_payload_renderer_provider_example(
    *, payload: dict[str, Any], client: str, runnable: dict[str, Any], payload_key: str
) -> None:
    assert payload["provider"] == client
    assert payload["transport"] == "server"
    assert payload["model"]
    assert payload["recipe_ids"]
    assert payload["guidance_hints"]
    assert isinstance(payload[payload_key], list) and payload[payload_key]
    assert isinstance(payload[str(runnable["instruction_key"])], str) and payload[str(runnable["instruction_key"])]
    assert isinstance(payload[str(runnable["prompt_key"])], str) and payload[str(runnable["prompt_key"])]


def _assert_live_provider_payload_renderer_anthropic_example(
    *, payload: dict[str, Any], client: str, runnable: dict[str, Any], payload_key: str
) -> None:
    assert payload["provider"] == client
    assert payload["transport"] == "server"
    assert payload["model"]
    assert payload["recipe_ids"]
    assert payload["guidance_hints"]
    assert isinstance(payload[payload_key], list) and payload[payload_key]
    assert isinstance(payload[str(runnable["instruction_key"])], str) and payload[str(runnable["instruction_key"])]
    assert isinstance(payload[str(runnable["prompt_key"])], list) and payload[str(runnable["prompt_key"])]
    first_message = payload[str(runnable["prompt_key"])][0]
    assert first_message["role"] == "user"
    assert isinstance(first_message["content"], str) and first_message["content"]


def _assert_live_provider_payload_renderer_mcp_example(
    *, payload: dict[str, Any], client: str, runnable: dict[str, Any], payload_key: str
) -> None:
    assert payload["client"] == client
    assert payload["transport"] == "server"
    assert payload["tool_server_name"] == "nextstat-server"
    assert payload["recipe_ids"]
    assert payload["guidance_hints"]
    assert isinstance(payload["dispatch"], dict) and payload["dispatch"]["mode"] == "execute_tool_server"
    assert isinstance(payload[payload_key], list) and payload[payload_key]


LIVE_PROVIDER_PAYLOAD_RENDERER_ASSERTIONS = {
    "provider_example": _assert_live_provider_payload_renderer_provider_example,
    "anthropic_example": _assert_live_provider_payload_renderer_anthropic_example,
    "mcp_example": _assert_live_provider_payload_renderer_mcp_example,
}


def assert_live_provider_payload_renderer(
    renderer: str, *, payload: dict[str, Any], client: str, runnable: dict[str, Any], payload_key: str
) -> None:
    fn = LIVE_PROVIDER_PAYLOAD_RENDERER_ASSERTIONS.get(renderer)
    assert callable(fn), f"missing live-provider assertion handler for renderer={renderer}"
    fn(payload=payload, client=client, runnable=runnable, payload_key=payload_key)


def _tool_names_for_live_provider_payload_renderer_provider_example(
    *, payload: dict[str, Any], payload_key: str
) -> list[str]:
    return [tool["function"]["name"] for tool in payload[payload_key]]


def _tool_names_for_live_provider_payload_renderer_anthropic_example(
    *, payload: dict[str, Any], payload_key: str
) -> list[str]:
    return [tool["function"]["name"] for tool in payload[payload_key]]


def _tool_names_for_live_provider_payload_renderer_mcp_example(
    *, payload: dict[str, Any], payload_key: str
) -> list[str]:
    return [tool["name"] for tool in payload[payload_key]]


LIVE_PROVIDER_TOOL_NAME_EXTRACTORS = {
    "provider_example": _tool_names_for_live_provider_payload_renderer_provider_example,
    "anthropic_example": _tool_names_for_live_provider_payload_renderer_anthropic_example,
    "mcp_example": _tool_names_for_live_provider_payload_renderer_mcp_example,
}


def tool_names_for_live_provider_payload(
    renderer: str, *, payload: dict[str, Any], payload_key: str
) -> list[str]:
    fn = LIVE_PROVIDER_TOOL_NAME_EXTRACTORS.get(renderer)
    assert callable(fn), f"missing live-provider tool-name handler for renderer={renderer}"
    return fn(payload=payload, payload_key=payload_key)
