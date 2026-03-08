import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_fixture_text(rel: str) -> str:
    return (_repo_root() / rel).read_text(encoding="utf-8")


class _ToolStubHandler(BaseHTTPRequestHandler):
    # Injected by server factory:
    execute_mode = "ok"  # "ok" | "http_500" | "invalid_envelope"
    schema_mode = "ok"  # "ok" | "invalid_descriptor"
    require_auth = False
    expected_token = "secret-token"

    def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover
        # Keep pytest output clean.
        return

    def _check_auth(self) -> bool:
        if not self.require_auth:
            return True
        if self.headers.get("Authorization") == f"Bearer {self.expected_token}":
            return True
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": "invalid api key"}).encode("utf-8"))
        return False

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/v1/tools/schema":
            self.send_response(404)
            self.end_headers()
            return
        if not self._check_auth():
            return

        if self.schema_mode == "invalid_descriptor":
            payload = {"schema_version": "wrong", "transport": "server", "tools": []}
        else:
            payload = json.loads(
                _load_fixture_text("docs/specs/nextstat_tool_schema_server_v1.example.json")
            )

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/tools/execute":
            self.send_response(404)
            self.end_headers()
            return
        if not self._check_auth():
            return

        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n).decode("utf-8") if n else ""
        payload = json.loads(raw) if raw else {}

        if self.execute_mode == "http_500":
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"detail": "boom"}).encode("utf-8"))
            return

        # Validate request shape.
        assert payload.get("name") == "nextstat_fit"
        assert isinstance(payload.get("arguments"), dict)
        assert payload["arguments"].get("workspace_json") == "ws"

        if self.execute_mode == "invalid_envelope":
            out = {"ok": True, "result": {"from": "server"}}
        else:
            out = {
                "schema_version": "nextstat.tool_result.v1",
                "ok": True,
                "result": {"from": "server"},
                "error": None,
                "meta": {"tool_name": "nextstat_fit", "nextstat_version": "server"},
            }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(out).encode("utf-8"))


def _serve(
    *,
    execute_mode: str = "ok",
    schema_mode: str = "ok",
    require_auth: bool = False,
    expected_token: str = "secret-token",
) -> tuple[HTTPServer, threading.Thread]:
    handler = type(
        "H",
        (_ToolStubHandler,),
        {
            "execute_mode": execute_mode,
            "schema_mode": schema_mode,
            "require_auth": require_auth,
            "expected_token": expected_token,
        },
    )
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


def _unused_local_url() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def test_get_toolkit_descriptor_server_transport_success():
    pytest.importorskip("nextstat")
    from nextstat.tools import get_toolkit_descriptor

    httpd, _t = _serve()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        descriptor = get_toolkit_descriptor(transport="server", server_url=url)
        assert descriptor["schema_version"] == "nextstat.tool_schema.v1"
        assert descriptor["transport"] == "server"
        assert isinstance(descriptor["tools"], list) and descriptor["tools"]
        assert isinstance(descriptor["capabilities"], list) and descriptor["capabilities"]
        assert isinstance(descriptor["guidance"], dict)
        assert isinstance(descriptor["guidance"]["hints"], list) and descriptor["guidance"]["hints"]
        assert isinstance(descriptor["guidance"]["recipes"], list) and descriptor["guidance"]["recipes"]
        assert all(recipe["transport"] == "server" for recipe in descriptor["guidance"]["recipes"])
    finally:
        httpd.shutdown()


def test_get_toolkit_server_transport_returns_descriptor_tools():
    pytest.importorskip("nextstat")
    from nextstat.tools import get_toolkit

    httpd, _t = _serve()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        tools = get_toolkit(transport="server", server_url=url)
        expected = json.loads(
            _load_fixture_text("docs/specs/nextstat_tool_schema_server_v1.example.json")
        )["tools"]
        assert tools == expected
    finally:
        httpd.shutdown()


def test_get_toolkit_descriptor_server_transport_auth_via_env(monkeypatch):
    pytest.importorskip("nextstat")
    from nextstat.tools import get_toolkit_descriptor

    httpd, _t = _serve(require_auth=True, expected_token="env-secret")
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        monkeypatch.setenv("NEXTSTAT_TOOLS_SERVER_URL", url)
        monkeypatch.setenv("NEXTSTAT_TOOLS_API_KEY", "env-secret")
        descriptor = get_toolkit_descriptor(transport="server")
        assert descriptor["transport"] == "server"
    finally:
        httpd.shutdown()


def test_get_toolkit_descriptor_server_transport_invalid_descriptor_raises():
    pytest.importorskip("nextstat")
    from nextstat.tools import get_toolkit_descriptor

    httpd, _t = _serve(schema_mode="invalid_descriptor")
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        with pytest.raises(RuntimeError, match="Invalid tool schema response"):
            get_toolkit_descriptor(transport="server", server_url=url)
    finally:
        httpd.shutdown()


def test_execute_tool_server_transport_success():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    httpd, _t = _serve()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        out = execute_tool("nextstat_fit", {"workspace_json": "ws"}, transport="server", server_url=url)
        assert out.get("schema_version") == "nextstat.tool_result.v1"
        assert out.get("ok") is True
        assert out.get("result") == {"from": "server"}
        assert out.get("meta", {}).get("nextstat_version") == "server"
    finally:
        httpd.shutdown()


def test_execute_tool_server_transport_success_with_auth():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    httpd, _t = _serve(require_auth=True, expected_token="explicit-secret")
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        out = execute_tool(
            "nextstat_fit",
            {"workspace_json": "ws"},
            transport="server",
            server_url=url,
            api_key="explicit-secret",
        )
        assert out.get("ok") is True
        assert out.get("result") == {"from": "server"}
    finally:
        httpd.shutdown()


def test_execute_tool_server_transport_falls_back_to_local_on_transport_error():
    nextstat = pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    url = _unused_local_url()
    ws = _load_fixture_text("tests/fixtures/simple_workspace.json")
    out = execute_tool(
        "nextstat_workspace_audit",
        {"workspace_json": ws, "execution": {"deterministic": True}},
        transport="server",
        server_url=url,
        timeout_s=1.0,
        fallback_to_local=True,
    )
    assert out.get("schema_version") == "nextstat.tool_result.v1"
    assert out.get("ok") is True
    assert out.get("meta", {}).get("nextstat_version") == getattr(nextstat, "__version__", None)
    warnings = out.get("meta", {}).get("warnings")
    assert isinstance(warnings, list) and warnings, "expected fallback warning in meta.warnings"
    assert any("fell back to local" in str(w) for w in warnings)


def test_execute_tool_server_transport_http_auth_error_does_not_fallback():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    httpd, _t = _serve(require_auth=True, expected_token="secret-token")
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        out = execute_tool(
            "nextstat_fit",
            {"workspace_json": "ws"},
            transport="server",
            server_url=url,
            fallback_to_local=True,
        )
        assert out.get("schema_version") == "nextstat.tool_result.v1"
        assert out.get("ok") is False
        assert out.get("meta", {}).get("nextstat_version") is None
        assert out.get("error", {}).get("type") == "ServerHTTPError"
        assert "HTTP 401" in out.get("error", {}).get("message", "")
    finally:
        httpd.shutdown()


def test_execute_tool_server_transport_invalid_envelope_does_not_fallback():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    httpd, _t = _serve(execute_mode="invalid_envelope")
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        out = execute_tool(
            "nextstat_fit",
            {"workspace_json": "ws"},
            transport="server",
            server_url=url,
            fallback_to_local=True,
        )
        assert out.get("schema_version") == "nextstat.tool_result.v1"
        assert out.get("ok") is False
        assert out.get("meta", {}).get("nextstat_version") is None
        assert out.get("error", {}).get("type") == "InvalidServerResponseError"
        assert "tool_result.v1 envelope" in out.get("error", {}).get("message", "")
    finally:
        httpd.shutdown()
