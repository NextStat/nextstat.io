import json
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
    mode = "ok"  # "ok" | "fail"

    def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover
        # Keep pytest output clean.
        return

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/tools/execute":
            self.send_response(404)
            self.end_headers()
            return

        n = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(n).decode("utf-8") if n else ""
        payload = json.loads(raw) if raw else {}

        if self.mode == "fail":
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"detail": "boom"}).encode("utf-8"))
            return

        # Validate request shape.
        assert payload.get("name") == "nextstat_fit"
        assert isinstance(payload.get("arguments"), dict)
        assert payload["arguments"].get("workspace_json") == "ws"

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


def _serve(mode: str) -> tuple[HTTPServer, threading.Thread]:
    handler = type("H", (_ToolStubHandler,), {"mode": mode})
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


def test_execute_tool_server_transport_success():
    pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    httpd, _t = _serve("ok")
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        out = execute_tool("nextstat_fit", {"workspace_json": "ws"}, transport="server", server_url=url)
        assert out.get("schema_version") == "nextstat.tool_result.v1"
        assert out.get("ok") is True
        assert out.get("result") == {"from": "server"}
        assert out.get("meta", {}).get("nextstat_version") == "server"
    finally:
        httpd.shutdown()


def test_execute_tool_server_transport_falls_back_to_local():
    nextstat = pytest.importorskip("nextstat")
    from nextstat.tools import execute_tool

    # Serve HTTP 500 so the client hits its fallback path quickly.
    httpd, _t = _serve("fail")
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        ws = _load_fixture_text("tests/fixtures/simple_workspace.json")
        out = execute_tool(
            "nextstat_workspace_audit",
            {"workspace_json": ws, "execution": {"deterministic": True}},
            transport="server",
            server_url=url,
            timeout_s=2.0,
            fallback_to_local=True,
        )
        assert out.get("schema_version") == "nextstat.tool_result.v1"
        assert out.get("ok") is True
        assert out.get("meta", {}).get("nextstat_version") == getattr(nextstat, "__version__", None)
        warnings = out.get("meta", {}).get("warnings")
        assert isinstance(warnings, list) and warnings, "expected fallback warning in meta.warnings"
    finally:
        httpd.shutdown()

