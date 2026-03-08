import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class _ScriptToolStubHandler(BaseHTTPRequestHandler):
    require_auth = True
    expected_token = "script-secret"
    call_log: list[dict[str, object]] = []

    def log_message(self, fmt: str, *args: object) -> None:  # pragma: no cover
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

        payload = json.loads(
            (_repo_root() / "docs" / "specs" / "nextstat_tool_schema_server_v1.example.json").read_text(
                encoding="utf-8"
            )
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
        self.call_log.append(payload)

        name = payload.get("name")
        args = payload.get("arguments") or {}
        response = _tool_response(name, args)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))


def _tool_response(name: str, args: dict) -> dict:
    meta = {
        "tool_name": name,
        "nextstat_version": "server",
        "deterministic": True,
        "eval_mode": "parity",
        "threads_requested": 1,
        "warnings": [],
    }
    result: dict[str, object]
    if name == "nextstat_workspace_audit":
        result = {"channels": 1, "samples": 2, "warnings": []}
    elif name == "nextstat_fit":
        result = {
            "nll": 10.0,
            "converged": True,
            "n_iter": 7,
            "poi_index": 0,
            "poi_value": 1.25,
            "poi_error": 0.2,
            "parameters": {"mu": {"value": 1.25, "error": 0.2}},
            "wall_time_s": 0.01,
        }
    elif name == "nextstat_discovery_asymptotic":
        result = {
            "mu_hat": 1.25,
            "nll_hat": 10.0,
            "nll_mu0": 11.125,
            "q0": 2.25,
            "z0": 1.5,
            "p0": 0.0668,
            "wall_time_s": 0.01,
        }
    elif name == "nextstat_hypotest":
        result = {
            "mu": float(args.get("mu", 1.0)),
            "cls": 0.12,
            "clsb": 0.08,
            "clb": 0.67,
            "wall_time_s": 0.01,
        }
    elif name == "nextstat_upper_limit":
        expected = bool(args.get("expected", False))
        result = {"obs_limit": 1.9, "wall_time_s": 0.01}
        if expected:
            result["exp_limits"] = [1.1, 1.4, 1.8, 2.2, 2.7]
    elif name == "nextstat_scan":
        result = {
            "poi_index": 0,
            "mu_hat": 1.25,
            "nll_hat": 10.0,
            "mu_values": [0.0, 0.5, 1.0],
            "points": [
                {"mu": 0.0, "q_mu": 2.0, "nll_mu": 11.0, "converged": True, "n_iter": 5},
                {"mu": 0.5, "q_mu": 0.5, "nll_mu": 10.25, "converged": True, "n_iter": 5},
                {"mu": 1.0, "q_mu": 0.1, "nll_mu": 10.05, "converged": True, "n_iter": 5},
            ],
            "wall_time_s": 0.01,
        }
    elif name == "nextstat_ranking":
        result = {
            "ranking": [
                {
                    "name": "syst_a",
                    "delta_mu_up": 0.2,
                    "delta_mu_down": -0.1,
                    "total_impact": 0.3,
                    "pull": 0.0,
                    "constraint": 1.0,
                    "rank": 1,
                }
            ],
            "wall_time_s": 0.01,
        }
    else:
        return {
            "schema_version": "nextstat.tool_result.v1",
            "ok": False,
            "result": None,
            "error": {"type": "ToolError", "message": f"Unknown tool: {name}"},
            "meta": meta,
        }
    return {
        "schema_version": "nextstat.tool_result.v1",
        "ok": True,
        "result": result,
        "error": None,
        "meta": meta,
    }


def _serve() -> tuple[HTTPServer, threading.Thread, type[_ScriptToolStubHandler]]:
    handler = type(
        "ScriptToolHandler",
        (_ScriptToolStubHandler,),
        {"call_log": []},
    )
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread, handler


def test_validate_tool_schema_descriptor_server_mode_with_auth_env(monkeypatch):
    httpd, _thread, _handler = _serve()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        env = dict(os.environ)
        env["NEXTSTAT_TOOLS_SERVER_URL"] = url
        env["NEXTSTAT_TOOLS_API_KEY"] = "script-secret"
        proc = subprocess.run(
            [sys.executable, "scripts/validate_tool_schema_descriptor.py", "--transport", "server"],
            cwd=_repo_root(),
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
        assert "Validated server tool descriptor" in proc.stdout
    finally:
        httpd.shutdown()


def test_tool_call_smoke_server_mode_uses_bearer_auth_env(monkeypatch):
    httpd, _thread, handler = _serve()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        env = dict(os.environ)
        env["NEXTSTAT_TOOLS_SERVER_URL"] = url
        env["NEXTSTAT_TOOLS_API_KEY"] = "script-secret"
        proc = subprocess.run(
            [sys.executable, "scripts/tool_call_smoke.py", "--transport", "server", "--timeout-s", "5"],
            cwd=_repo_root(),
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
        assert "== nextstat_fit ==" in proc.stdout
        assert "nextstat.tool_result.v1" in proc.stdout
        assert len(handler.call_log) == 5
        assert all(
            call.get("arguments", {}).get("execution") == {"deterministic": True}
            for call in handler.call_log
        )
    finally:
        httpd.shutdown()


def test_e2e_discovery_server_mode_writes_summary_with_auth_env(tmp_path):
    httpd, _thread, handler = _serve()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}"
        env = dict(os.environ)
        env["NEXTSTAT_TOOLS_SERVER_URL"] = url
        env["NEXTSTAT_TOOLS_API_KEY"] = "script-secret"
        out_dir = tmp_path / "e2e"
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/e2e_discovery.py",
                "--workspace",
                "tests/fixtures/simple_workspace.json",
                "--out-dir",
                str(out_dir),
                "--transport",
                "server",
                "--timeout-s",
                "5",
                "--scan-points",
                "3",
                "--ranking-top-n",
                "1",
            ],
            cwd=_repo_root(),
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["inputs"]["transport"] == "server"
        assert summary["inputs"]["server_url"] is None
        assert summary["key_results"]["poi_value"] == 1.25
        calls = json.loads((out_dir / "calls.json").read_text(encoding="utf-8"))
        assert len(calls) == 7
        assert len(handler.call_log) == 7
        assert all(
            call.get("arguments", {}).get("execution", {}).get("threads") == 1
            for call in handler.call_log
        )
    finally:
        httpd.shutdown()
