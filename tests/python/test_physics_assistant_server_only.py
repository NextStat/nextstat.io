import importlib.util
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_demo_module():
    path = _repo_root() / "demos" / "physics_assistant" / "run_demo_server_only.py"
    spec = importlib.util.spec_from_file_location("nextstat_demo_run_demo_server_only", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_server_tools_sends_bearer_auth_for_schema_and_execute(monkeypatch):
    module = _load_demo_module()

    captured: dict[str, object] = {}

    class _Response:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _HttpxStub:
        def get(self, url, *, timeout, headers):
            captured["get"] = {"url": url, "timeout": timeout, "headers": dict(headers)}
            return _Response({"schema_version": "nextstat.tool_schema.v1", "transport": "server", "tools": [], "capabilities": []})

        def post(self, url, *, json, timeout, headers):
            captured["post"] = {
                "url": url,
                "payload": json,
                "timeout": timeout,
                "headers": dict(headers),
            }
            return _Response(
                {
                    "schema_version": "nextstat.tool_result.v1",
                    "ok": True,
                    "result": {"from": "server"},
                    "error": None,
                    "meta": {"tool_name": json["name"], "nextstat_version": "server"},
                }
            )

    monkeypatch.setitem(sys.modules, "httpx", _HttpxStub())

    cfg = module.ExecCfg(deterministic=True, eval_mode="parity", threads=1)
    tools = module.ServerTools(
        server_url="http://server:3742",
        api_key="secret-key",
        exec_cfg=cfg,
    )
    tools.wait_ready(timeout_s=0.1)
    out = tools.call("nextstat_fit", {"workspace_json": "ws"})

    assert out["ok"] is True
    assert captured["get"]["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["post"]["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["post"]["payload"]["name"] == "nextstat_fit"
    assert captured["post"]["payload"]["arguments"]["execution"] == {
        "deterministic": True,
        "eval_mode": "parity",
        "threads": 1,
    }
