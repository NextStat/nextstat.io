from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import selectors
from typing import Any, Dict, List, Optional, Sequence, Union

from .._errors import MissingBackendDependency
from .base import EntitySpan


class Gliner2MlxBackend:
    """GLiNER2-on-MLX backend via a small Swift CLI.

    We intentionally avoid Swift/Py bridges: the Python package stays pure-Python,
    while the optional CLI can be built on macOS and invoked from here.

    Protocol: JSONL over stdin/stdout.

    Request line:
      {"text": "...", "labels": ["time", "event", ...]}

    Response line:
      {"entities": [{"label": "time", "text": "84 days", "start": 10, "end": 17, "score": null}], "error": null}
    """

    name = "mlx"

    def __init__(
        self,
        model_id: str = "fastino/gliner2-base-v1",
        *,
        cli_path: Optional[str] = None,
        timeout_s: float = 30.0,
    ):
        self._model_id = model_id
        self._timeout_s = timeout_s

        if cli_path is None:
            cli_path = os.environ.get("NEXTSTAT_GLINER2_MLX_CLI")
        if not cli_path:
            cli_path = shutil.which("gliner2_mlx_cli")

        if not cli_path:
            raise MissingBackendDependency(
                "Backend 'mlx' requires the Swift CLI 'gliner2_mlx_cli'. "
                "Build it from bindings/nextstat-nlp/tools/gliner2_mlx_cli and set "
                "NEXTSTAT_GLINER2_MLX_CLI=/path/to/gliner2_mlx_cli."
            )

        self._cli_path = cli_path
        self._lock = threading.Lock()
        self._proc = self._start_proc()
        self._sel = selectors.DefaultSelector()
        assert self._proc.stdout is not None
        self._sel.register(self._proc.stdout, selectors.EVENT_READ)

    def _start_proc(self) -> subprocess.Popen[str]:
        try:
            return subprocess.Popen(
                [
                    self._cli_path,
                    "--jsonl",
                    "--model-id",
                    self._model_id,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            raise MissingBackendDependency(
                f"Backend 'mlx' CLI not found: {self._cli_path}"
            ) from e

    def _read_json_line(self) -> Dict[str, Any]:
        """Read lines until we get a valid JSON response dict.

        Some MLX/Hub dependencies print to stdout during/after model load; we ignore
        non-JSON lines to keep the protocol robust.
        """
        assert self._proc.stdout is not None

        deadline = time.monotonic() + self._timeout_s
        while True:
            timeout = max(0.0, deadline - time.monotonic())
            if timeout == 0.0:
                raise TimeoutError("mlx backend: timed out waiting for CLI response")

            events = self._sel.select(timeout)
            if not events:
                raise TimeoutError("mlx backend: timed out waiting for CLI response")

            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("mlx backend: empty response (CLI crashed?)")
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and ("entities" in obj or "error" in obj):
                return obj

    def extract_entities(self, text: str, schema: Union[Sequence[str], Dict[str, str]]) -> List[EntitySpan]:
        if isinstance(schema, dict):
            labels = list(schema.keys())
        else:
            labels = list(schema)

        req = {"text": text, "labels": labels}
        line = json.dumps(req, ensure_ascii=False)

        with self._lock:
            if self._proc.poll() is not None:
                # Restart once.
                self._proc = self._start_proc()
                self._sel.close()
                self._sel = selectors.DefaultSelector()
                assert self._proc.stdout is not None
                self._sel.register(self._proc.stdout, selectors.EVENT_READ)

            assert self._proc.stdin is not None

            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()

            resp = self._read_json_line()
        if resp.get("error"):
            raise RuntimeError(f"mlx backend error: {resp['error']}")

        out: List[EntitySpan] = []
        for e in resp.get("entities", []):
            out.append(
                EntitySpan(
                    label=str(e.get("label", "")),
                    text=str(e.get("text", "")),
                    start=int(e.get("start", -1)),
                    end=int(e.get("end", -1)),
                    score=float(e["score"]) if e.get("score") is not None else None,
                )
            )
        return [x for x in out if x.start >= 0 and x.end >= 0 and x.text]

    def environment(self) -> Dict[str, Any]:
        return {
            "backend": "mlx",
            "model_id": self._model_id,
            "cli_path": self._cli_path,
        }
