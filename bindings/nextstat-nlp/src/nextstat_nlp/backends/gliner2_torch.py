from __future__ import annotations

import inspect
import time
from typing import Any, Dict, List, Sequence, Union

from .base import EntitySpan


class Gliner2TorchBackend:
    name = "gliner2"

    def __init__(self, model_id: str = "fastino/gliner2-base-v1", *, device: str = "cpu", max_retries: int = 2):
        from gliner2 import GLiNER2

        self._model_id = model_id
        self._device = device
        self._extract_kwargs: Dict[str, Any] = {}

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                self._extractor = GLiNER2.from_pretrained(model_id)
                self._maybe_move_to_device(device)
                self._maybe_pass_device_to_extract(device)
                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
        if last_err is not None:
            raise last_err

    def _maybe_move_to_device(self, device: str) -> None:
        if device == "cpu":
            return
        try:
            import torch
        except Exception:
            return
        try:
            if device == "mps" and not getattr(torch.backends, "mps", None):
                return
            if device == "mps" and not torch.backends.mps.is_available():
                return
        except Exception:
            return

        # Best-effort: GLiNER2 may or may not expose `.to()`. If it doesn't, this is a no-op.
        try:
            if hasattr(self._extractor, "to"):
                self._extractor = self._extractor.to(device)
                return
        except Exception:
            pass
        try:
            inner = getattr(self._extractor, "model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to(device)
        except Exception:
            pass

    def _maybe_pass_device_to_extract(self, device: str) -> None:
        # Some GLiNER wrappers take `device=...` on the call site; keep compatibility.
        try:
            sig = inspect.signature(self._extractor.extract_entities)
        except Exception:
            return
        if "device" in sig.parameters and device != "cpu":
            self._extract_kwargs["device"] = device

    def extract_entities(self, text: str, schema: Union[Sequence[str], Dict[str, str]]) -> List[EntitySpan]:
        result = self._extractor.extract_entities(
            text,
            schema,
            include_confidence=True,
            include_spans=True,
            **self._extract_kwargs,
        )
        entities = result.get("entities", {}) if isinstance(result, dict) else {}
        out: List[EntitySpan] = []
        for label, items in entities.items():
            for it in items:
                if isinstance(it, str):
                    continue
                out.append(
                    EntitySpan(
                        label=str(label),
                        text=str(it.get("text", "")),
                        start=int(it.get("start", -1)),
                        end=int(it.get("end", -1)),
                        score=float(it.get("confidence")) if it.get("confidence") is not None else None,
                    )
                )
        return [e for e in out if e.start >= 0 and e.end >= 0 and e.text]

    def environment(self) -> Dict[str, Any]:
        env: Dict[str, Any] = {
            "backend": "gliner2",
            "model_id": self._model_id,
            "device": self._device,
        }
        try:
            import torch
            env["torch"] = torch.__version__
        except Exception:
            pass
        try:
            import gliner2
            env["gliner2"] = getattr(gliner2, "__version__", "unknown")
        except Exception:
            pass
        return env
