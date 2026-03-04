from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Sequence, Union

from .base import EntitySpan


class Gliner2OnnxBackend:
    name = "onnx"

    def __init__(
        self,
        model_id: str = "lmo3/gliner2-large-v1-onnx",
        providers: Sequence[str] | None = None,
        *,
        num_threads: int | None = None,
    ):
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self._model_id = model_id
        self._providers = list(providers)
        self._num_threads = num_threads

        # ONNX backend does not require torch; avoid noisy imports.
        os.environ.setdefault("TRANSFORMERS_NO_TORCH", "1")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
        # Hide advisory warnings (e.g. tokenizer regex notes) in production logs.
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Many CI / locked-down environments disallow writing to ~/.cache.
        # Use a writable temp cache unless the user explicitly configured HF_HOME.
        os.environ.setdefault("HF_HOME", os.path.join(tempfile.gettempdir(), "nextstat_hf"))
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.environ["HF_HOME"], "xdg_cache"))
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        # Import after cache env vars are set (huggingface_hub reads them at import time).
        try:
            from huggingface_hub.utils import logging as hf_logging  # type: ignore

            hf_logging.set_verbosity_error()
        except Exception:
            pass
        try:
            from transformers.utils import logging as tf_logging  # type: ignore

            tf_logging.set_verbosity_error()
        except Exception:
            pass

        from gliner2_onnx import GLiNER2ONNXRuntime

        # gliner2-onnx (as of 0.1.1) does not accept ORT SessionOptions via
        # `from_pretrained()`. Best-effort thread control via env vars only.
        if num_threads is not None:
            os.environ.setdefault("OMP_NUM_THREADS", str(num_threads))
            os.environ.setdefault("MKL_NUM_THREADS", str(num_threads))
            os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_threads))
            os.environ.setdefault("NUMEXPR_NUM_THREADS", str(num_threads))

        kwargs: Dict[str, Any] = {"providers": self._providers}
        self._runtime = GLiNER2ONNXRuntime.from_pretrained(model_id, **kwargs)

    def extract_entities(self, text: str, schema: Union[Sequence[str], Dict[str, str]]) -> List[EntitySpan]:
        if isinstance(schema, dict):
            labels = list(schema.keys())
        else:
            labels = list(schema)
        ents = self._runtime.extract_entities(text, labels)
        out: List[EntitySpan] = []
        for e in ents:
            out.append(EntitySpan(
                label=str(e.label),
                text=str(e.text),
                start=int(e.start),
                end=int(e.end),
                score=float(e.score),
            ))
        return out

    def environment(self) -> Dict[str, Any]:
        env: Dict[str, Any] = {
            "backend": "gliner2-onnx",
            "model_id": self._model_id,
            "providers": self._providers,
        }
        if self._num_threads is not None:
            env["num_threads"] = self._num_threads
        try:
            import onnxruntime as ort
            env["onnxruntime"] = ort.__version__
        except Exception:
            pass
        try:
            import gliner2_onnx
            env["gliner2_onnx"] = getattr(gliner2_onnx, "__version__", "unknown")
        except Exception:
            pass
        return env
