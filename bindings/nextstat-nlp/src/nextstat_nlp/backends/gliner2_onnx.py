from __future__ import annotations

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
        from gliner2_onnx import GLiNER2ONNXRuntime

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self._model_id = model_id
        self._providers = list(providers)
        self._num_threads = num_threads

        kwargs: Dict[str, Any] = {"providers": self._providers}
        if num_threads is not None:
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = num_threads
                so.inter_op_num_threads = num_threads
                kwargs["session_options"] = so
            except Exception:
                pass

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
