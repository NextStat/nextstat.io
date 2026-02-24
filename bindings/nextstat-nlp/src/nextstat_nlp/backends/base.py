from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Protocol, Sequence, Union

from .._errors import MissingBackendDependency


BackendName = Literal["gliner2", "onnx", "heuristic", "mlx"]


@dataclass(frozen=True)
class EntitySpan:
    label: str
    text: str
    start: int
    end: int
    score: Optional[float] = None


class ExtractorBackend(Protocol):
    name: str

    def extract_entities(self, text: str, schema: Union[Sequence[str], Dict[str, str]]) -> List[EntitySpan]:
        ...

    def environment(self) -> Dict[str, Any]:
        ...


def get_backend(
    name: BackendName,
    *,
    model_id: Optional[str] = None,
    device: Optional[str] = None,
    providers: Optional[Sequence[str]] = None,
    num_threads: Optional[int] = None,
) -> ExtractorBackend:
    if name == "gliner2":
        try:
            from .gliner2_torch import Gliner2TorchBackend
        except Exception as e:  # pragma: no cover
            raise MissingBackendDependency(
                "Backend 'gliner2' requires extra 'nextstat-nlp[gliner2]' (pip install gliner2)."
            ) from e
        return Gliner2TorchBackend(
            model_id=model_id or "fastino/gliner2-base-v1",
            device=device or "cpu",
        )
    if name == "onnx":
        try:
            from .gliner2_onnx import Gliner2OnnxBackend
        except Exception as e:  # pragma: no cover
            raise MissingBackendDependency(
                "Backend 'onnx' requires extra 'nextstat-nlp[onnx]' (pip install gliner2-onnx onnxruntime)."
            ) from e
        return Gliner2OnnxBackend(
            model_id=model_id or "lmo3/gliner2-large-v1-onnx",
            providers=providers,
            num_threads=num_threads,
        )
    if name == "mlx":
        try:
            from .gliner2_mlx import Gliner2MlxBackend
        except Exception as e:  # pragma: no cover
            raise MissingBackendDependency(
                "Backend 'mlx' requires the optional Swift CLI 'gliner2_mlx_cli'."
            ) from e
        return Gliner2MlxBackend(model_id=model_id or "fastino/gliner2-base-v1")
    if name == "heuristic":
        from .heuristic import HeuristicBackend

        return HeuristicBackend()
    raise ValueError(f"Unknown backend: {name}")
