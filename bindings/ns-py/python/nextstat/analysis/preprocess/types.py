"""Types for TREx-like preprocessing pipeline + provenance.

Design goals (v0):
- Dependency-free (no numpy).
- Deterministic provenance ordering.
- JSON-serializable structures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


Json = Any


@dataclass(frozen=True)
class PreprocessRecord:
    kind: str
    channel: str
    sample: str
    modifier: str
    modifier_type: str
    changed: bool
    metrics: Mapping[str, Json] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Json]:
        return dict(asdict(self))


@dataclass(frozen=True)
class PreprocessStepProvenance:
    name: str
    params: Mapping[str, Json]
    input_sha256: str
    output_sha256: str
    records: list[PreprocessRecord]

    def to_dict(self) -> dict[str, Json]:
        d = dict(asdict(self))
        d["records"] = [r.to_dict() for r in self.records]
        return d


@dataclass(frozen=True)
class PreprocessProvenance:
    version: str
    steps: list[PreprocessStepProvenance]
    input_sha256: str
    output_sha256: str

    def to_dict(self) -> dict[str, Json]:
        d = dict(asdict(self))
        d["steps"] = [s.to_dict() for s in self.steps]
        return d


__all__ = [
    "Json",
    "PreprocessProvenance",
    "PreprocessRecord",
    "PreprocessStepProvenance",
]

