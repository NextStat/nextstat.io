"""Provenance / audit-trail utilities for reproducible extraction runs."""
from __future__ import annotations

import datetime
import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Sequence

from ._hashing import sha256_text
from .backends.base import ExtractorBackend


@dataclass(frozen=True)
class ProvenanceBundle:
    """Immutable record of the extraction environment and inputs."""
    model_id: str
    model_version: str
    backend: str
    runtime_versions: Dict[str, str]
    config: Dict[str, Any]
    input_hashes: list[str]
    timestamp: str  # ISO-8601

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def collect_provenance(
    backend: ExtractorBackend,
    texts: Sequence[str],
    *,
    config: Dict[str, Any] | None = None,
) -> ProvenanceBundle:
    """Build a ``ProvenanceBundle`` from a backend and input texts."""
    env = backend.environment()
    return ProvenanceBundle(
        model_id=env.get("model_id", ""),
        model_version=env.get("model_version", ""),
        backend=env.get("backend", backend.name),
        runtime_versions={
            "python": sys.version,
            "platform": platform.platform(),
            **{k: v for k, v in env.items() if k not in ("backend", "model_id", "model_version")},
        },
        config=config or {},
        input_hashes=[sha256_text(t) for t in texts],
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
