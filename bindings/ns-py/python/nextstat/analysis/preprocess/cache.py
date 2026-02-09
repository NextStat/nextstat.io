"""Content-hash keyed cache for preprocessing results.

Avoids recomputing expensive transforms when the input workspace and
pipeline configuration have not changed.  Cache is opt-in and never
changes numeric outputs.

This module is intentionally dependency-free (no numpy).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import build_pipeline_from_config
from .pipeline import PreprocessPipeline, PreprocessResult
from .types import PreprocessProvenance, PreprocessRecord, PreprocessStepProvenance

# Version tag embedded in cache keys so that code changes invalidate the cache.
_CACHE_VERSION = "preprocess_cache_v0"


def _json_canon(obj: Any) -> str:
    """Canonical JSON for hashing (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(
    workspace: Mapping[str, Any],
    steps_config: Sequence[Mapping[str, Any]],
) -> str:
    """Compute a deterministic cache key from workspace content + pipeline config.

    The key incorporates:
    - SHA256 of the canonical workspace JSON
    - SHA256 of the canonical steps config JSON
    - Cache version tag (invalidates on code changes)
    """
    ws_hash = _sha256(_json_canon(workspace))
    cfg_hash = _sha256(_json_canon(list(steps_config)))
    combined = f"{_CACHE_VERSION}:{ws_hash}:{cfg_hash}"
    return _sha256(combined)


class CachedPreprocessPipeline:
    """Wrap ``PreprocessPipeline`` with a content-hash keyed filesystem cache.

    Parameters
    ----------
    cache_dir
        Directory for cache files.  Created on first write.
    steps_config
        Step specs (same format as ``execution.preprocessing.steps``).
        If ``None``, uses the default TREx-standard pipeline.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        steps_config: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._steps_config: list[dict[str, Any]] = (
            [dict(s) for s in steps_config] if steps_config is not None else []
        )
        self._use_default = steps_config is None

    def _get_steps_for_key(self) -> list[dict[str, Any]]:
        if self._use_default:
            from .config import DEFAULT_STEPS
            return DEFAULT_STEPS
        return self._steps_config

    def run(self, workspace: dict[str, Any]) -> PreprocessResult:
        """Run preprocessing with caching.

        If a cached result exists for the same workspace + config, return it
        without recomputing.  Otherwise, run the pipeline, cache the result,
        and return it.
        """
        steps_for_key = self._get_steps_for_key()
        key = cache_key(workspace, steps_for_key)
        cached = self._load(key)
        if cached is not None:
            return cached

        pipeline = build_pipeline_from_config(
            self._steps_config if not self._use_default else None
        )
        result = pipeline.run(workspace)
        self._save(key, result)
        return result

    def _cache_path(self, key: str) -> Path:
        # Use first 2 chars as subdirectory to avoid huge flat dirs.
        return self._cache_dir / key[:2] / f"{key}.json"

    def _load(self, key: str) -> PreprocessResult | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            ws = data["workspace"]
            prov_d = data["provenance"]
            prov = PreprocessProvenance(
                version=prov_d["version"],
                steps=[
                    PreprocessStepProvenance(
                        name=s["name"],
                        params=s["params"],
                        input_sha256=s["input_sha256"],
                        output_sha256=s["output_sha256"],
                        records=[
                            PreprocessRecord(
                                kind=r["kind"],
                                channel=r["channel"],
                                sample=r["sample"],
                                modifier=r["modifier"],
                                modifier_type=r["modifier_type"],
                                changed=r["changed"],
                                metrics=r.get("metrics", {}),
                            )
                            for r in s["records"]
                        ],
                    )
                    for s in prov_d["steps"]
                ],
                input_sha256=prov_d["input_sha256"],
                output_sha256=prov_d["output_sha256"],
            )
            return PreprocessResult(workspace=ws, provenance=prov)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Corrupt cache entry — treat as miss.
            return None

    def _save(self, key: str, result: PreprocessResult) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "workspace": result.workspace,
            "provenance": result.provenance_dict(),
        }
        # Write atomically via temp file to avoid corrupt reads.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(_json_canon(data))
        tmp.rename(path)

    def clear(self) -> int:
        """Remove all cached entries.  Returns number of files removed."""
        count = 0
        if self._cache_dir.exists():
            for p in self._cache_dir.rglob("*.json"):
                p.unlink()
                count += 1
        return count


__all__ = [
    "CachedPreprocessPipeline",
    "cache_key",
]
