"""Build a PreprocessPipeline from a configuration dict (analysis spec YAML).

This module bridges the analysis spec ``execution.preprocessing`` section
and the underlying PreprocessStep implementations.

This module is intentionally dependency-free (no numpy).
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .hygiene import NegativeBinsPolicy
from .pipeline import (
    NegativeBinsHygieneStep,
    PreprocessPipeline,
    PreprocessResult,
    PreprocessStep,
    PruneSystematicsStep,
    SmoothHistoSysStep,
    SymmetrizeHistoSysStep,
)
from .prune import PruneMethod
from .smooth import SmoothMethod
from .symmetrize import NegativePolicy, SymmetrizeMethod


# ---------------------------------------------------------------------------
# Default pipeline config (TREx-standard order)
# ---------------------------------------------------------------------------

DEFAULT_STEPS: list[dict[str, Any]] = [
    {"kind": "negative_bins_hygiene", "params": {"policy": "clamp_renorm"}},
    {"kind": "symmetrize_histosys", "params": {"method": "absmean"}},
    {"kind": "smooth_histosys", "params": {"method": "353qh_twice"}},
    {"kind": "prune_systematics", "params": {"shape_threshold": 0.005}},
]


# ---------------------------------------------------------------------------
# Step factory
# ---------------------------------------------------------------------------


def _build_step(step_spec: Mapping[str, Any]) -> PreprocessStep:
    """Construct a single PreprocessStep from a spec dict."""
    kind = step_spec["kind"]
    params = dict(step_spec.get("params") or {})

    if kind == "negative_bins_hygiene":
        return NegativeBinsHygieneStep(
            policy=params.get("policy", "clamp_renorm"),
            tol=float(params.get("tol", 1e-12)),
            renorm=bool(params.get("renorm", True)),
            record_unchanged=bool(params.get("record_unchanged", False)),
        )

    if kind == "symmetrize_histosys":
        return SymmetrizeHistoSysStep(
            method=params.get("method", "absmean"),
            negative_policy=params.get("negative_policy", "error"),
            record_unchanged=bool(params.get("record_unchanged", True)),
        )

    if kind == "smooth_histosys":
        return SmoothHistoSysStep(
            method=params.get("method", "353qh_twice"),
            sigma=float(params.get("sigma", 1.5)),
            apply_maxvariation=bool(params.get("apply_maxvariation", True)),
            record_unchanged=bool(params.get("record_unchanged", True)),
        )

    if kind == "prune_systematics":
        return PruneSystematicsStep(
            shape_threshold=float(params.get("shape_threshold", 0.005)),
            norm_threshold=float(params.get("norm_threshold", 0.005)),
            prune_method=params.get("prune_method", "shape"),
            record_unchanged=bool(params.get("record_unchanged", False)),
        )

    raise ValueError(f"unknown preprocessing step kind: {kind!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pipeline_from_config(
    steps: Sequence[Mapping[str, Any]] | None = None,
) -> PreprocessPipeline:
    """Build a ``PreprocessPipeline`` from a list of step specs.

    Each step spec is ``{"kind": "...", "params": {...}}``.
    If *steps* is ``None``, the default TREx-standard pipeline is used.
    """
    if steps is None:
        steps = DEFAULT_STEPS
    return PreprocessPipeline([_build_step(s) for s in steps])


def preprocess_workspace(
    workspace: dict[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
    in_place: bool = False,
) -> PreprocessResult:
    """Preprocess a pyhf workspace dict using config from the analysis spec.

    Parameters
    ----------
    workspace
        A pyhf workspace dict (will be deep-copied unless *in_place*).
    config
        The ``execution.preprocessing`` section of the analysis spec.
        If ``None`` or ``config["enabled"]`` is false, returns workspace unchanged.
    in_place
        If ``True``, mutate *workspace* directly (saves memory for large workspaces).

    Returns
    -------
    PreprocessResult
        The (possibly modified) workspace and full provenance.
    """
    if config is not None and not bool(config.get("enabled", True)):
        # Preprocessing disabled — return as-is with empty provenance.
        from .types import PreprocessProvenance

        import copy
        import hashlib

        ws = workspace if in_place else copy.deepcopy(workspace)
        sha = hashlib.sha256(
            json.dumps(ws, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        prov = PreprocessProvenance(
            version="v0", steps=[], input_sha256=sha, output_sha256=sha,
        )
        return PreprocessResult(workspace=ws, provenance=prov)

    steps_cfg = None
    if config is not None:
        steps_cfg = config.get("steps")

    pipeline = build_pipeline_from_config(steps_cfg)
    return pipeline.run(workspace, in_place=in_place)


# Need PreprocessResult in this module's namespace for the return type.
from .pipeline import PreprocessResult as PreprocessResult  # noqa: E402


__all__ = [
    "DEFAULT_STEPS",
    "build_pipeline_from_config",
    "preprocess_workspace",
]
