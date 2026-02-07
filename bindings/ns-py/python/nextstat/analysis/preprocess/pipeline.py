"""Workspace preprocessing pipeline (TREx replacement).

Input/Output format: pyhf Workspace JSON as a Python dict.
The pipeline is intentionally lightweight and dependency-free.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .hygiene import NegativeBinsPolicy, apply_negative_bins_policy
from .symmetrize import NegativePolicy, SymmetrizeMethod, symmetrize_shapes
from .types import Json, PreprocessProvenance, PreprocessRecord, PreprocessStepProvenance


def _sha256_json(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _allclose(a: Sequence[float], b: Sequence[float], *, atol: float, rtol: float) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        if abs(fx - fy) > (atol + rtol * abs(fy)):
            return False
    return True


def _iter_workspace_deterministic(workspace: Mapping[str, Any]) -> Iterable[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    channels = list(workspace.get("channels") or [])
    channels_sorted = sorted(channels, key=lambda c: str(c.get("name") or ""))
    for ch in channels_sorted:
        samples = list(ch.get("samples") or [])
        samples_sorted = sorted(samples, key=lambda s: str(s.get("name") or ""))
        for sample in samples_sorted:
            modifiers = list(sample.get("modifiers") or [])
            modifiers_sorted = sorted(
                modifiers,
                key=lambda m: (str(m.get("name") or ""), str(m.get("type") or "")),
            )
            for modifier in modifiers_sorted:
                yield ch, sample, modifier


@dataclass(frozen=True)
class PreprocessResult:
    workspace: dict[str, Any]
    provenance: PreprocessProvenance

    def provenance_dict(self) -> dict[str, Json]:
        return self.provenance.to_dict()


class PreprocessStep:
    name: str

    def params(self) -> Mapping[str, Json]:
        return {}

    def apply(self, workspace: dict[str, Any]) -> list[PreprocessRecord]:
        raise NotImplementedError


class SymmetrizeHistoSysStep(PreprocessStep):
    """Symmetrize all `histosys` modifiers in a pyhf workspace."""

    name = "symmetrize_histosys_v0"

    def __init__(
        self,
        *,
        method: SymmetrizeMethod = "absmean",
        negative_policy: NegativePolicy = "error",
        record_unchanged: bool = True,
        compare_atol: float = 1e-12,
        compare_rtol: float = 1e-12,
    ) -> None:
        self._method = method
        self._negative_policy = negative_policy
        self._record_unchanged = bool(record_unchanged)
        self._compare_atol = float(compare_atol)
        self._compare_rtol = float(compare_rtol)

    def params(self) -> Mapping[str, Json]:
        return {
            "method": self._method,
            "negative_policy": self._negative_policy,
            "record_unchanged": self._record_unchanged,
            "compare_atol": self._compare_atol,
            "compare_rtol": self._compare_rtol,
        }

    def apply(self, workspace: dict[str, Any]) -> list[PreprocessRecord]:
        records: list[PreprocessRecord] = []

        for ch, sample, modifier in _iter_workspace_deterministic(workspace):
            if modifier.get("type") != "histosys":
                continue

            mod_data = modifier.get("data") or {}
            hi = mod_data.get("hi_data")
            lo = mod_data.get("lo_data")
            nominal = sample.get("data") or []

            # Support pre-validation pipelines where a histosys may be one-sided.
            hi_seq = None if hi is None else list(hi)
            lo_seq = None if lo is None else list(lo)

            res = symmetrize_shapes(
                nominal,
                up=hi_seq,
                down=lo_seq,
                method=self._method,
                negative_policy=self._negative_policy,
            )

            before_hi = [] if hi_seq is None else [float(x) for x in hi_seq]
            before_lo = [] if lo_seq is None else [float(x) for x in lo_seq]
            after_hi = [float(x) for x in res.up]
            after_lo = [float(x) for x in res.down]

            changed_hi = True
            changed_lo = True
            if hi_seq is not None and len(before_hi) == len(after_hi):
                changed_hi = not _allclose(before_hi, after_hi, atol=self._compare_atol, rtol=self._compare_rtol)
            if lo_seq is not None and len(before_lo) == len(after_lo):
                changed_lo = not _allclose(before_lo, after_lo, atol=self._compare_atol, rtol=self._compare_rtol)

            changed = bool(changed_hi or changed_lo or hi_seq is None or lo_seq is None)

            if changed:
                modifier["data"] = dict(mod_data)
                modifier["data"]["hi_data"] = after_hi
                modifier["data"]["lo_data"] = after_lo

            if changed or self._record_unchanged:
                records.append(
                    PreprocessRecord(
                        kind="histosys.symmetrize",
                        channel=str(ch.get("name") or ""),
                        sample=str(sample.get("name") or ""),
                        modifier=str(modifier.get("name") or ""),
                        modifier_type="histosys",
                        changed=changed,
                        metrics={
                            "method": self._method,
                            "negative_policy": self._negative_policy,
                            "bins": len(after_hi),
                        },
                    )
                )

        return records


class NegativeBinsHygieneStep(PreprocessStep):
    """Apply explicit negative-bin policies to nominal + histosys templates."""

    name = "negative_bins_hygiene_v0"

    def __init__(
        self,
        *,
        policy: NegativeBinsPolicy = "error",
        tol: float = 1e-12,
        renorm: bool = True,
        record_unchanged: bool = False,
    ) -> None:
        self._policy = policy
        self._tol = float(tol)
        self._renorm = bool(renorm)
        self._record_unchanged = bool(record_unchanged)

    def params(self) -> Mapping[str, Json]:
        return {
            "policy": self._policy,
            "tol": self._tol,
            "renorm": self._renorm,
            "record_unchanged": self._record_unchanged,
        }

    def apply(self, workspace: dict[str, Any]) -> list[PreprocessRecord]:
        records: list[PreprocessRecord] = []

        channels = list(workspace.get("channels") or [])
        for ch in sorted(channels, key=lambda c: str(c.get("name") or "")):
            samples = list(ch.get("samples") or [])
            for sample in sorted(samples, key=lambda s: str(s.get("name") or "")):
                nominal = list(sample.get("data") or [])
                r_nom = apply_negative_bins_policy(nominal, policy=self._policy, tol=self._tol, renorm=self._renorm)
                if r_nom.changed:
                    sample["data"] = r_nom.bins
                if r_nom.changed or self._record_unchanged:
                    records.append(
                        PreprocessRecord(
                            kind="bins.hygiene",
                            channel=str(ch.get("name") or ""),
                            sample=str(sample.get("name") or ""),
                            modifier="__nominal__",
                            modifier_type="nominal",
                            changed=r_nom.changed,
                            metrics={
                                "policy": self._policy,
                                "n_negative": r_nom.n_negative,
                                "min_bin": r_nom.min_bin,
                                "sum_before": r_nom.sum_before,
                                "sum_after": r_nom.sum_after,
                                "scale": r_nom.scale,
                                "warnings": list(r_nom.warnings),
                            },
                        )
                    )

        for ch, sample, modifier in _iter_workspace_deterministic(workspace):
            if modifier.get("type") != "histosys":
                continue
            mod_data = modifier.get("data") or {}
            hi = list(mod_data.get("hi_data") or [])
            lo = list(mod_data.get("lo_data") or [])

            r_hi = apply_negative_bins_policy(hi, policy=self._policy, tol=self._tol, renorm=self._renorm)
            r_lo = apply_negative_bins_policy(lo, policy=self._policy, tol=self._tol, renorm=self._renorm)

            if r_hi.changed or r_lo.changed:
                modifier["data"] = dict(mod_data)
                if r_hi.changed:
                    modifier["data"]["hi_data"] = r_hi.bins
                if r_lo.changed:
                    modifier["data"]["lo_data"] = r_lo.bins

            if r_hi.changed or self._record_unchanged:
                records.append(
                    PreprocessRecord(
                        kind="bins.hygiene",
                        channel=str(ch.get("name") or ""),
                        sample=str(sample.get("name") or ""),
                        modifier=str(modifier.get("name") or ""),
                        modifier_type="histosys.hi",
                        changed=r_hi.changed,
                        metrics={
                            "policy": self._policy,
                            "side": "hi_data",
                            "n_negative": r_hi.n_negative,
                            "min_bin": r_hi.min_bin,
                            "sum_before": r_hi.sum_before,
                            "sum_after": r_hi.sum_after,
                            "scale": r_hi.scale,
                            "warnings": list(r_hi.warnings),
                        },
                    )
                )
            if r_lo.changed or self._record_unchanged:
                records.append(
                    PreprocessRecord(
                        kind="bins.hygiene",
                        channel=str(ch.get("name") or ""),
                        sample=str(sample.get("name") or ""),
                        modifier=str(modifier.get("name") or ""),
                        modifier_type="histosys.lo",
                        changed=r_lo.changed,
                        metrics={
                            "policy": self._policy,
                            "side": "lo_data",
                            "n_negative": r_lo.n_negative,
                            "min_bin": r_lo.min_bin,
                            "sum_before": r_lo.sum_before,
                            "sum_after": r_lo.sum_after,
                            "scale": r_lo.scale,
                            "warnings": list(r_lo.warnings),
                        },
                    )
                )

        return records


class PreprocessPipeline:
    """Run a sequence of preprocessing steps on a workspace."""

    def __init__(self, steps: Sequence[PreprocessStep]) -> None:
        self._steps = list(steps)

    @property
    def steps(self) -> list[PreprocessStep]:
        return list(self._steps)

    def run(self, workspace: Mapping[str, Any], *, in_place: bool = False) -> PreprocessResult:
        ws = workspace if in_place else copy.deepcopy(dict(workspace))
        assert isinstance(ws, dict)

        pipeline_input_sha = _sha256_json(ws)
        step_prov: list[PreprocessStepProvenance] = []

        for step in self._steps:
            before_sha = _sha256_json(ws)
            records = step.apply(ws)
            after_sha = _sha256_json(ws)
            step_prov.append(
                PreprocessStepProvenance(
                    name=step.name,
                    params=dict(step.params()),
                    input_sha256=before_sha,
                    output_sha256=after_sha,
                    records=records,
                )
            )

        pipeline_output_sha = _sha256_json(ws)
        prov = PreprocessProvenance(
            version="v0",
            steps=step_prov,
            input_sha256=pipeline_input_sha,
            output_sha256=pipeline_output_sha,
        )
        return PreprocessResult(workspace=ws, provenance=prov)


__all__ = [
    "NegativeBinsHygieneStep",
    "PreprocessPipeline",
    "PreprocessResult",
    "PreprocessStep",
    "SymmetrizeHistoSysStep",
]

