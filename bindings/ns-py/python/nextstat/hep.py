"""HEP-specific convenience wrappers.

This module exposes the stable-first scalar GVM workflow plus the wider
research-grade reporting stack, all intentionally kept out of the stable
top-level namespace.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from . import _core as _core  # type: ignore
except ImportError:  # pragma: no cover
    _core = None  # type: ignore

_measurement_combine_build_spec_json = getattr(_core, "measurement_combine_build_spec_json", None)
_measurement_combine_build_spec_from_manifest_json = getattr(
    _core, "measurement_combine_build_spec_from_manifest_json", None
)
_measurement_combine_json = getattr(_core, "measurement_combine_json", None)
_measurement_combine_calibrate_json = getattr(_core, "measurement_combine_calibrate_json", None)
_measurement_combine_calibrate_study_json = getattr(_core, "measurement_combine_calibrate_study_json", None)
_measurement_combine_scenario_study_json = getattr(_core, "measurement_combine_scenario_study_json", None)
_measurement_combine_calibration_campaign_json = getattr(
    _core, "measurement_combine_calibration_campaign_json", None
)
_measurement_combine_solver_parity_scenario_study_json = getattr(
    _core, "measurement_combine_solver_parity_scenario_study_json", None
)
_measurement_combine_solver_parity_calibration_campaign_json = getattr(
    _core, "measurement_combine_solver_parity_calibration_campaign_json", None
)
_measurement_combine_solver_parity_scenario_study_reports_json = getattr(
    _core, "measurement_combine_solver_parity_scenario_study_reports_json", None
)
_measurement_combine_solver_parity_calibration_campaign_reports_json = getattr(
    _core, "measurement_combine_solver_parity_calibration_campaign_reports_json", None
)
_measurement_combine_solver_parity_scenario_study_markdown = getattr(
    _core, "measurement_combine_solver_parity_scenario_study_markdown", None
)
_measurement_combine_solver_parity_calibration_campaign_markdown = getattr(
    _core, "measurement_combine_solver_parity_calibration_campaign_markdown", None
)
_measurement_combine_solver_parity_scenario_study_summary_json = getattr(
    _core, "measurement_combine_solver_parity_scenario_study_summary_json", None
)
_measurement_combine_solver_parity_scenario_study_summary_markdown = getattr(
    _core, "measurement_combine_solver_parity_scenario_study_summary_markdown", None
)
_measurement_combine_solver_parity_calibration_campaign_summary_json = getattr(
    _core, "measurement_combine_solver_parity_calibration_campaign_summary_json", None
)
_measurement_combine_solver_parity_calibration_campaign_summary_markdown = getattr(
    _core, "measurement_combine_solver_parity_calibration_campaign_summary_markdown", None
)
_measurement_combine_calibration_campaign_summary_json = getattr(
    _core, "measurement_combine_calibration_campaign_summary_json", None
)
_measurement_combine_calibration_campaign_summary_markdown = getattr(
    _core, "measurement_combine_calibration_campaign_summary_markdown", None
)
_measurement_combine_calibration_campaign_brief_json = getattr(
    _core, "measurement_combine_calibration_campaign_brief_json", None
)
_measurement_combine_calibration_campaign_brief_markdown = getattr(
    _core, "measurement_combine_calibration_campaign_brief_markdown", None
)
_measurement_combine_calibration_campaign_family_report_json = getattr(
    _core, "measurement_combine_calibration_campaign_family_report_json", None
)
_measurement_combine_calibration_campaign_family_report_markdown = getattr(
    _core, "measurement_combine_calibration_campaign_family_report_markdown", None
)
_measurement_combine_calibration_campaign_family_matrix_json = getattr(
    _core, "measurement_combine_calibration_campaign_family_matrix_json", None
)
_measurement_combine_calibration_campaign_family_matrix_markdown = getattr(
    _core, "measurement_combine_calibration_campaign_family_matrix_markdown", None
)
_measurement_combine_calibration_campaign_portfolio_json = getattr(
    _core, "measurement_combine_calibration_campaign_portfolio_json", None
)
_measurement_combine_calibration_campaign_portfolio_markdown = getattr(
    _core, "measurement_combine_calibration_campaign_portfolio_markdown", None
)
_measurement_combine_calibration_campaign_portfolio_stability_json = getattr(
    _core, "measurement_combine_calibration_campaign_portfolio_stability_json", None
)
_measurement_combine_calibration_campaign_portfolio_stability_markdown = getattr(
    _core, "measurement_combine_calibration_campaign_portfolio_stability_markdown", None
)


def _coerce_spec_json(spec_or_path: dict[str, Any] | str | Path) -> str:
    if isinstance(spec_or_path, dict):
        return json.dumps(spec_or_path)
    if isinstance(spec_or_path, Path):
        return spec_or_path.read_text(encoding="utf-8")
    text = str(spec_or_path)
    path = Path(text)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return text


def _coerce_text_payload(text_or_path: str | Path) -> str:
    if isinstance(text_or_path, Path):
        return text_or_path.read_text(encoding="utf-8")
    text = str(text_or_path)
    path = Path(text)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return text


def _coerce_json_payload(payload_or_path: dict[str, Any] | str | Path) -> str:
    return _coerce_spec_json(payload_or_path)


def build_measurement_combination_spec(
    measurements_table: str | Path,
    stat_covariance_table: str | Path,
    *,
    poi: str = "mu",
    systematics_table: str | Path | None = None,
    correlations_table: str | Path | None = None,
) -> dict[str, Any]:
    """Build a stable-first measurement-combination spec from tabular inputs.

    Each argument may be either raw CSV/TSV text or a filesystem path.
    """
    if _measurement_combine_build_spec_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_build_spec_json is unavailable; rebuild/"
            "reinstall the NextStat Python extension to use "
            "nextstat.hep.build_measurement_combination_spec"
        )
    return json.loads(
        _measurement_combine_build_spec_json(
            _coerce_text_payload(measurements_table),
            _coerce_text_payload(stat_covariance_table),
            poi=str(poi),
            systematics_table=(
                None if systematics_table is None else _coerce_text_payload(systematics_table)
            ),
            correlations_table=(
                None if correlations_table is None else _coerce_text_payload(correlations_table)
            ),
        )
    )


def build_measurement_combination_spec_from_manifest(
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Build a stable-first measurement-combination spec from a YAML/JSON manifest path."""
    if _measurement_combine_build_spec_from_manifest_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_build_spec_from_manifest_json is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.build_measurement_combination_spec_from_manifest"
        )
    return json.loads(
        _measurement_combine_build_spec_from_manifest_json(str(Path(manifest_path)))
    )


def combine_measurements(
    spec_or_path: dict[str, Any] | str | Path,
    *,
    ci_level: float = 0.68,
    solver: str = "auto",
) -> dict[str, Any]:
    """Combine scalar measurements with correlated systematics.

    Stable-first wrapper for the Rust measurement-combination engine.
    Accepts either a parsed spec dict, a JSON string, or a filesystem path.
    """
    if _measurement_combine_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_json is unavailable; rebuild/reinstall the "
            "NextStat Python extension to use nextstat.hep.combine_measurements"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    return json.loads(
        _measurement_combine_json(spec_json, ci_level=float(ci_level), solver=str(solver))
    )

def calibrate_measurements(
    spec_or_path: dict[str, Any] | str | Path,
    *,
    ci_level: float = 0.68,
    solver: str = "auto",
    n_toys: int = 128,
    seed: int = 42,
) -> dict[str, Any]:
    """Calibrate GVM GOF behavior with deterministic toy generation.

    Stable-first wrapper for the Rust measurement-combination calibration engine.
    Accepts either a parsed spec dict, a JSON string, or a filesystem path.
    """
    if _measurement_combine_calibrate_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibrate_json is unavailable; rebuild/reinstall "
            "the NextStat Python extension to use nextstat.hep.calibrate_measurements"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    return json.loads(
        _measurement_combine_calibrate_json(
            spec_json,
            ci_level=float(ci_level),
            solver=str(solver),
            n_toys=int(n_toys),
            seed=int(seed),
        )
    )

def calibrate_measurements_study(
    spec_or_path: dict[str, Any] | str | Path,
    *,
    ci_level: float = 0.68,
    solver: str = "auto",
    n_toys: int = 128,
    seeds: list[int] | tuple[int, ...] = (42, 43, 44),
) -> dict[str, Any]:
    """Run repeated-seed GVM toy calibration and aggregate stability diagnostics.

    Stable-first wrapper for the Rust calibration-study engine.
    Accepts either a parsed spec dict, a JSON string, or a filesystem path.
    """
    if _measurement_combine_calibrate_study_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibrate_study_json is unavailable; rebuild/"
            "reinstall the NextStat Python extension to use "
            "nextstat.hep.calibrate_measurements_study"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    return json.loads(
        _measurement_combine_calibrate_study_json(
            spec_json,
            ci_level=float(ci_level),
            solver=str(solver),
            n_toys=int(n_toys),
            seeds=[int(seed) for seed in seeds],
        )
    )


def study_measurement_combination_scenarios(
    spec_or_path: dict[str, Any] | str | Path,
    scenarios_or_path: dict[str, Any] | str | Path,
    *,
    ci_level: float = 0.68,
    solver: str = "auto",
) -> dict[str, Any]:
    """Run baseline-relative multi-scenario studies for research-grade GVM combinations.

    Research-grade wrapper for the Rust scenario-study engine.
    Accepts either parsed dicts, JSON strings, or filesystem paths for both inputs.
    """
    if _measurement_combine_scenario_study_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_scenario_study_json is unavailable; rebuild/"
            "reinstall the NextStat Python extension to use "
            "nextstat.hep.study_measurement_combination_scenarios"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    scenario_json = _coerce_json_payload(scenarios_or_path)
    return json.loads(
        _measurement_combine_scenario_study_json(
            spec_json,
            scenario_json,
            ci_level=float(ci_level),
            solver=str(solver),
        )
    )

def calibrate_measurement_combination_scenarios(
    spec_or_path: dict[str, Any] | str | Path,
    scenarios_or_path: dict[str, Any] | str | Path,
    *,
    ci_level: float = 0.68,
    solver: str = "auto",
    n_toys: int = 128,
    seeds: list[int] | tuple[int, ...] = (42, 43, 44),
) -> dict[str, Any]:
    """Run repeated-seed calibration campaigns across named GVM scenarios.

    Research-grade wrapper for the Rust calibration-campaign engine.
    Accepts either parsed dicts, JSON strings, or filesystem paths for both inputs.
    """
    if _measurement_combine_calibration_campaign_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_json is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.calibrate_measurement_combination_scenarios"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    scenario_json = _coerce_json_payload(scenarios_or_path)
    return json.loads(
        _measurement_combine_calibration_campaign_json(
            spec_json,
            scenario_json,
            ci_level=float(ci_level),
            solver=str(solver),
            n_toys=int(n_toys),
            seeds=[int(seed) for seed in seeds],
        )
    )


def compare_measurement_combination_scenario_study_solvers(
    spec_or_path: dict[str, Any] | str | Path,
    scenarios_or_path: dict[str, Any] | str | Path,
    *,
    ci_level: float = 0.68,
    lhs_solver: str = "numerical-paper",
    rhs_solver: str = "analytic-perturbative",
) -> dict[str, Any]:
    """Compare two solver modes on the same research-grade scenario study."""
    if _measurement_combine_solver_parity_scenario_study_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_solver_parity_scenario_study_json is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.compare_measurement_combination_scenario_study_solvers"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    scenario_json = _coerce_json_payload(scenarios_or_path)
    return json.loads(
        _measurement_combine_solver_parity_scenario_study_json(
            spec_json,
            scenario_json,
            ci_level=float(ci_level),
            lhs_solver=str(lhs_solver),
            rhs_solver=str(rhs_solver),
        )
    )


def compare_measurement_combination_calibration_campaign_solvers(
    spec_or_path: dict[str, Any] | str | Path,
    scenarios_or_path: dict[str, Any] | str | Path,
    *,
    ci_level: float = 0.68,
    lhs_solver: str = "numerical-paper",
    rhs_solver: str = "analytic-perturbative",
    n_toys: int = 128,
    seeds: list[int] | tuple[int, ...] = (42, 43, 44),
) -> dict[str, Any]:
    """Compare two solver modes on the same research-grade calibration campaign."""
    if _measurement_combine_solver_parity_calibration_campaign_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_solver_parity_calibration_campaign_json is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.compare_measurement_combination_calibration_campaign_solvers"
        )
    spec_json = _coerce_json_payload(spec_or_path)
    scenario_json = _coerce_json_payload(scenarios_or_path)
    return json.loads(
        _measurement_combine_solver_parity_calibration_campaign_json(
            spec_json,
            scenario_json,
            ci_level=float(ci_level),
            lhs_solver=str(lhs_solver),
            rhs_solver=str(rhs_solver),
            n_toys=int(n_toys),
            seeds=[int(seed) for seed in seeds],
        )
    )


def compare_measurement_combination_scenario_study_solver_reports(
    lhs_report_or_path: dict[str, Any] | str | Path,
    rhs_report_or_path: dict[str, Any] | str | Path,
    *,
    lhs_solver: str = "numerical-paper",
    rhs_solver: str = "analytic-perturbative",
) -> dict[str, Any]:
    """Compare two precomputed scenario-study reports without rerunning fits."""
    if _measurement_combine_solver_parity_scenario_study_reports_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_solver_parity_scenario_study_reports_json is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.compare_measurement_combination_scenario_study_solver_reports"
        )
    lhs_json = _coerce_json_payload(lhs_report_or_path)
    rhs_json = _coerce_json_payload(rhs_report_or_path)
    return json.loads(
        _measurement_combine_solver_parity_scenario_study_reports_json(
            lhs_json,
            rhs_json,
            lhs_solver=str(lhs_solver),
            rhs_solver=str(rhs_solver),
        )
    )


def compare_measurement_combination_calibration_campaign_solver_reports(
    lhs_report_or_path: dict[str, Any] | str | Path,
    rhs_report_or_path: dict[str, Any] | str | Path,
    *,
    lhs_solver: str = "numerical-paper",
    rhs_solver: str = "analytic-perturbative",
) -> dict[str, Any]:
    """Compare two precomputed calibration-campaign reports without rerunning fits or toys."""
    if _measurement_combine_solver_parity_calibration_campaign_reports_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_solver_parity_calibration_campaign_reports_json "
            "is unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.compare_measurement_combination_calibration_campaign_solver_reports"
        )
    lhs_json = _coerce_json_payload(lhs_report_or_path)
    rhs_json = _coerce_json_payload(rhs_report_or_path)
    return json.loads(
        _measurement_combine_solver_parity_calibration_campaign_reports_json(
            lhs_json,
            rhs_json,
            lhs_solver=str(lhs_solver),
            rhs_solver=str(rhs_solver),
        )
    )


def render_measurement_combination_scenario_study_solver_parity(
    report_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a scenario-study solver parity report as Markdown."""
    if _measurement_combine_solver_parity_scenario_study_markdown is None:
        raise ImportError(
            "nextstat._core.measurement_combine_solver_parity_scenario_study_markdown is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.render_measurement_combination_scenario_study_solver_parity"
        )
    report_json = _coerce_json_payload(report_or_path)
    return _measurement_combine_solver_parity_scenario_study_markdown(report_json)


def render_measurement_combination_calibration_campaign_solver_parity(
    report_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a calibration-campaign solver parity report as Markdown."""
    if _measurement_combine_solver_parity_calibration_campaign_markdown is None:
        raise ImportError(
            "nextstat._core.measurement_combine_solver_parity_calibration_campaign_markdown is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.render_measurement_combination_calibration_campaign_solver_parity"
        )
    report_json = _coerce_json_payload(report_or_path)
    return _measurement_combine_solver_parity_calibration_campaign_markdown(report_json)


def summarize_measurement_combination_scenario_study_solver_parity(
    report_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Summarize a scenario-study solver parity artifact into a compact digest."""
    if _measurement_combine_solver_parity_scenario_study_summary_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_solver_parity_scenario_study_summary_json is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.summarize_measurement_combination_scenario_study_solver_parity"
        )
    report_json = _coerce_json_payload(report_or_path)
    return json.loads(_measurement_combine_solver_parity_scenario_study_summary_json(report_json))


def render_measurement_combination_scenario_study_solver_parity_summary(
    summary_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a scenario-study solver parity digest as Markdown."""
    if _measurement_combine_solver_parity_scenario_study_summary_markdown is None:
        raise ImportError(
            "nextstat._core.measurement_combine_solver_parity_scenario_study_summary_markdown is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.render_measurement_combination_scenario_study_solver_parity_summary"
        )
    summary_json = _coerce_json_payload(summary_or_path)
    return _measurement_combine_solver_parity_scenario_study_summary_markdown(summary_json)


def summarize_measurement_combination_calibration_campaign_solver_parity(
    report_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Summarize a calibration-campaign solver parity artifact into a compact digest."""
    if _measurement_combine_solver_parity_calibration_campaign_summary_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_solver_parity_calibration_campaign_summary_json "
            "is unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.summarize_measurement_combination_calibration_campaign_solver_parity"
        )
    report_json = _coerce_json_payload(report_or_path)
    return json.loads(
        _measurement_combine_solver_parity_calibration_campaign_summary_json(report_json)
    )


def render_measurement_combination_calibration_campaign_solver_parity_summary(
    summary_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a calibration-campaign solver parity digest as Markdown."""
    if _measurement_combine_solver_parity_calibration_campaign_summary_markdown is None:
        raise ImportError(
            "nextstat._core.measurement_combine_solver_parity_calibration_campaign_summary_markdown "
            "is unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.render_measurement_combination_calibration_campaign_solver_parity_summary"
        )
    summary_json = _coerce_json_payload(summary_or_path)
    return _measurement_combine_solver_parity_calibration_campaign_summary_markdown(summary_json)


def summarize_measurement_combination_calibration_campaign(
    report_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Summarize a calibration-campaign artifact into a compact research digest."""
    if _measurement_combine_calibration_campaign_summary_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_summary_json is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.summarize_measurement_combination_calibration_campaign"
        )
    report_json = _coerce_json_payload(report_or_path)
    return json.loads(_measurement_combine_calibration_campaign_summary_json(report_json))


def render_measurement_combination_calibration_campaign_summary(
    summary_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a calibration-campaign digest as Markdown."""
    if _measurement_combine_calibration_campaign_summary_markdown is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_summary_markdown is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.render_measurement_combination_calibration_campaign_summary"
        )
    summary_json = _coerce_json_payload(summary_or_path)
    return _measurement_combine_calibration_campaign_summary_markdown(summary_json)


def build_measurement_combination_calibration_campaign_brief(
    summaries_or_paths: list[dict[str, Any] | str | Path] | tuple[dict[str, Any] | str | Path, ...],
    *,
    labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a comparative brief from multiple calibration-campaign digests."""
    if _measurement_combine_calibration_campaign_brief_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_brief_json is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.build_measurement_combination_calibration_campaign_brief"
        )
    items = list(summaries_or_paths)
    if labels is not None and len(labels) != len(items):
        raise ValueError("labels must be omitted or provided exactly once per summary")
    payload: list[tuple[str, str]] = []
    for idx, item in enumerate(items):
        label = (
            str(labels[idx])
            if labels is not None
            else (Path(str(item)).stem if not isinstance(item, dict) else f"artifact_{idx + 1}")
        )
        payload.append((label, _coerce_json_payload(item)))
    return json.loads(_measurement_combine_calibration_campaign_brief_json(payload))


def render_measurement_combination_calibration_campaign_brief(
    brief_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a comparative calibration-campaign brief as Markdown."""
    if _measurement_combine_calibration_campaign_brief_markdown is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_brief_markdown is unavailable; "
            "rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.render_measurement_combination_calibration_campaign_brief"
        )
    brief_json = _coerce_json_payload(brief_or_path)
    return _measurement_combine_calibration_campaign_brief_markdown(brief_json)


def build_measurement_combination_calibration_campaign_family_report(
    briefs_or_paths: list[dict[str, Any] | str | Path] | tuple[dict[str, Any] | str | Path, ...],
    *,
    labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a family-level comparative report from multiple campaign briefs."""
    if _measurement_combine_calibration_campaign_family_report_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_family_report_json is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.build_measurement_combination_calibration_campaign_family_report"
        )
    items = list(briefs_or_paths)
    if labels is not None and len(labels) != len(items):
        raise ValueError("labels must be omitted or provided exactly once per brief")
    payload: list[tuple[str, str]] = []
    for idx, item in enumerate(items):
        label = (
            str(labels[idx])
            if labels is not None
            else (Path(str(item)).stem if not isinstance(item, dict) else f"family_{idx + 1}")
        )
        payload.append((label, _coerce_json_payload(item)))
    return json.loads(_measurement_combine_calibration_campaign_family_report_json(payload))


def render_measurement_combination_calibration_campaign_family_report(
    report_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a family-level comparative campaign report as Markdown."""
    if _measurement_combine_calibration_campaign_family_report_markdown is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_family_report_markdown is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.render_measurement_combination_calibration_campaign_family_report"
        )
    report_json = _coerce_json_payload(report_or_path)
    return _measurement_combine_calibration_campaign_family_report_markdown(report_json)


def build_measurement_combination_calibration_campaign_family_matrix(
    report_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build a machine-readable dominance matrix from a family-level report."""
    if _measurement_combine_calibration_campaign_family_matrix_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_family_matrix_json is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.build_measurement_combination_calibration_campaign_family_matrix"
        )
    report_json = _coerce_json_payload(report_or_path)
    return json.loads(_measurement_combine_calibration_campaign_family_matrix_json(report_json))


def render_measurement_combination_calibration_campaign_family_matrix(
    matrix_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a family-level dominance matrix as Markdown."""
    if _measurement_combine_calibration_campaign_family_matrix_markdown is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_family_matrix_markdown is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.render_measurement_combination_calibration_campaign_family_matrix"
        )
    matrix_json = _coerce_json_payload(matrix_or_path)
    return _measurement_combine_calibration_campaign_family_matrix_markdown(matrix_json)


def build_measurement_combination_calibration_campaign_portfolio(
    matrices_or_paths: list[dict[str, Any] | str | Path] | tuple[dict[str, Any] | str | Path, ...],
    *,
    labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a portfolio-level report from multiple family-matrix artifacts."""
    if _measurement_combine_calibration_campaign_portfolio_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_portfolio_json is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.build_measurement_combination_calibration_campaign_portfolio"
        )
    items = list(matrices_or_paths)
    if labels is not None and len(labels) != len(items):
        raise ValueError("labels must be omitted or provided exactly once per matrix")
    payload: list[tuple[str, str]] = []
    for idx, item in enumerate(items):
        label = (
            str(labels[idx])
            if labels is not None
            else (Path(str(item)).stem if not isinstance(item, dict) else f"portfolio_{idx + 1}")
        )
        payload.append((label, _coerce_json_payload(item)))
    return json.loads(_measurement_combine_calibration_campaign_portfolio_json(payload))


def render_measurement_combination_calibration_campaign_portfolio(
    report_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a portfolio-level campaign report as Markdown."""
    if _measurement_combine_calibration_campaign_portfolio_markdown is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_portfolio_markdown is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.render_measurement_combination_calibration_campaign_portfolio"
        )
    report_json = _coerce_json_payload(report_or_path)
    return _measurement_combine_calibration_campaign_portfolio_markdown(report_json)


def build_measurement_combination_calibration_campaign_portfolio_stability(
    portfolios_or_paths: list[dict[str, Any] | str | Path] | tuple[dict[str, Any] | str | Path, ...],
    *,
    labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build a stability report from multiple portfolio artifacts."""
    if _measurement_combine_calibration_campaign_portfolio_stability_json is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_portfolio_stability_json is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.build_measurement_combination_calibration_campaign_portfolio_stability"
        )
    items = list(portfolios_or_paths)
    if labels is not None and len(labels) != len(items):
        raise ValueError("labels must be omitted or provided exactly once per portfolio")
    payload: list[tuple[str, str]] = []
    for idx, item in enumerate(items):
        label = (
            str(labels[idx])
            if labels is not None
            else (Path(str(item)).stem if not isinstance(item, dict) else f"run_{idx + 1}")
        )
        payload.append((label, _coerce_json_payload(item)))
    return json.loads(_measurement_combine_calibration_campaign_portfolio_stability_json(payload))


def render_measurement_combination_calibration_campaign_portfolio_stability(
    report_or_path: dict[str, Any] | str | Path,
) -> str:
    """Render a portfolio-stability report as Markdown."""
    if _measurement_combine_calibration_campaign_portfolio_stability_markdown is None:
        raise ImportError(
            "nextstat._core.measurement_combine_calibration_campaign_portfolio_stability_markdown is "
            "unavailable; rebuild/reinstall the NextStat Python extension to use "
            "nextstat.hep.render_measurement_combination_calibration_campaign_portfolio_stability"
        )
    report_json = _coerce_json_payload(report_or_path)
    return _measurement_combine_calibration_campaign_portfolio_stability_markdown(report_json)


__all__ = [
    "build_measurement_combination_spec",
    "build_measurement_combination_spec_from_manifest",
    "combine_measurements",
    "calibrate_measurements",
    "calibrate_measurements_study",
    "study_measurement_combination_scenarios",
    "calibrate_measurement_combination_scenarios",
    "compare_measurement_combination_scenario_study_solvers",
    "compare_measurement_combination_calibration_campaign_solvers",
    "compare_measurement_combination_scenario_study_solver_reports",
    "compare_measurement_combination_calibration_campaign_solver_reports",
    "render_measurement_combination_scenario_study_solver_parity",
    "render_measurement_combination_calibration_campaign_solver_parity",
    "summarize_measurement_combination_scenario_study_solver_parity",
    "render_measurement_combination_scenario_study_solver_parity_summary",
    "summarize_measurement_combination_calibration_campaign_solver_parity",
    "render_measurement_combination_calibration_campaign_solver_parity_summary",
    "summarize_measurement_combination_calibration_campaign",
    "render_measurement_combination_calibration_campaign_summary",
    "build_measurement_combination_calibration_campaign_brief",
    "render_measurement_combination_calibration_campaign_brief",
    "build_measurement_combination_calibration_campaign_family_report",
    "render_measurement_combination_calibration_campaign_family_report",
    "build_measurement_combination_calibration_campaign_family_matrix",
    "render_measurement_combination_calibration_campaign_family_matrix",
    "build_measurement_combination_calibration_campaign_portfolio",
    "render_measurement_combination_calibration_campaign_portfolio",
    "build_measurement_combination_calibration_campaign_portfolio_stability",
    "render_measurement_combination_calibration_campaign_portfolio_stability",
]
