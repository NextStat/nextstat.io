#!/usr/bin/env python3
"""HEP-ADR-01: Generate the canonical HEP surface matrix.

Produces hep_surface_matrix_v1.json — the single source of truth for every
public HEP surface across CLI, Python API, tool layer, and server.

Usage:
    python scripts/hep_surface_matrix.py          # write to repo root
    python scripts/hep_surface_matrix.py --check   # verify without writing
"""

from __future__ import annotations

import json
import sys
from datetime import date, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "hep_surface_matrix_v1.json"


# ── Surface definitions ──────────────────────────────────────────────────────
# Each surface: (name, layer, maturity_class, owner_slice, support_matrix_ref)

def _cli(name, maturity, owner, ref=None):
    return {"name": name, "layer": "cli", "maturity_class": maturity,
            "owner_slice": owner, "support_matrix_ref": ref or ""}

def _py(name, maturity, owner, ref=None):
    return {"name": name, "layer": "python", "maturity_class": maturity,
            "owner_slice": owner, "support_matrix_ref": ref or ""}

def _tool(name, maturity, owner, ref=None):
    return {"name": name, "layer": "tool", "maturity_class": maturity,
            "owner_slice": owner, "support_matrix_ref": ref or ""}

def _server(name, maturity, owner, ref=None):
    return {"name": name, "layer": "server", "maturity_class": maturity,
            "owner_slice": owner, "support_matrix_ref": ref or ""}


SL_REF = "docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md"
GVM_REF = "docs/benchmarks/gvm-stable-first-support-matrix-2026-03-07.md"
HEPDATA_REF = "docs/benchmarks/hepdata-import-support-matrix-2026-03-08.md"
SL_EXPORT_REF = "docs/benchmarks/simplified-likelihood-exporter-acceptance-2026-03-09.md"
HF_REF = "docs/benchmarks/histfactory-support-matrix-2026-03-17.md"


SURFACES = [
    # ── HistFactory / pyhf core ──────────────────────────────────────────
    # Promoted with dedicated support matrix (HEP-ADR-02)
    _cli("run", "stable", "histfactory", HF_REF),
    _cli("validate", "stable", "histfactory", HF_REF),
    _cli("fit", "stable", "histfactory", HF_REF),
    _cli("audit", "stable", "histfactory", HF_REF),
    _cli("hypotest", "stable", "histfactory", HF_REF),
    _cli("hypotest-toys", "stable", "histfactory", HF_REF),
    _cli("upper-limit", "stable", "histfactory", HF_REF),
    _cli("mass-scan", "stable", "histfactory", HF_REF),
    _cli("scan", "stable", "histfactory", HF_REF),
    _cli("combine", "stable", "histfactory", HF_REF),
    _cli("significance", "stable", "histfactory", HF_REF),
    _cli("goodness-of-fit", "stable", "histfactory", HF_REF),
    _cli("report", "stable", "histfactory", HF_REF),
    _cli("validation-report", "stable", "histfactory", HF_REF),
    _cli("build-hists", "stable", "histfactory", HF_REF),

    _py("HistFactoryModel", "stable", "histfactory", HF_REF),
    _py("from_pyhf", "stable", "histfactory", HF_REF),
    _py("from_histfactory_xml", "stable", "histfactory", HF_REF),
    _py("histfactory_bin_edges_by_channel", "stable", "histfactory", HF_REF),
    _py("from_arrow", "stable", "histfactory", HF_REF),
    _py("to_arrow", "stable", "histfactory", HF_REF),
    _py("apply_patchset", "stable", "histfactory", HF_REF),
    _py("fit", "stable", "histfactory", HF_REF),
    _py("fit_batch", "stable", "histfactory", HF_REF),
    _py("fit_toys", "stable", "histfactory", HF_REF),
    _py("hypotest", "stable", "histfactory", HF_REF),
    _py("hypotest_toys", "stable", "histfactory", HF_REF),
    _py("profile_scan", "stable", "histfactory", HF_REF),
    _py("upper_limit", "stable", "histfactory", HF_REF),
    _py("upper_limits", "stable", "histfactory", HF_REF),
    _py("cls_curve", "stable", "histfactory", HF_REF),
    _py("asimov_data", "stable", "histfactory", HF_REF),
    _py("poisson_toys", "stable", "histfactory", HF_REF),
    _py("ranking", "stable", "histfactory", HF_REF),
    _py("workspace_audit", "stable", "histfactory", HF_REF),
    _py("read_root_histogram", "stable", "histfactory", HF_REF),

    _tool("nextstat_fit", "stable", "histfactory", HF_REF),
    _tool("nextstat_hypotest", "stable", "histfactory", HF_REF),
    _tool("nextstat_hypotest_toys", "stable", "histfactory", HF_REF),
    _tool("nextstat_upper_limit", "stable", "histfactory", HF_REF),
    _tool("nextstat_scan", "stable", "histfactory", HF_REF),
    _tool("nextstat_ranking", "stable", "histfactory", HF_REF),
    _tool("nextstat_workspace_audit", "stable", "histfactory", HF_REF),
    _tool("nextstat_discovery_asymptotic", "stable", "simplified_likelihood", SL_REF),
    _tool("nextstat_read_root_histogram", "stable", "histfactory", HF_REF),

    _server("POST /v1/fit", "stable", "histfactory", HF_REF),
    _server("POST /v1/ranking", "stable", "histfactory", HF_REF),
    _server("POST /v1/batch/fit", "stable", "histfactory", HF_REF),
    _server("POST /v1/batch/toys", "stable", "histfactory", HF_REF),
    _server("POST /v1/tools/execute", "stable", "infrastructure"),

    # ── Simplified Likelihood ────────────────────────────────────────────
    _cli("simplify workspace", "stable", "simplified_likelihood", SL_EXPORT_REF),

    # ── Unbinned ─────────────────────────────────────────────────────────
    # Pending dedicated support matrix
    _cli("unbinned-fit", "stable", "unbinned"),
    _cli("hybrid-fit", "stable", "unbinned"),
    _cli("unbinned-scan", "stable", "unbinned"),
    _cli("unbinned-fit-toys", "stable", "unbinned"),
    _cli("unbinned-merge-toys", "stable", "unbinned"),
    _cli("unbinned-ranking", "stable", "unbinned"),
    _cli("unbinned-hypotest", "stable", "unbinned"),
    _cli("unbinned-hypotest-toys", "stable", "unbinned"),

    _py("UnbinnedModel", "stable", "unbinned"),
    _py("HybridModel", "stable", "unbinned"),

    _server("POST /v1/unbinned/fit", "stable", "unbinned"),

    # ── GVM stable core ─────────────────────────────────────────────────
    _cli("combine-measurements", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-build-spec", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-calibrate", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-calibrate-study", "stable", "gvm", GVM_REF),

    _py("hep.build_measurement_combination_spec", "stable", "gvm", GVM_REF),
    _py("hep.build_measurement_combination_spec_from_manifest", "stable", "gvm", GVM_REF),
    _py("hep.combine_measurements", "stable", "gvm", GVM_REF),
    _py("hep.calibrate_measurements", "stable", "gvm", GVM_REF),
    _py("hep.calibrate_measurements_study", "stable", "gvm", GVM_REF),

    # ── GVM promoted (scenario/campaign/parity/reporting) ────────────────
    # Promoted with dedicated support matrix (GVM-05A/B/C/D graduation)
    _cli("combine-measurements-scenario-study", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-calibration-campaign", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-solver-parity-scenario-study", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-solver-parity-calibration-campaign", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-solver-parity-scenario-study-from-reports", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-solver-parity-calibration-campaign-from-reports", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-solver-parity-scenario-study-summarize", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-solver-parity-calibration-campaign-summarize", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-calibration-campaign-summarize", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-calibration-campaign-brief", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-calibration-campaign-family-report", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-calibration-campaign-family-matrix", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-calibration-campaign-portfolio", "stable", "gvm", GVM_REF),
    _cli("combine-measurements-calibration-campaign-portfolio-stability", "stable", "gvm", GVM_REF),

    _py("hep.study_measurement_combination_scenarios", "stable", "gvm", GVM_REF),
    _py("hep.calibrate_measurement_combination_scenarios", "stable", "gvm", GVM_REF),
    _py("hep.compare_measurement_combination_scenario_study_solvers", "stable", "gvm", GVM_REF),
    _py("hep.compare_measurement_combination_calibration_campaign_solvers", "stable", "gvm", GVM_REF),
    _py("hep.compare_measurement_combination_scenario_study_solver_reports", "stable", "gvm", GVM_REF),
    _py("hep.compare_measurement_combination_calibration_campaign_solver_reports", "stable", "gvm", GVM_REF),
    _py("hep.render_measurement_combination_scenario_study_solver_parity", "stable", "gvm", GVM_REF),
    _py("hep.render_measurement_combination_calibration_campaign_solver_parity", "stable", "gvm", GVM_REF),
    _py("hep.summarize_measurement_combination_scenario_study_solver_parity", "stable", "gvm", GVM_REF),
    _py("hep.render_measurement_combination_scenario_study_solver_parity_summary", "stable", "gvm", GVM_REF),
    _py("hep.summarize_measurement_combination_calibration_campaign_solver_parity", "stable", "gvm", GVM_REF),
    _py("hep.render_measurement_combination_calibration_campaign_solver_parity_summary", "stable", "gvm", GVM_REF),
    _py("hep.summarize_measurement_combination_calibration_campaign", "stable", "gvm", GVM_REF),
    _py("hep.render_measurement_combination_calibration_campaign_summary", "stable", "gvm", GVM_REF),
    _py("hep.build_measurement_combination_calibration_campaign_brief", "stable", "gvm", GVM_REF),
    _py("hep.render_measurement_combination_calibration_campaign_brief", "stable", "gvm", GVM_REF),
    _py("hep.build_measurement_combination_calibration_campaign_family_report", "stable", "gvm", GVM_REF),
    _py("hep.render_measurement_combination_calibration_campaign_family_report", "stable", "gvm", GVM_REF),
    _py("hep.build_measurement_combination_calibration_campaign_family_matrix", "stable", "gvm", GVM_REF),
    _py("hep.render_measurement_combination_calibration_campaign_family_matrix", "stable", "gvm", GVM_REF),
    _py("hep.build_measurement_combination_calibration_campaign_portfolio", "stable", "gvm", GVM_REF),
    _py("hep.render_measurement_combination_calibration_campaign_portfolio", "stable", "gvm", GVM_REF),
    _py("hep.build_measurement_combination_calibration_campaign_portfolio_stability", "stable", "gvm", GVM_REF),
    _py("hep.render_measurement_combination_calibration_campaign_portfolio_stability", "stable", "gvm", GVM_REF),

    # ── Viz ───────────────────────────────────────────────────────────────
    # Pending dedicated support matrix
    _cli("viz profile", "stable", "viz"),
    _cli("viz cls", "stable", "viz"),
    _cli("viz ranking", "stable", "viz"),
    _cli("viz pulls", "stable", "viz"),
    _cli("viz gammas", "stable", "viz"),
    _cli("viz corr", "stable", "viz"),
    _cli("viz distributions", "stable", "viz"),
    _cli("viz separation", "stable", "viz"),
    _cli("viz summary", "stable", "viz"),
    _cli("viz pie", "stable", "viz"),
    _cli("viz render", "stable", "viz"),

    # ── Import / Export ──────────────────────────────────────────────────
    # Pending dedicated support matrix (except HEPData, which has its own)
    _cli("import histfactory", "stable", "import_export"),
    _cli("import trex-config", "stable", "import_export"),
    _cli("import cabinetry", "stable", "import_export"),
    _cli("import hepdata", "stable", "hepdata", HEPDATA_REF),
    _cli("import patchset", "stable", "hepdata", HEPDATA_REF),
    _cli("export histfactory", "stable", "import_export"),
    _cli("convert", "stable", "import_export"),
    _cli("trex import-config", "stable", "import_export"),

    # ── Preprocess ───────────────────────────────────────────────────────
    # Pending dedicated support matrix
    _cli("preprocess smooth", "stable", "preprocess"),
    _cli("preprocess prune", "stable", "preprocess"),

    # ── Infrastructure ───────────────────────────────────────────────────
    # Pending dedicated support matrix
    _cli("config schema", "stable", "infrastructure"),

    _server("GET /v1/health", "stable", "infrastructure"),
    _server("GET /v1/openapi.json", "stable", "infrastructure"),
    _server("GET /v1/tools/schema", "stable", "infrastructure"),
    _server("POST /v1/models", "stable", "infrastructure"),
    _server("GET /v1/models", "stable", "infrastructure"),
    _server("DELETE /v1/models/{id}", "stable", "infrastructure"),
    _server("POST /v1/jobs/submit", "stable", "infrastructure"),
    _server("GET /v1/jobs/{id}", "stable", "infrastructure"),
    _server("DELETE /v1/jobs/{id}", "stable", "infrastructure"),
    _server("GET /v1/jobs", "stable", "infrastructure"),
]


def build_matrix() -> dict:
    return {
        "schema_version": "v1",
        "generated": str(date.today()),
        "description": (
            "Canonical HEP surface inventory. "
            "Every public HEP surface across CLI, Python, tool, and server layers. "
            "Generated by scripts/hep_surface_matrix.py (HEP-ADR-01)."
        ),
        "maturity_classes": {
            "stable": "Promoted with support matrix, acceptance doc, gate, and release evidence.",
            "research": "Visible and functional but not covered by the stable product promise.",
            "internal": "Not public-facing; internal tooling only.",
        },
        "summary": {
            "total": len(SURFACES),
            "by_maturity": {
                m: sum(1 for s in SURFACES if s["maturity_class"] == m)
                for m in ("stable", "research", "internal")
            },
            "by_layer": {
                l: sum(1 for s in SURFACES if s["layer"] == l)
                for l in ("cli", "python", "tool", "server")
            },
            "by_owner": {
                o: sum(1 for s in SURFACES if s["owner_slice"] == o)
                for o in sorted({s["owner_slice"] for s in SURFACES})
            },
        },
        "surfaces": SURFACES,
    }


def main():
    check_only = "--check" in sys.argv

    matrix = build_matrix()

    if check_only:
        if OUTPUT.exists():
            with open(OUTPUT) as f:
                existing = json.load(f)
            if existing["surfaces"] == matrix["surfaces"]:
                print(f"OK: {OUTPUT.name} is up to date ({len(SURFACES)} surfaces)")
                sys.exit(0)
            else:
                print(f"DRIFT: {OUTPUT.name} differs from generated matrix")
                sys.exit(1)
        else:
            print(f"MISSING: {OUTPUT.name} does not exist")
            sys.exit(1)

    with open(OUTPUT, "w") as f:
        json.dump(matrix, f, indent=2, ensure_ascii=False)
        f.write("\n")

    stable = matrix["summary"]["by_maturity"]["stable"]
    research = matrix["summary"]["by_maturity"]["research"]
    print(f"Wrote {OUTPUT.name}: {len(SURFACES)} surfaces "
          f"({stable} stable, {research} research)")


if __name__ == "__main__":
    main()
