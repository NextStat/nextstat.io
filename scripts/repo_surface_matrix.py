"""Repo-wide surface matrix generator.

Produces a canonical inventory of every public surface across all domains.
Sources:
  - hep_surface_matrix_v1.json (lossless import of 141 HEP runtime surfaces)
  - scripts/release_surface_matrix_v1.json (release governance cross-links)
  - Explicitly defined documentation, tutorial, parity, and artifact surfaces

Usage:
  python scripts/repo_surface_matrix.py              # regenerate
  python scripts/repo_surface_matrix.py --check      # validate (no write)
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections import Counter
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


# ── Release surface → HEP owner_slice mapping ────────────────────────────

_RELEASE_TO_HEP_OWNERS: dict[str, list[str]] = {
    "gvm_stable_first": ["gvm"],
    "simplified_likelihood_stable_surface": ["simplified_likelihood"],
    # simplified_likelihood_exporter_surface: standalone (same owner, separate release slice)
    "hepdata_import_stable_surface": ["hepdata"],
    "histfactory_stable_surface": ["histfactory"],
}

# RWS-08: per-surface overrides for import_export owner (mixed governance)
_SURFACE_RELEASE_OVERRIDES: dict[str, tuple[str, str]] = {
    # TREx import utilities → root_trexfitter_parity (optional)
    "import trex-config": ("root_trexfitter_parity", "optional"),
    "trex import-config": ("root_trexfitter_parity", "optional"),
    # HistFactory/cabinetry import/export → histfactory_stable_surface (required)
    "import histfactory": ("histfactory_stable_surface", "required"),
    "export histfactory": ("histfactory_stable_surface", "required"),
    "import cabinetry": ("histfactory_stable_surface", "required"),
    "convert": ("histfactory_stable_surface", "required"),
}

# Release slice → support doc for surfaces imported from HEP with empty support_matrix_ref
_RELEASE_SLICE_SUPPORT_DOC: dict[str, str] = {
    "histfactory_stable_surface": "docs/benchmarks/histfactory-support-matrix-2026-03-17.md",
    "root_trexfitter_parity": "docs/tutorials/root-trexfitter-parity.md",
}

# RWS-10: owner-level support docs for HEP owners without a dedicated release slice.
# These owners are stable-optional (part of the shipped product, don't block release)
# with real governance refs, not migration relabels.
_OWNER_GOVERNANCE: dict[str, dict[str, str]] = {
    "infrastructure": {
        "support_contract_ref": "docs/references/server-api.md",
        "gate_ref": "docs/references/tool-api.md",
    },
    "unbinned": {
        "support_contract_ref": "docs/references/unbinned-parquet-schema.md",
        "gate_ref": "docs/references/cli.md",
    },
    "viz": {
        "support_contract_ref": "docs/references/cli.md",
        "gate_ref": "docs/references/plot-artifacts.md",
    },
    "preprocess": {
        "support_contract_ref": "docs/references/cli.md",
        "gate_ref": "docs/references/cli.md",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────


def _surf(
    surface_id: str,
    domain: str,
    owner_slice: str,
    interface_layer: str,
    surface_kind: str,
    public_status: str,
    release_status: str = "not_release_governed",
    support_contract_ref: str = "",
    acceptance_ref: str = "",
    gate_ref: str = "",
    workflow_ref: str = "",
    validation_bundle_ref: str = "",
    release_surface_ref: str = "",
    notes: str = "",
) -> dict:
    return {
        "surface_id": surface_id,
        "domain": domain,
        "owner_slice": owner_slice,
        "interface_layer": interface_layer,
        "surface_kind": surface_kind,
        "public_status": public_status,
        "release_status": release_status,
        "support_contract_ref": support_contract_ref,
        "acceptance_ref": acceptance_ref,
        "gate_ref": gate_ref,
        "workflow_ref": workflow_ref,
        "validation_bundle_ref": validation_bundle_ref,
        "release_surface_ref": release_surface_ref,
        "notes": notes,
    }


def _doc(
    surface_id: str,
    domain: str,
    owner_slice: str,
    doc_path: str,
    public_status: str = "stable",
    release_status: str = "not_release_governed",
    notes: str = "",
) -> dict:
    return _surf(
        surface_id=surface_id,
        domain=domain,
        owner_slice=owner_slice,
        interface_layer="docs",
        surface_kind="documentation",
        public_status=public_status,
        release_status=release_status,
        support_contract_ref=doc_path,
        notes=notes,
    )


def _tutorial(
    surface_id: str,
    domain: str,
    owner_slice: str,
    doc_path: str,
    public_status: str = "stable",
    release_status: str = "not_release_governed",
    notes: str = "",
) -> dict:
    return _surf(
        surface_id=surface_id,
        domain=domain,
        owner_slice=owner_slice,
        interface_layer="docs",
        surface_kind="tutorial",
        public_status=public_status,
        release_status=release_status,
        support_contract_ref=doc_path,
        notes=notes,
    )


# ── HEP import ────────────────────────────────────────────────────────────


def _import_hep_surfaces(repo: Path) -> list[dict]:
    hep = json.loads((repo / "hep_surface_matrix_v1.json").read_text(encoding="utf-8"))
    release = json.loads(
        (repo / "scripts" / "release_surface_matrix_v1.json").read_text(encoding="utf-8")
    )

    # Build owner_slice → (release_id, required) mapping
    owner_release: dict[str, tuple[str, bool]] = {}
    for rel in release["surfaces"]:
        owners = _RELEASE_TO_HEP_OWNERS.get(rel["id"], [])
        for owner in owners:
            owner_release[owner] = (rel["id"], rel["required_for_release"])

    surfaces: list[dict] = []
    for s in hep["surfaces"]:
        owner = s["owner_slice"]
        name = s["name"]

        # RWS-08: per-surface overrides (import_export mixed governance)
        if name in _SURFACE_RELEASE_OVERRIDES:
            rel_id, release_status = _SURFACE_RELEASE_OVERRIDES[name]
        else:
            rel_id, rel_required = owner_release.get(owner, ("", False))
            if rel_id:
                release_status = "required" if rel_required else "optional"
            else:
                owner_gov = _OWNER_GOVERNANCE.get(owner)
                if owner_gov:
                    # RWS-10: stable-optional with real governance refs
                    release_status = "optional"
                    rel_id = ""
                else:
                    # No governance infrastructure — honest migration
                    release_status = "not_release_governed"
                    rel_id = ""

        # Governed surfaces must have support_contract_ref;
        # fall back to release-slice doc, then owner-level governance
        support_ref = s.get("support_matrix_ref", "")
        gate_ref = ""
        owner_gov = _OWNER_GOVERNANCE.get(owner)
        if not support_ref and release_status in ("required", "optional"):
            if rel_id:
                support_ref = _RELEASE_SLICE_SUPPORT_DOC.get(rel_id, "")
            elif owner_gov:
                support_ref = owner_gov.get("support_contract_ref", "")
                gate_ref = owner_gov.get("gate_ref", "")

        surfaces.append(
            _surf(
                surface_id=f"hep.{owner}.{s['name']}.{s['layer']}",
                domain="hep",
                owner_slice=owner,
                interface_layer=s["layer"],
                surface_kind="runtime",
                public_status=s["maturity_class"],
                release_status=release_status,
                support_contract_ref=support_ref,
                gate_ref=gate_ref,
                validation_bundle_ref="hep_validation_bundle" if release_status in ("required", "optional") else "",
                release_surface_ref=rel_id,
            )
        )

    return surfaces


# ── Sampler runtime surfaces (RWS-05) ────────────────────────────────────


def _sampler_runtime_surfaces() -> list[dict]:
    """NUTS/WALNUTS/MAMS sampler runtime surfaces.

    All three are shipped public stable API via nextstat.sample().
    Governed as optional repo-bundle-only (no dedicated CI workflow yet).
    """
    return [
        _surf(
            surface_id="bayesian.nuts.nuts_core.python",
            domain="bayesian",
            owner_slice="nuts",
            interface_layer="python",
            surface_kind="runtime",
            public_status="stable",
            release_status="optional",
            support_contract_ref="docs/references/nuts-sampler.md",
            gate_ref="scripts/benchmarks/bench_nuts_vs_cmdstan.py",
            notes="RWS-05: NUTS v13, Stan-parity certified",
        ),
        _surf(
            surface_id="bayesian.walnuts.walnuts_core.python",
            domain="bayesian",
            owner_slice="walnuts",
            interface_layer="python",
            surface_kind="runtime",
            public_status="stable",
            release_status="optional",
            support_contract_ref="docs/references/walnuts-sampler.md",
            gate_ref="scripts/benchmarks/bench_walnuts_vs_nuts.py",
            notes="RWS-05: Window-adaptive NUTS (CPU + CUDA)",
        ),
        _surf(
            surface_id="bayesian.mams.mams_core.python",
            domain="bayesian",
            owner_slice="mams",
            interface_layer="python",
            surface_kind="runtime",
            public_status="stable",
            release_status="optional",
            support_contract_ref="docs/references/walnuts-sampler.md",
            gate_ref="scripts/benchmarks/bench_sampler_matrix.py",
            notes="RWS-05: Metropolis-Adjusted Multivariate Slice sampler",
        ),
    ]


# ── Ads governed surfaces ─────────────────────────────────────────────────


def _ads_governed_surfaces(repo: Path) -> list[dict]:
    """Ads surfaces with full internal governance infrastructure (CI, gates, baselines).

    Ads is internal-only — these are NOT public stable surfaces.
    They have strong engineering quality (CI workflows, gate scripts, baselines)
    but are classified as internal, not part of the public release contract.
    """
    return [
        # ── ads-timeseries ──
        _surf(
            surface_id="ads.timeseries.ads_timeseries_core.python",
            domain="ads",
            owner_slice="timeseries",
            interface_layer="python",
            surface_kind="runtime",
            public_status="internal",
            release_status="not_release_governed",
            support_contract_ref="docs/benchmarks/ads-timeseries-support-matrix-2026-03-08.md",
            acceptance_ref="docs/benchmarks/ads-timeseries-stable-surface-acceptance-2026-03-08.md",
            gate_ref="scripts/benchmarks/ads_timeseries_stable_surface_gate.sh",
            workflow_ref=".github/workflows/ads-timeseries-stable-surface.yml",
            notes="Internal: strong engineering governance but not public-facing",
        ),
        # ── ads-variance-reduction ──
        _surf(
            surface_id="ads.variance_reduction.ads_variance_reduction_core.python",
            domain="ads",
            owner_slice="variance_reduction",
            interface_layer="python",
            surface_kind="runtime",
            public_status="internal",
            release_status="not_release_governed",
            support_contract_ref="docs/benchmarks/ads-variance-reduction-stable-surface-acceptance-2026-03-09.md",
            acceptance_ref="docs/benchmarks/ads-variance-reduction-stable-surface-acceptance-2026-03-09.md",
            gate_ref="scripts/benchmarks/ads_variance_reduction_stable_surface_gate.sh",
            workflow_ref=".github/workflows/ads-variance-reduction-stable-surface.yml",
            notes="Internal: strong engineering governance but not public-facing",
        ),
    ]


# ── Bayesian design governed surfaces (RWS-04) ──────────────────────────


def _bayesian_design_runtime_surface(repo: Path) -> list[dict]:
    """Bayesian design runtime surface — FDA-aligned trial design Python API.

    One release slice (bayesian_design_stable_surface) covers the runtime API
    and all documentation facets. Optional for release.
    """
    release = json.loads(
        (repo / "scripts" / "release_surface_matrix_v1.json").read_text(encoding="utf-8")
    )
    release_by_id = {r["id"]: r for r in release["surfaces"]}

    rel = release_by_id.get("bayesian_design_stable_surface")
    if rel:
        release_status = "required" if rel["required_for_release"] else "optional"
    else:
        release_status = "not_release_governed"

    return [
        _surf(
            surface_id="bayesian.bayes_design.bayes_design_core.python",
            domain="bayesian",
            owner_slice="bayes_design",
            interface_layer="python",
            surface_kind="runtime",
            public_status="stable",
            release_status=release_status,
            support_contract_ref="docs/references/bayesian-trial-design-artifacts.md",
            acceptance_ref="docs/benchmarks/bayesian-design-release-pr-checklist-2026-03-08.md",
            gate_ref="scripts/benchmarks/bench_bayesian_design_report_bundle.py",
            release_surface_ref="bayesian_design_stable_surface" if rel else "",
            notes="RWS-04: FDA Bayesian trial design core API",
        ),
    ]


# ── Standalone release surfaces (not in HEP matrix) ──────────────────────


def _standalone_release_surfaces() -> list[dict]:
    return [
        _surf(
            surface_id="pharma.m15_reporting.m15_reporting_stable_surface.cli",
            domain="pharma",
            owner_slice="m15_reporting",
            interface_layer="cli",
            surface_kind="runtime",
            public_status="stable",
            release_status="required",
            support_contract_ref="docs/references/m15-reporting.md",
            gate_ref="scripts/benchmarks/m15_reporting_stable_surface_gate.sh",
            workflow_ref=".github/workflows/release-candidate.yml",
            validation_bundle_ref="hep_validation_bundle",
            release_surface_ref="m15_reporting_stable_surface",
        ),
        _surf(
            surface_id="hep.simplified_likelihood.exporter_surface.tool",
            domain="hep",
            owner_slice="simplified_likelihood",
            interface_layer="tool",
            surface_kind="runtime",
            public_status="stable",
            release_status="required",
            support_contract_ref="docs/benchmarks/simplified-likelihood-support-matrix-2026-03-08.md",
            gate_ref="scripts/benchmarks/simplified_likelihood_exporter_surface_gate.sh",
            workflow_ref=".github/workflows/simplified-likelihood-exporter-surface.yml",
            validation_bundle_ref="hep_validation_bundle",
            release_surface_ref="simplified_likelihood_exporter_surface",
            notes="SL exporter: separate release slice from SL core, same owner",
        ),
        _surf(
            surface_id="hep.histfactory.root_trexfitter_parity.docs",
            domain="hep",
            owner_slice="histfactory",
            interface_layer="docs",
            surface_kind="parity",
            public_status="stable",
            release_status="optional",
            support_contract_ref="docs/tutorials/root-trexfitter-parity.md",
            release_surface_ref="root_trexfitter_parity",
            notes="ROOT/TREx parity is optional for release; informational only",
        ),
    ]


# ── Migration runtime surfaces (day-1 bulk import, RWS-03..06 burn-down) ─


def _migration_runtime_surfaces() -> list[dict]:
    """Runtime surfaces imported as stable+not_release_governed per ADR.

    These represent domains where public stable claims already exist but
    machine-readable release governance has not yet been wired. Each row
    must be burned down by the corresponding RWS phase (03-06).
    """
    return [
        # ── Pharma PK/NLME (RWS-06: promoted to optional with governance refs) ──
        _surf(
            surface_id="pharma.pk_nlme.foce_saem_core.python",
            domain="pharma",
            owner_slice="pk_nlme",
            interface_layer="python",
            surface_kind="runtime",
            public_status="stable",
            release_status="optional",
            support_contract_ref="docs/releases/pharma-release-evidence-policy.md",
            gate_ref="docs/benchmarks/pharma-parity-report.md",
            notes="RWS-06: FOCE/SAEM core API, stable-optional",
        ),
        # ── Pharma survival (RWS-06: promoted to optional with governance refs) ──
        _surf(
            surface_id="pharma.survival.survival_core.python",
            domain="pharma",
            owner_slice="survival",
            interface_layer="python",
            surface_kind="runtime",
            public_status="stable",
            release_status="optional",
            support_contract_ref="docs/releases/pharma-release-evidence-policy.md",
            gate_ref="scripts/benchmarks/bench_survival_vs_classification.py",
            notes="RWS-06: survival core API, stable-optional",
        ),
        # ── Ads churn (RWS-03: reclassified as internal — ads is internal-only) ──
        _surf(
            surface_id="ads.churn.churn_core.python",
            domain="ads",
            owner_slice="churn",
            interface_layer="python",
            surface_kind="runtime",
            public_status="internal",
            release_status="not_release_governed",
            notes="RWS-03: internal-only (no public contract, no governance infrastructure)",
        ),
    ]


# ── Documentation surfaces (day-1 bulk import) ───────────────────────────


def _documentation_surfaces() -> list[dict]:
    return [
        # ── Platform references (RWS-07: promoted to optional) ──
        _doc("platform.cli.cli_reference.docs", "platform", "cli",
             "docs/references/cli.md",
             release_status="optional"),
        _doc("platform.python_api.python_api_reference.docs", "platform", "python_api",
             "docs/references/python-api.md",
             release_status="optional"),
        _doc("platform.rust_api.rust_api_reference.docs", "platform", "rust_api",
             "docs/references/rust-api.md",
             release_status="optional"),
        _doc("platform.io.arrow_parquet_io.docs", "platform", "io",
             "docs/references/arrow-parquet-io.md",
             release_status="optional"),
        _doc("platform.config.analysis_config.docs", "platform", "config",
             "docs/references/analysis-config.md",
             release_status="optional"),
        _doc("platform.optimizer.optimizer_convergence.docs", "platform", "optimizer",
             "docs/references/optimizer-convergence.md",
             release_status="optional"),
        _doc("platform.viz.plot_artifacts.docs", "platform", "viz",
             "docs/references/plot-artifacts.md",
             release_status="optional"),
        _doc("platform.packaging.python_packaging.docs", "platform", "packaging",
             "docs/references/python-packaging.md",
             release_status="optional"),
        # ── HEP references (RWS-07: promoted to optional) ──
        _doc("hep.histfactory.root_histfactory_comparison.docs", "hep", "histfactory",
             "docs/references/root-histfactory-comparison.md",
             release_status="optional"),
        _doc("hep.simplified_likelihood.sl_artifacts.docs", "hep", "simplified_likelihood",
             "docs/references/simplified-likelihood-artifacts.md",
             release_status="optional"),
        # ── Bayesian references ──
        _doc("bayesian.nuts.nuts_sampler.docs", "bayesian", "nuts",
             "docs/references/nuts-sampler.md",
             release_status="optional"),
        _doc("bayesian.walnuts.walnuts_sampler.docs", "bayesian", "walnuts",
             "docs/references/walnuts-sampler.md",
             release_status="optional"),
        _doc("bayesian.bayes_design.trial_design_artifacts.docs", "bayesian", "bayes_design",
             "docs/references/bayesian-trial-design-artifacts.md",
             release_status="optional"),
        # ── Pharma references (RWS-06: promoted to required, linked to release slice) ──
        _surf(
            surface_id="pharma.m15_reporting.m15_reporting_reference.docs",
            domain="pharma", owner_slice="m15_reporting",
            interface_layer="docs", surface_kind="documentation",
            public_status="stable", release_status="required",
            support_contract_ref="docs/references/m15-reporting.md",
            validation_bundle_ref="hep_validation_bundle",
            release_surface_ref="m15_reporting_stable_surface",
        ),
        # ── HEP acceptance specs (RWS-07: promoted to optional) ──
        _doc("hep.hepdata.hepdata_import_acceptance.docs", "hep", "hepdata",
             "docs/specs/hep/hepdata_import_acceptance_v1.md",
             release_status="optional",
             notes="acceptance spec for hepdata import"),
        # ── Bayesian acceptance specs (RWS-04: promoted to optional) ──
        _surf(
            surface_id="bayesian.bayes_design.report_acceptance.docs",
            domain="bayesian", owner_slice="bayes_design",
            interface_layer="docs", surface_kind="documentation",
            public_status="stable", release_status="optional",
            support_contract_ref="docs/specs/pharma/bayesian_design_report_acceptance_v0.md",
            acceptance_ref="docs/specs/pharma/bayesian_design_report_acceptance_v0.md",
        ),
        _surf(
            surface_id="bayesian.bayes_design.appendix_render_acceptance.docs",
            domain="bayesian", owner_slice="bayes_design",
            interface_layer="docs", surface_kind="documentation",
            public_status="stable", release_status="optional",
            support_contract_ref="docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md",
            acceptance_ref="docs/specs/pharma/bayesian_design_appendix_render_acceptance_v0.md",
        ),
        _surf(
            surface_id="bayesian.bayes_design.regulatory_appendix_acceptance.docs",
            domain="bayesian", owner_slice="bayes_design",
            interface_layer="docs", surface_kind="documentation",
            public_status="stable", release_status="optional",
            support_contract_ref="docs/specs/pharma/bayesian_design_regulatory_appendix_acceptance_v0.md",
            acceptance_ref="docs/specs/pharma/bayesian_design_regulatory_appendix_acceptance_v0.md",
        ),
        _surf(
            surface_id="bayesian.bayes_design.bundle_acceptance.docs",
            domain="bayesian", owner_slice="bayes_design",
            interface_layer="docs", surface_kind="documentation",
            public_status="stable", release_status="optional",
            support_contract_ref="docs/specs/pharma/bayesian_design_report_bundle_acceptance_v0.md",
            acceptance_ref="docs/specs/pharma/bayesian_design_report_bundle_acceptance_v0.md",
        ),
        _surf(
            surface_id="bayesian.bayes_design.validation_pack_acceptance.docs",
            domain="bayesian", owner_slice="bayes_design",
            interface_layer="docs", surface_kind="documentation",
            public_status="stable", release_status="optional",
            support_contract_ref="docs/specs/pharma/bayesian_design_validation_pack_acceptance_v0.md",
            acceptance_ref="docs/specs/pharma/bayesian_design_validation_pack_acceptance_v0.md",
        ),
        _surf(
            surface_id="bayesian.bayes_design.prior_conflict_acceptance.docs",
            domain="bayesian", owner_slice="bayes_design",
            interface_layer="docs", surface_kind="documentation",
            public_status="stable", release_status="optional",
            support_contract_ref="docs/specs/pharma/bayesian_prior_conflict_diagnostic_acceptance_v0.md",
            acceptance_ref="docs/specs/pharma/bayesian_prior_conflict_diagnostic_acceptance_v0.md",
        ),
        # ── Benchmark runtime gates (RWS-04: promoted to optional) ──
        _doc("bayesian.bayes_design.packaging_runtime_gate.docs", "bayesian", "bayes_design",
             "docs/benchmarks/bayesian-design-report-packaging-runtime-gate.md",
             release_status="optional"),
        _doc("hep.hepdata.import_runtime_gate.docs", "hep", "hepdata",
             "docs/benchmarks/hepdata-import-runtime-gate.md",
             release_status="optional"),
        _surf(
            surface_id="pharma.m15_reporting.runtime_gate.docs",
            domain="pharma", owner_slice="m15_reporting",
            interface_layer="docs", surface_kind="documentation",
            public_status="stable", release_status="required",
            support_contract_ref="docs/benchmarks/m15-reporting-runtime-gate.md",
            validation_bundle_ref="hep_validation_bundle",
            release_surface_ref="m15_reporting_stable_surface",
        ),
        # ── Guide (RWS-07: promoted to optional) ──
        _doc("platform.hpc.htcondor_guide.docs", "platform", "hpc",
             "docs/guides/htcondor-hpc.md",
             release_status="optional"),
        # ── HEP neural density estimation (RWS-07: promoted to optional) ──
        _doc("hep.unbinned.neural_density_estimation.docs", "hep", "unbinned",
             "docs/neural-density-estimation.md",
             release_status="optional"),
        # ── GVM guides (RWS-07: promoted to optional) ──
        _doc("hep.gvm.external_validation_kit.docs", "hep", "gvm",
             "docs/guides/gvm-external-validation-kit.md",
             release_status="optional"),
        _doc("hep.gvm.external_validator_outreach.docs", "hep", "gvm",
             "docs/guides/gvm-external-validator-outreach-pack.md",
             release_status="optional"),
        _doc("hep.gvm.external_validation_tracker.docs", "hep", "gvm",
             "docs/guides/gvm-external-validation-tracker-template.md",
             release_status="optional"),
    ]


def _tutorial_surfaces() -> list[dict]:
    return [
        # ── HEP tutorials (RWS-07: promoted to optional) ──
        _tutorial("hep.histfactory.frequentist_tutorial.docs", "hep", "histfactory",
                  "docs/tutorials/phase-3.1-frequentist.md",
                  release_status="optional"),
        _tutorial("hep.histfactory.trex_replacement.docs", "hep", "histfactory",
                  "docs/tutorials/trex-replacement-workflow.md",
                  release_status="optional"),
        # ── Pharma tutorials (RWS-06: promoted to optional) ──
        _tutorial("pharma.pk_nlme.pharma_pk.docs", "pharma", "pk_nlme",
                  "docs/tutorials/pharma-pk.md",
                  release_status="optional"),
        _tutorial("pharma.pk_nlme.phase13_pk.docs", "pharma", "pk_nlme",
                  "docs/tutorials/phase-13-pk.md",
                  release_status="optional"),
        _tutorial("pharma.pk_nlme.phase13_nlme.docs", "pharma", "pk_nlme",
                  "docs/tutorials/phase-13-nlme.md",
                  release_status="optional"),
        _tutorial("pharma.survival.pharma_survival.docs", "pharma", "survival",
                  "docs/tutorials/pharma-survival.md",
                  release_status="optional"),
        _tutorial("pharma.pk_nlme.nonmem_migration.docs", "pharma", "pk_nlme",
                  "docs/tutorials/nonmem-migration.md",
                  release_status="optional"),
        _tutorial("pharma.pk_nlme.scm_tutorial.docs", "pharma", "pk_nlme",
                  "docs/tutorials/scm-tutorial.md",
                  release_status="optional"),
        # ── HEP GVM tutorials/quickstarts (RWS-07: promoted to optional) ──
        _tutorial("hep.gvm.gvm_measurement_combinations.docs", "hep", "gvm",
                  "docs/tutorials/hep-gvm-measurement-combinations.md",
                  release_status="optional"),
        _tutorial("hep.gvm.gvm_stable_first_quickstart.docs", "hep", "gvm",
                  "docs/quickstarts/hep-gvm-stable-first.md",
                  release_status="optional"),
        # ── Ads tutorials (internal-only) ──
        _tutorial("ads.churn.churn_subscription.docs", "ads", "churn",
                  "docs/tutorials/churn-subscription.md",
                  public_status="internal",
                  notes="RWS-03: internal-only (ads domain is not public-facing)"),
    ]


# ── Build matrix ──────────────────────────────────────────────────────────


def build_matrix(repo: Path | None = None) -> dict:
    repo = repo or _repo_root()

    surfaces: list[dict] = []
    surfaces.extend(_import_hep_surfaces(repo))
    surfaces.extend(_standalone_release_surfaces())
    surfaces.extend(_bayesian_design_runtime_surface(repo))
    surfaces.extend(_sampler_runtime_surfaces())
    surfaces.extend(_ads_governed_surfaces(repo))
    surfaces.extend(_migration_runtime_surfaces())
    surfaces.extend(_documentation_surfaces())
    surfaces.extend(_tutorial_surfaces())

    # Sort for stable output
    surfaces.sort(key=lambda s: (s["domain"], s["owner_slice"], s["surface_id"]))

    total = len(surfaces)
    summary = {
        "total": total,
        "by_domain": dict(Counter(s["domain"] for s in surfaces).most_common()),
        "by_surface_kind": dict(Counter(s["surface_kind"] for s in surfaces).most_common()),
        "by_public_status": dict(Counter(s["public_status"] for s in surfaces).most_common()),
        "by_release_status": dict(Counter(s["release_status"] for s in surfaces).most_common()),
    }

    bundle_slot_contract = {
        "version": "v1",
        "required_fields": [
            "surface_id",
            "domain",
            "owner_slice",
            "surface_kind",
            "public_status",
            "release_status",
        ],
        "validation_bundle_slots": [
            "hep_validation_bundle",
        ],
    }

    return {
        "schema_version": "nextstat.repo_surface_matrix.v1",
        "generated": datetime.date.today().isoformat(),
        "description": (
            "Canonical repo-wide surface inventory. "
            "Every public surface, its domain, maturity, and release governance."
        ),
        "summary": summary,
        "surfaces": surfaces,
        "bundle_slot_contract": bundle_slot_contract,
    }


# ── Check mode ────────────────────────────────────────────────────────────


def _load_existing(repo: Path) -> dict | None:
    path = repo / "repo_surface_matrix_v1.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def check_matrix(repo: Path | None = None) -> bool:
    repo = repo or _repo_root()
    existing = _load_existing(repo)
    if existing is None:
        print("repo_surface_matrix_v1.json does not exist.", file=sys.stderr)
        return False

    fresh = build_matrix(repo)

    existing_ids = {s["surface_id"] for s in existing["surfaces"]}
    fresh_ids = {s["surface_id"] for s in fresh["surfaces"]}

    missing = fresh_ids - existing_ids
    extra = existing_ids - fresh_ids

    ok = True
    if missing:
        print(f"Missing surfaces in committed file: {sorted(missing)}", file=sys.stderr)
        ok = False
    if extra:
        print(f"Extra surfaces in committed file: {sorted(extra)}", file=sys.stderr)
        ok = False

    # Check field-level parity for shared surfaces
    existing_by_id = {s["surface_id"]: s for s in existing["surfaces"]}
    fresh_by_id = {s["surface_id"]: s for s in fresh["surfaces"]}
    for sid in sorted(fresh_ids & existing_ids):
        e = existing_by_id[sid]
        f = fresh_by_id[sid]
        for key in (
            "domain", "owner_slice", "interface_layer", "surface_kind",
            "public_status", "release_status",
        ):
            if e.get(key) != f.get(key):
                print(
                    f"Drift: {sid}.{key}: committed={e.get(key)!r} vs generated={f.get(key)!r}",
                    file=sys.stderr,
                )
                ok = False

    if ok:
        print(f"repo_surface_matrix_v1: ok ({existing['summary']['total']} surfaces)")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Repo-wide surface matrix generator.")
    parser.add_argument("--check", action="store_true", help="Validate without writing.")
    args = parser.parse_args()

    repo = _repo_root()

    if args.check:
        return 0 if check_matrix(repo) else 1

    matrix = build_matrix(repo)
    out = repo / "repo_surface_matrix_v1.json"
    out.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({matrix['summary']['total']} surfaces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
