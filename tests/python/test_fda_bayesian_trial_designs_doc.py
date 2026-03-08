from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WHITEPAPER_PATH = REPO_ROOT / "docs/whitepapers/fda-bayesian-trial-designs.md"


def test_fda_bayesian_trial_designs_whitepaper_exists_and_is_current() -> None:
    assert WHITEPAPER_PATH.exists(), f"missing whitepaper: {WHITEPAPER_PATH}"

    text = WHITEPAPER_PATH.read_text(encoding="utf-8")
    required_markers = [
        "# FDA Bayesian Trial Designs for Drugs and Biologics",
        "March 2026",
        "## Regulatory Baseline (March 2026)",
        "Use of Bayesian Methodology in Clinical Trials of Drugs and Biological Products",
        "2026-01-12",
        "Bayesian Statistics in CDER and CBER: Demonstration Project",
        "2026-01-13",
        "Adaptive Designs for Clinical Trials of Drugs and Biologics",
        "CID Meeting Program",
        "ICH E20",
        "June 2025",
        "expected by October 2026",
        "operating characteristics",
        "prior sensitivity",
        "## What NextStat Can Reuse Today",
        "## Product Thesis",
        "## TDD Plan",
        "## Acceptance Gates",
        "## Non-Goals",
    ]

    for marker in required_markers:
        assert marker in text, f"whitepaper missing marker: {marker}"


def test_fda_bayesian_trial_designs_is_indexed_in_docs() -> None:
    docs_index = (REPO_ROOT / "docs/README.md").read_text(encoding="utf-8")
    biologists_persona = (REPO_ROOT / "docs/personas/biologists.md").read_text(encoding="utf-8")

    expected_ref = "docs/whitepapers/fda-bayesian-trial-designs.md"
    assert expected_ref in docs_index
    assert expected_ref in biologists_persona
