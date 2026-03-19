"""GPU public-access closure invariants (NSS-05).

Validates that the GPU domain now has a credible public-access closure:

  1. GPU contract doc exists.
  2. GPU CI workflows exist.
  3. Claim matrix entry is `sota` with valid proof/access refs.
  4. Installation quickstart documents the canonical source-build GPU path.
  5. Python API documents the narrow shipped CUDA subset.
  6. README guard snippets are present and honest.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

CLAIM_MATRIX = REPO / ".internal" / "claims" / "nextstat_sota_claim_matrix_v1.json"
GPU_CONTRACT = REPO / "docs" / "gpu-contract.md"
RUST_CI = REPO / ".github" / "workflows" / "rust-tests.yml"
UNBINNED_GPU_CI = REPO / ".github" / "workflows" / "unbinned-toy-parity.yml"
INSTALL_QUICKSTART = REPO / "docs" / "tutorials" / "installation-quickstart.md"
PYTHON_API = REPO / "docs" / "references" / "python-api.md"


def _gpu_entry() -> dict:
    data = json.loads(CLAIM_MATRIX.read_text(encoding="utf-8"))
    return {e["domain"]: e for e in data["domains"]}["gpu"]


def test_gpu_contract_doc_exists() -> None:
    assert GPU_CONTRACT.exists(), f"Missing GPU contract: {GPU_CONTRACT}"


def test_gpu_ci_workflows_exist() -> None:
    assert RUST_CI.exists(), f"Missing CI workflow: {RUST_CI}"
    text = RUST_CI.read_text(encoding="utf-8").lower()
    assert "cuda" in text
    assert UNBINNED_GPU_CI.exists(), f"Missing GPU parity workflow: {UNBINNED_GPU_CI}"
    parity_text = UNBINNED_GPU_CI.read_text(encoding="utf-8").lower()
    assert "cuda" in parity_text or "metal" in parity_text


def test_gpu_is_sota() -> None:
    assert _gpu_entry()["status"] == "sota"


def test_gpu_proof_and_access_refs_valid() -> None:
    entry = _gpu_entry()
    for ref in entry["proof_refs"]:
        assert (REPO / ref).exists(), f"GPU proof_ref {ref!r} missing"
    for ref in entry["access_refs"]:
        assert (REPO / ref).exists(), f"GPU access_ref {ref!r} missing"


def test_gpu_install_quickstart_documents_canonical_source_build() -> None:
    text = INSTALL_QUICKSTART.read_text(encoding="utf-8")
    assert "canonical public GPU path is a **source build of the Python bindings**" in text
    assert 'maturin develop --release --features cuda' in text
    assert 'maturin develop --release --features metal' in text
    assert 'nextstat.has_cuda()' in text
    assert 'nextstat.has_metal()' in text
    assert "not part of the default wheel contract" in text


def test_gpu_python_api_documents_narrow_public_cuda_subset() -> None:
    text = PYTHON_API.read_text(encoding="utf-8")
    assert '`device="cuda"` is now part of the stable public surface for a narrow model subset only' in text
    assert "The shipped CUDA subset was direct-verified on Tesla V100 + CUDA 12.6" in text
    assert "nextstat.has_cuda() -> bool" in text
    assert "nextstat.has_metal() -> bool" in text


def test_gpu_guard_snippets_in_readme() -> None:
    entry = _gpu_entry()
    readme = (REPO / entry["guard_doc"]).read_text(encoding="utf-8")
    for snippet in entry["guard_snippets"]:
        assert snippet in readme, f"GPU guard snippet missing: {snippet!r}"
