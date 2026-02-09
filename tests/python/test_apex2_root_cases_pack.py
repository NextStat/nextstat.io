from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_apex2_root_cases_minimal_pack_exists_and_has_expected_cases():
    repo = _repo_root()
    cases_path = repo / "tests" / "fixtures" / "trex_parity_pack" / "cases_minimal.json"
    assert cases_path.exists(), f"missing cases pack: {cases_path}"

    obj = json.loads(cases_path.read_text())
    assert isinstance(obj, dict)
    cases = obj.get("cases")
    assert isinstance(cases, list) and cases, "cases_minimal.json must contain non-empty 'cases' list"

    by_name = {c.get("name"): c for c in cases if isinstance(c, dict)}
    assert "simple_fixture" in by_name
    assert "histfactory_fixture" in by_name

    simple = by_name["simple_fixture"]
    assert simple.get("mode") == "pyhf-json"
    pyhf_json = repo / str(simple.get("pyhf_json"))
    assert pyhf_json.exists(), f"missing pyhf fixture: {pyhf_json}"

    hf = by_name["histfactory_fixture"]
    assert hf.get("mode") == "histfactory-xml"
    hf_xml = repo / str(hf.get("histfactory_xml"))
    hf_rootdir = repo / str(hf.get("rootdir"))
    assert hf_xml.exists(), f"missing HistFactory fixture XML: {hf_xml}"
    assert hf_rootdir.exists(), f"missing HistFactory rootdir: {hf_rootdir}"

