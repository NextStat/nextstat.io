from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_repo_root() / "bindings" / "ns-py" / "python"))

import nextstat.m15_report as ns_m15_report


def _load_example(name: str) -> dict[str, object]:
    return json.loads((_repo_root() / "docs" / "specs" / name).read_text(encoding="utf-8"))


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    return (
        _load_example("m15_assessment_table_v1.example.json"),
        _load_example("m15_map_v1.example.json"),
        _load_example("m15_mar_v1.example.json"),
        _load_example("m15_profile_diff_report_v1.example.json"),
        _load_example("m15_bundle_manifest_v1.example.json"),
    )


def test_m15_publishable_pdf_is_deterministic(tmp_path: Path) -> None:
    assessment, map_doc, mar, profile_diff, bundle = _inputs()
    out_a = tmp_path / "m15_report_a.pdf"
    out_b = tmp_path / "m15_report_b.pdf"

    ns_m15_report.write_m15_publishable_pdf(out_a, assessment, map_doc, mar, profile_diff, bundle)
    ns_m15_report.write_m15_publishable_pdf(
        out_b,
        json.dumps(assessment),
        json.dumps(map_doc),
        json.dumps(mar),
        json.dumps(profile_diff),
        json.dumps(bundle),
    )

    first = out_a.read_bytes()
    second = out_b.read_bytes()
    assert first == second
    assert first.startswith(b"%PDF-")


def test_m15_publishable_docx_is_deterministic_and_zip_normalized(tmp_path: Path) -> None:
    assessment, map_doc, mar, profile_diff, bundle = _inputs()
    out_a = tmp_path / "m15_report_a.docx"
    out_b = tmp_path / "m15_report_b.docx"

    ns_m15_report.write_m15_publishable_docx(out_a, assessment, map_doc, mar, profile_diff, bundle)
    ns_m15_report.write_m15_publishable_docx(
        out_b,
        json.dumps(assessment),
        json.dumps(map_doc),
        json.dumps(mar),
        json.dumps(profile_diff),
        json.dumps(bundle),
    )

    first = out_a.read_bytes()
    second = out_b.read_bytes()
    assert first == second

    with zipfile.ZipFile(out_a) as archive:
        names = archive.namelist()
        assert names == sorted(names)
        assert "word/document.xml" in names
        assert "docProps/core.xml" in names
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)


def test_m15_publishable_docx_contains_expected_sections(tmp_path: Path) -> None:
    from docx import Document

    assessment, map_doc, mar, profile_diff, bundle = _inputs()
    out = tmp_path / "m15_report.docx"

    ns_m15_report.write_m15_publishable_docx(out, assessment, map_doc, mar, profile_diff, bundle)

    paragraphs = [p.text.strip() for p in Document(out).paragraphs if p.text.strip()]
    joined = "\n".join(paragraphs)

    assert "ICH M15 Publishable Report" in joined
    assert "Assessment Table Summary" in joined
    assert "Model Analysis Plan" in joined
    assert "Model Analysis Report" in joined
    assert "Profile Diff Summary" in joined
    assert "Bundle Integrity" in joined
    assert "Elena Voss" in joined
    assert "Martin Hale" in joined
    assert "Priya Nair" in joined
