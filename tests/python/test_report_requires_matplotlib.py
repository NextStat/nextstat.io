import importlib.util
from pathlib import Path

import pytest


def test_report_renderer_requires_matplotlib(tmp_path: Path):
    import nextstat.report as report

    has_mpl = importlib.util.find_spec("matplotlib") is not None
    if not has_mpl:
        with pytest.raises(ImportError):
            report.render_report(input_dir=tmp_path, pdf=tmp_path / "out.pdf", svg_dir=None)
        return

    out_pdf = tmp_path / "out.pdf"
    report.render_report(input_dir=tmp_path, pdf=out_pdf, svg_dir=None)
    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0
