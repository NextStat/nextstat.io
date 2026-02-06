import pytest


def test_report_renderer_requires_matplotlib():
    import nextstat.report as report

    with pytest.raises(ImportError):
        report.render_report(input_dir=".", pdf="out.pdf", svg_dir=None)  # type: ignore[arg-type]

