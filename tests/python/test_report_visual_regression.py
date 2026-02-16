from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ahash16_hex(path: Path) -> str:
    from PIL import Image

    img = Image.open(path).convert("L").resize((16, 16))
    if hasattr(img, "get_flattened_data"):
        pixels = list(img.get_flattened_data())
    else:
        pixels = list(img.getdata())
    avg = sum(float(p) for p in pixels) / float(len(pixels))
    bits = "".join("1" if float(p) >= avg else "0" for p in pixels)
    return f"{int(bits, 2):064x}"


def _hex_hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError(f"hex length mismatch: {len(a)} != {len(b)}")
    return sum((int(ca, 16) ^ int(cb, 16)).bit_count() for ca, cb in zip(a, b))


def _svg_metrics(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    return {
        "size_bytes": int(path.stat().st_size),
        "path_count": int(text.count("<path")),
        "text_count": int(text.count("<text")),
        "use_count": int(text.count("<use")),
        "fill_count": int(text.count("fill:")),
        "stroke_count": int(text.count("stroke:")),
        "nextstat": int("NEXTSTAT" in text),
        "mc_unc": int("MC unc." in text),
        "channel": int("Channel:" in text),
    }


def _render_fixture_report(tmp_path: Path) -> Path:
    import nextstat.report as report

    src = _repo_root() / "tests" / "fixtures" / "trex_report_goldens" / "histfactory_v0"
    out_dir = tmp_path / "viz_regression"
    svg_dir = out_dir / "svg"
    pdf = out_dir / "report.pdf"

    # Keep matplotlib cache writable/stable inside test sandbox.
    os.environ.setdefault("MPLCONFIGDIR", str(tmp_path / "mplconfig"))

    report.render_report(input_dir=src, pdf=pdf, svg_dir=svg_dir, png_dir=svg_dir, png_dpi=220)
    assert pdf.exists() and pdf.stat().st_size > 0
    return svg_dir


def _golden_path() -> Path:
    return (
        _repo_root()
        / "tests"
        / "fixtures"
        / "trex_report_goldens"
        / "histfactory_v0"
        / "visual_regression_v1.json"
    )


def test_report_visual_regression_png_svg(tmp_path: Path):
    # This regression suite validates render style; skip in minimal envs.
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib not installed")
    if importlib.util.find_spec("PIL") is None:
        pytest.skip("Pillow not installed")

    svg_dir = _render_fixture_report(tmp_path)
    golden_file = _golden_path()
    golden = json.loads(golden_file.read_text(encoding="utf-8"))

    got_png: dict[str, dict[str, int | str]] = {}
    for name in sorted(golden["png"].keys()):
        p = svg_dir / name
        assert p.exists(), f"missing rendered PNG: {p}"
        got_png[name] = {
            "ahash16": _ahash16_hex(p),
            "size_bytes": int(p.stat().st_size),
        }

    got_svg: dict[str, dict[str, int]] = {}
    for name in sorted(golden["svg"].keys()):
        p = svg_dir / name
        assert p.exists(), f"missing rendered SVG: {p}"
        got_svg[name] = _svg_metrics(p)

    if os.environ.get("NS_RECORD_VISUAL_GOLDENS") == "1":
        golden_file.write_text(
            json.dumps({"png": got_png, "svg": got_svg}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    # PNG tolerance: average-hash Hamming distance + size envelope.
    for name, want in golden["png"].items():
        got = got_png[name]
        dist = _hex_hamming(str(got["ahash16"]), str(want["ahash16"]))
        max_hamming = int(want.get("max_hamming", 24))
        assert (
            dist <= max_hamming
        ), f"{name}: perceptual hash drift too large (hamming={dist}, allowed={max_hamming})"

        got_size = int(got["size_bytes"])
        want_size = int(want["size_bytes"])
        size_tol = int(want.get("size_tolerance_bytes", max(2048, int(0.25 * want_size))))
        assert abs(got_size - want_size) <= size_tol, (
            f"{name}: size drift too large (got={got_size}, want={want_size}, tol={size_tol})"
        )

    # SVG tolerance: structural metrics, not raw text hash.
    for name, want in golden["svg"].items():
        got = got_svg[name]
        for key in ["nextstat", "mc_unc", "channel"]:
            assert int(got[key]) == int(want[key]), f"{name}: marker `{key}` mismatch"

        for key in ["path_count", "text_count", "use_count", "fill_count", "stroke_count"]:
            g = int(got[key])
            w = int(want[key])
            rel = float(want.get("relative_tolerance", 0.35))
            tol = max(3, int(round(w * rel)))
            assert abs(g - w) <= tol, f"{name}: `{key}` drift too large (got={g}, want={w}, tol={tol})"

        g_size = int(got["size_bytes"])
        w_size = int(want["size_bytes"])
        size_rel = float(want.get("size_relative_tolerance", 0.35))
        size_tol = max(4096, int(round(w_size * size_rel)))
        assert abs(g_size - w_size) <= size_tol, (
            f"{name}: svg size drift too large (got={g_size}, want={w_size}, tol={size_tol})"
        )
