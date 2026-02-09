from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_record_trex_baseline_module():
    # Load `tests/record_trex_baseline.py` as a module without requiring `tests/` to be a package.
    tests_dir = Path(__file__).resolve().parents[1]
    mod_path = tests_dir / "record_trex_baseline.py"
    spec = importlib.util.spec_from_file_location("record_trex_baseline", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Needed for `from __future__ import annotations` + dataclasses: dataclasses resolves
    # string annotations via `sys.modules[cls.__module__].__dict__`.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_record_trex_baseline_staging_normalizes_combination_xml_paths(tmp_path: Path) -> None:
    rtb = _load_record_trex_baseline_module()

    export_dir = tmp_path / "export"
    stage_dir = tmp_path / "stage"
    (export_dir / "spec").mkdir(parents=True)

    ws_xml = export_dir / "spec" / "ws.xml"
    ws_xml.write_text("<dummy/>", encoding="utf-8")

    # Emulate generators that embed absolute paths in OutputFilePrefix and <Input>.
    combo = export_dir / "combination.xml"
    combo.write_text(
        "\n".join(
            [
                f'<Combination OutputFilePrefix="{(tmp_path / "ABS_OUT" / "ws").as_posix()}">',
                f"  <Input>{ws_xml.resolve().as_posix()}</Input>",
                "</Combination>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    staged = rtb._stage_histfactory_export_dir(export_dir=export_dir, stage_dir=stage_dir)
    staged_text = staged.combination_xml.read_text(encoding="utf-8")

    assert 'OutputFilePrefix="spec/ws"' in staged_text
    assert "<Input>spec/ws.xml</Input>" in staged_text


def test_record_trex_baseline_staging_skips_model_root_files(tmp_path: Path) -> None:
    rtb = _load_record_trex_baseline_module()

    export_dir = tmp_path / "export"
    stage_dir = tmp_path / "stage"
    (export_dir / "spec").mkdir(parents=True)

    (export_dir / "spec" / "ws.xml").write_text("<dummy/>", encoding="utf-8")
    (export_dir / "combination.xml").write_text(
        "\n".join(
            [
                f'<Combination OutputFilePrefix="{(tmp_path / "ABS_OUT" / "ws").as_posix()}">',
                f"  <Input>{(export_dir / 'spec' / 'ws.xml').resolve().as_posix()}</Input>",
                "</Combination>",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # Files that must NOT be copied into the staged tree.
    (export_dir / "spec" / "foo_model.root").write_bytes(b"nope")
    (export_dir / "bar_model.root").write_bytes(b"nope")

    # A normal ROOT file that should be copied.
    (export_dir / "data.root").write_bytes(b"ok")
    (export_dir / "spec" / "other.root").write_bytes(b"ok")

    rtb._stage_histfactory_export_dir(export_dir=export_dir, stage_dir=stage_dir)

    assert not (stage_dir / "spec" / "foo_model.root").exists()
    assert not (stage_dir / "bar_model.root").exists()
    assert (stage_dir / "data.root").exists()
    assert (stage_dir / "spec" / "other.root").exists()
