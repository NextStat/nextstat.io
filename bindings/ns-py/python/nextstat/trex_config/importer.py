from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .parser import TrexConfigBlock, TrexConfigDoc, TrexConfigEntry


class TrexConfigImportError(ValueError):
    pass


@dataclass(frozen=True)
class _Source:
    kind: str
    name: str | None
    key: str
    line: int
    value: str


@dataclass(frozen=True)
class _Mapping:
    target: str
    source: _Source


def _lower(s: str) -> str:
    return s.strip().lower()


def _unescape(s: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue
        if i + 1 >= len(s):
            out.append("\\")
            i += 1
            continue
        nxt = s[i + 1]
        if nxt in ('"', "'", "\\"):
            out.append(nxt)
            i += 2
            continue
        out.append("\\")
        out.append(nxt)
        i += 2
    return "".join(out)


def _parse_atom(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    if s[0] in ('"', "'") and len(s) >= 2 and s[-1] == s[0]:
        return _unescape(s[1:-1])
    return s


def _split_list(raw: str) -> list[str]:
    """Split TREx list values (comma/whitespace/semicolon separators, quote-aware)."""
    v = raw.strip()
    inner = v
    if v.startswith("[") and v.endswith("]") and len(v) >= 2:
        inner = v[1:-1]

    out: list[str] = []
    cur: list[str] = []
    in_quote: str | None = None
    esc = False

    def push() -> None:
        tok = "".join(cur).strip()
        cur.clear()
        if tok:
            out.append(_parse_atom(tok))

    for ch in inner:
        if esc:
            cur.append(ch)
            esc = False
            continue
        if ch == "\\":
            cur.append(ch)
            esc = True
            continue
        if in_quote is not None:
            cur.append(ch)
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ('"', "'"):
            cur.append(ch)
            in_quote = ch
            continue
        if ch in (",", ";") or ch.isspace():
            push()
            continue
        cur.append(ch)
    push()

    return [x for x in out if x.strip()]


def _as_bool(raw: str) -> Optional[bool]:
    v = raw.strip().lower()
    if v == "":
        return True
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return None


def _as_f64(raw: str) -> float:
    try:
        return float(raw.strip())
    except Exception as e:  # pragma: no cover
        raise TrexConfigImportError(f"invalid float: {raw!r}: {e}") from e


def _as_binning_edges(raw: str) -> list[float]:
    v = raw.strip()
    if v.startswith("[") and v.endswith("]") and len(v) >= 2:
        v = v[1:-1]
    parts = [p for p in (x.strip() for x in v.replace(",", " ").replace(";", " ").split()) if p]
    if len(parts) < 2:
        raise TrexConfigImportError(f"binning must have >= 2 edges, got: {raw!r}")
    return [_as_f64(p) for p in parts]


def _last_any(block: TrexConfigBlock, keys: Iterable[str]) -> TrexConfigEntry | None:
    for k in keys:
        e = block.last(k)
        if e is not None:
            return e
    return None


def _all_any(block: TrexConfigBlock, keys: Iterable[str]) -> list[TrexConfigEntry]:
    out: list[TrexConfigEntry] = []
    for k in keys:
        out.extend(block.entries(k))
    return out


def _collect_unmapped(
    *,
    block: TrexConfigBlock,
    used_keys: set[str],
) -> list[_Source]:
    out: list[_Source] = []
    for key_lc, items in block._attrs.items():  # noqa: SLF001 (internal use; stable within module)
        if key_lc in used_keys:
            continue
        for e in items:
            out.append(
                _Source(
                    kind=block.kind,
                    name=block.name,
                    key=e.key,
                    line=e.line,
                    value=e.value.raw,
                )
            )
    return out


def trex_doc_to_analysis_spec_v0(
    doc: TrexConfigDoc,
    *,
    source_path: Path | None = None,
    out_path: Path | None = None,
    threads: int = 1,
    workspace_out: str = "tmp/trex_workspace.json",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Best-effort conversion: TREx `.config` doc -> analysis spec v0 + mapping report.

    This maps only the subset supported by the v0 TREx replacement pipeline (ReadFrom=NTUP
    and a HIST wrapper over HistFactory exports).
    Unsupported keys are recorded in the report.
    """
    if threads < 1:
        raise TrexConfigImportError("--threads must be >= 1")

    blocks = doc.blocks
    if not blocks:
        raise TrexConfigImportError("empty config")

    # Treat Global + Job/Fit blocks as global-ish sources (last-write-wins in order).
    globals_blocks = [b for b in blocks if b.kind.lower() in ("global", "job", "fit")]
    region_blocks = [b for b in blocks if b.kind.lower() == "region"]
    sample_blocks = [b for b in blocks if b.kind.lower() == "sample"]
    syst_blocks = [b for b in blocks if b.kind.lower() == "systematic"]

    mapped: list[_Mapping] = []
    unmapped: list[_Source] = []
    notes: list[str] = []

    def take_global(keys: list[str]) -> TrexConfigEntry | None:
        hit: TrexConfigEntry | None = None
        for b in globals_blocks:
            e = _last_any(b, keys)
            if e is not None:
                hit = e
        return hit

    # Defaults match Rust importer behavior.
    read_from_e = take_global(["ReadFrom"])
    read_from = (read_from_e.value.raw.strip() if read_from_e else "NTUP").upper()
    if read_from not in ("NTUP", "HIST"):
        raise TrexConfigImportError(f"ReadFrom={read_from} is not supported yet (only NTUP or HIST)")
    if read_from_e is not None:
        mapped.append(
            _Mapping(
                target="inputs.trex_config_yaml.read_from",
                source=_Source(
                    kind="Global",
                    name=None,
                    key=read_from_e.key,
                    line=read_from_e.line,
                    value=read_from_e.value.raw,
                ),
            )
        )

    if read_from == "HIST":
        # HIST wrapper: use an existing HistFactory export dir containing combination.xml.
        histo_e = take_global(["HistoPath", "HistPath", "ExportDir"])
        combo_e = take_global(["CombinationXml", "CombinationXML", "HistFactoryXml"])
        if histo_e is None and combo_e is None:
            raise TrexConfigImportError("ReadFrom=HIST requires HistoPath/ExportDir or CombinationXml")

        meas_e = take_global(["Measurement"])
        poi_e = take_global(["POI", "Poi"])

        # base_dir: relative path from out_path directory to source config directory if possible.
        base_dir_value: str | None = None
        if source_path is not None:
            cfg_dir = source_path.parent
            if out_path is None:
                base_dir_value = "."
            else:
                out_dir = out_path.parent
                try:
                    rel = os.path.relpath(str(cfg_dir.resolve()), str(out_dir.resolve()))
                    base_dir_value = "." if rel == "." else rel
                except Exception:
                    base_dir_value = str(cfg_dir)

        trex_yaml: dict[str, Any] = {
            "base_dir": base_dir_value,
            "read_from": "HIST",
        }
        if histo_e is not None and histo_e.value.raw.strip():
            trex_yaml["histo_path"] = _parse_atom(histo_e.value.raw)
            mapped.append(
                _Mapping(
                    target="inputs.trex_config_yaml.histo_path",
                    source=_Source("Global", None, histo_e.key, histo_e.line, histo_e.value.raw),
                )
            )
        if combo_e is not None and combo_e.value.raw.strip():
            trex_yaml["combination_xml"] = _parse_atom(combo_e.value.raw)
            mapped.append(
                _Mapping(
                    target="inputs.trex_config_yaml.combination_xml",
                    source=_Source("Global", None, combo_e.key, combo_e.line, combo_e.value.raw),
                )
            )

        if meas_e is not None and meas_e.value.raw.strip():
            trex_yaml["measurement"] = meas_e.value.raw.strip()
            mapped.append(
                _Mapping(
                    target="inputs.trex_config_yaml.measurement",
                    source=_Source("Global", None, meas_e.key, meas_e.line, meas_e.value.raw),
                )
            )
        if poi_e is not None and poi_e.value.raw.strip():
            trex_yaml["poi"] = poi_e.value.raw.strip()
            mapped.append(
                _Mapping(
                    target="inputs.trex_config_yaml.poi",
                    source=_Source("Global", None, poi_e.key, poi_e.line, poi_e.value.raw),
                )
            )

        spec: dict[str, Any] = {
            "$schema": "https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json",
            "schema_version": "trex_analysis_spec_v0",
            "analysis": {"name": "TREx Config (HIST)", "description": "Converted from TREx config (HIST).", "tags": ["trex-config", "hist"]},
            "inputs": {"mode": "trex_config_yaml", "trex_config_yaml": trex_yaml},
            "execution": {
                "determinism": {"threads": int(threads)},
                "import": {"enabled": True, "output_json": workspace_out},
                "fit": {"enabled": False, "output_json": "tmp/trex_fit.json"},
                "profile_scan": {"enabled": False, "start": 0.0, "stop": 5.0, "points": 21, "output_json": "tmp/trex_scan.json"},
                "report": {
                    "enabled": False,
                    "out_dir": "tmp/trex_report",
                    "overwrite": True,
                    "include_covariance": False,
                    "histfactory_xml": None,
                    "render": {"enabled": False, "pdf": None, "svg_dir": None, "python": None},
                    "skip_uncertainty": False,
                    "uncertainty_grouping": "prefix_1",
                },
            },
            "gates": {"baseline_compare": {"enabled": False, "baseline_dir": "tmp/baselines", "require_same_host": True, "max_slowdown": 1.3}},
        }

        # Unmapped keys from global blocks too.
        used_globals: set[str] = set()
        for k in ("readfrom", "histopath", "histpath", "exportdir", "combinationxml", "combinationxml", "histfactoryxml", "measurement", "poi", "poi"):
            used_globals.add(k)
        for gb in globals_blocks:
            unmapped.extend(_collect_unmapped(block=gb, used_keys=used_globals))

        report: dict[str, Any] = {
            "version": 1,
            "source_path": str(source_path) if source_path is not None else None,
            "out_path": str(out_path) if out_path is not None else None,
            "mapped": [
                {
                    "target": m.target,
                    "source": {
                        "kind": m.source.kind,
                        "name": m.source.name,
                        "key": m.source.key,
                        "line": m.source.line,
                        "value": m.source.value,
                    },
                }
                for m in mapped
            ],
            "unmapped": [
                {"kind": u.kind, "name": u.name, "key": u.key, "line": u.line, "value": u.value}
                for u in unmapped
            ],
            "notes": notes,
        }
        return spec, report

    tree_e = take_global(["TreeName", "Tree"])
    tree_name = tree_e.value.raw.strip() if tree_e and tree_e.value.raw.strip() else "events"
    if tree_e is not None:
        mapped.append(
            _Mapping(
                target="inputs.trex_config_yaml.tree_name",
                source=_Source("Global", None, tree_e.key, tree_e.line, tree_e.value.raw),
            )
        )

    meas_e = take_global(["Measurement"])
    measurement = meas_e.value.raw.strip() if meas_e and meas_e.value.raw.strip() else "meas"
    if meas_e is not None:
        mapped.append(
            _Mapping(
                target="inputs.trex_config_yaml.measurement",
                source=_Source("Global", None, meas_e.key, meas_e.line, meas_e.value.raw),
            )
        )

    poi_e = take_global(["POI", "Poi"])
    poi = poi_e.value.raw.strip() if poi_e and poi_e.value.raw.strip() else "mu"
    if poi_e is not None:
        mapped.append(
            _Mapping(
                target="inputs.trex_config_yaml.poi",
                source=_Source("Global", None, poi_e.key, poi_e.line, poi_e.value.raw),
            )
        )

    # base_dir: relative path from out_path directory to source config directory if possible.
    base_dir_value: str | None = None
    if source_path is not None:
        cfg_dir = source_path.parent
        if out_path is None:
            base_dir_value = "."
        else:
            out_dir = out_path.parent
            try:
                rel = os.path.relpath(str(cfg_dir.resolve()), str(out_dir.resolve()))
                base_dir_value = "." if rel == "." else rel
            except Exception:
                base_dir_value = str(cfg_dir)

    # Regions
    regions: list[dict[str, Any]] = []
    for b in region_blocks:
        used: set[str] = set()
        name = b.name
        if not name:
            notes.append(f"dropped Region block at line {b.start_line}: missing name")
            unmapped.extend(_collect_unmapped(block=b, used_keys=used))
            continue

        var_e = _last_any(b, ["Variable", "Var"])
        if var_e is None or not var_e.value.raw.strip():
            notes.append(f"dropped Region={name!r}: missing Variable")
            unmapped.extend(_collect_unmapped(block=b, used_keys=used))
            continue
        used.add(_lower(var_e.key))

        bin_e = _last_any(b, ["Binning", "BinEdges"])
        if bin_e is None or not bin_e.value.raw.strip():
            notes.append(f"dropped Region={name!r}: missing Binning")
            unmapped.extend(_collect_unmapped(block=b, used_keys=used))
            continue
        used.add(_lower(bin_e.key))

        try:
            edges = _as_binning_edges(bin_e.value.raw)
        except TrexConfigImportError as e:
            notes.append(f"dropped Region={name!r}: {e}")
            unmapped.extend(_collect_unmapped(block=b, used_keys=used))
            continue

        out: dict[str, Any] = {
            "name": name,
            "variable": var_e.value.raw.strip(),
            "binning_edges": edges,
            "selection": None,
        }
        mapped.append(_Mapping("inputs.trex_config_yaml.regions[].name", _Source(b.kind, name, "Region", b.start_line, name)))
        mapped.append(_Mapping("inputs.trex_config_yaml.regions[].variable", _Source(b.kind, name, var_e.key, var_e.line, var_e.value.raw)))
        mapped.append(_Mapping("inputs.trex_config_yaml.regions[].binning_edges", _Source(b.kind, name, bin_e.key, bin_e.line, bin_e.value.raw)))

        sel_e = _last_any(b, ["Selection", "Cut"])
        if sel_e is not None and sel_e.value.raw.strip():
            used.add(_lower(sel_e.key))
            out["selection"] = sel_e.value.raw.strip()
            mapped.append(
                _Mapping(
                    "inputs.trex_config_yaml.regions[].selection",
                    _Source(b.kind, name, sel_e.key, sel_e.line, sel_e.value.raw),
                )
            )

        df_e = _last_any(b, ["DataFile"])
        if df_e is not None and df_e.value.raw.strip():
            used.add(_lower(df_e.key))
            out["data_file"] = df_e.value.raw.strip()
            mapped.append(
                _Mapping(
                    "inputs.trex_config_yaml.regions[].data_file",
                    _Source(b.kind, name, df_e.key, df_e.line, df_e.value.raw),
                )
            )

        dt_e = _last_any(b, ["DataTreeName", "DataTree"])
        if dt_e is not None and dt_e.value.raw.strip():
            used.add(_lower(dt_e.key))
            out["data_tree_name"] = dt_e.value.raw.strip()
            mapped.append(
                _Mapping(
                    "inputs.trex_config_yaml.regions[].data_tree_name",
                    _Source(b.kind, name, dt_e.key, dt_e.line, dt_e.value.raw),
                )
            )

        unmapped.extend(_collect_unmapped(block=b, used_keys=used))
        regions.append(out)

    # Samples
    samples: list[dict[str, Any]] = []
    for b in sample_blocks:
        used: set[str] = set()
        name = b.name
        if not name:
            notes.append(f"dropped Sample block at line {b.start_line}: missing name")
            unmapped.extend(_collect_unmapped(block=b, used_keys=used))
            continue

        typ_e = _last_any(b, ["Type"])
        kind = "mc"
        if typ_e is not None and typ_e.value.raw.strip().lower() == "data":
            kind = "data"
        if typ_e is not None:
            used.add(_lower(typ_e.key))

        file_e = _last_any(b, ["File", "Path"])
        if file_e is None or not file_e.value.raw.strip():
            notes.append(f"dropped Sample={name!r}: missing File")
            unmapped.extend(_collect_unmapped(block=b, used_keys=used))
            continue
        used.add(_lower(file_e.key))

        out: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "file": file_e.value.raw.strip(),
            "tree_name": None,
            "weight": None,
            "regions": None,
            "norm_factors": [],
            "norm_sys": [],
            "stat_error": False,
        }
        mapped.append(_Mapping("inputs.trex_config_yaml.samples[].name", _Source(b.kind, name, "Sample", b.start_line, name)))
        mapped.append(_Mapping("inputs.trex_config_yaml.samples[].file", _Source(b.kind, name, file_e.key, file_e.line, file_e.value.raw)))

        tr_e = _last_any(b, ["TreeName", "Tree"])
        if tr_e is not None and tr_e.value.raw.strip():
            used.add(_lower(tr_e.key))
            out["tree_name"] = tr_e.value.raw.strip()
            mapped.append(_Mapping("inputs.trex_config_yaml.samples[].tree_name", _Source(b.kind, name, tr_e.key, tr_e.line, tr_e.value.raw)))

        w_e = _last_any(b, ["Weight"])
        if w_e is not None and w_e.value.raw.strip():
            used.add(_lower(w_e.key))
            out["weight"] = w_e.value.raw.strip()
            mapped.append(_Mapping("inputs.trex_config_yaml.samples[].weight", _Source(b.kind, name, w_e.key, w_e.line, w_e.value.raw)))

        regs_e = _last_any(b, ["Regions"])
        if regs_e is not None and regs_e.value.raw.strip():
            used.add(_lower(regs_e.key))
            out["regions"] = _split_list(regs_e.value.raw)
            mapped.append(_Mapping("inputs.trex_config_yaml.samples[].regions", _Source(b.kind, name, regs_e.key, regs_e.line, regs_e.value.raw)))

        for nf_e in _all_any(b, ["NormFactor"]):
            used.add(_lower(nf_e.key))
            for tok in _split_list(nf_e.value.raw):
                if tok and tok not in out["norm_factors"]:
                    out["norm_factors"].append(tok)
            mapped.append(_Mapping("inputs.trex_config_yaml.samples[].norm_factors[]", _Source(b.kind, name, nf_e.key, nf_e.line, nf_e.value.raw)))

        for ns_e in _all_any(b, ["NormSys"]):
            used.add(_lower(ns_e.key))
            toks = [t for t in ns_e.value.raw.replace(",", " ").split() if t.strip()]
            if len(toks) < 3:
                notes.append(f"ignored Sample={name!r} NormSys={ns_e.value.raw!r}: expected 'name lo hi'")
                continue
            try:
                out["norm_sys"].append({"name": toks[0], "lo": _as_f64(toks[1]), "hi": _as_f64(toks[2])})
                mapped.append(_Mapping("inputs.trex_config_yaml.samples[].norm_sys[]", _Source(b.kind, name, ns_e.key, ns_e.line, ns_e.value.raw)))
            except TrexConfigImportError as e:
                notes.append(f"ignored Sample={name!r} NormSys={ns_e.value.raw!r}: {e}")

        for se_e in _all_any(b, ["StatError"]):
            used.add(_lower(se_e.key))
            se = _as_bool(se_e.value.raw)
            if se is True:
                out["stat_error"] = True
                mapped.append(_Mapping("inputs.trex_config_yaml.samples[].stat_error", _Source(b.kind, name, se_e.key, se_e.line, se_e.value.raw)))

        unmapped.extend(_collect_unmapped(block=b, used_keys=used))
        samples.append(out)

    # Systematics
    systematics: list[dict[str, Any]] = []
    for b in syst_blocks:
        used: set[str] = set()
        name = b.name
        if not name:
            notes.append(f"dropped Systematic block at line {b.start_line}: missing name")
            unmapped.extend(_collect_unmapped(block=b, used_keys=used))
            continue

        typ_e = _last_any(b, ["Type"])
        if typ_e is None or not typ_e.value.raw.strip():
            notes.append(f"dropped Systematic={name!r}: missing Type")
            unmapped.extend(_collect_unmapped(block=b, used_keys=used))
            continue
        used.add(_lower(typ_e.key))
        typ = typ_e.value.raw.strip().lower()
        if typ in ("normsys",):
            typ = "norm"
        if typ in ("weightsys",):
            typ = "weight"
        if typ in ("treesys",):
            typ = "tree"
        if typ not in ("norm", "weight", "tree"):
            notes.append(f"dropped Systematic={name!r}: unknown Type={typ_e.value.raw!r}")
            unmapped.extend(_collect_unmapped(block=b, used_keys=used))
            continue

        samp_e = _last_any(b, ["Samples"])
        if samp_e is None or not samp_e.value.raw.strip():
            notes.append(f"dropped Systematic={name!r}: missing Samples")
            unmapped.extend(_collect_unmapped(block=b, used_keys=used))
            continue
        used.add(_lower(samp_e.key))
        samp_list = _split_list(samp_e.value.raw)
        if not samp_list:
            notes.append(f"dropped Systematic={name!r}: Samples list is empty")
            unmapped.extend(_collect_unmapped(block=b, used_keys=used))
            continue

        out: dict[str, Any] = {"name": name, "type": typ, "samples": samp_list, "regions": None}
        mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].name", _Source(b.kind, name, "Systematic", b.start_line, name)))
        mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].type", _Source(b.kind, name, typ_e.key, typ_e.line, typ_e.value.raw)))
        mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].samples", _Source(b.kind, name, samp_e.key, samp_e.line, samp_e.value.raw)))

        regs_e = _last_any(b, ["Regions"])
        if regs_e is not None and regs_e.value.raw.strip():
            used.add(_lower(regs_e.key))
            out["regions"] = _split_list(regs_e.value.raw)
            mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].regions", _Source(b.kind, name, regs_e.key, regs_e.line, regs_e.value.raw)))

        if typ == "norm":
            lo_e = _last_any(b, ["Lo"])
            hi_e = _last_any(b, ["Hi"])
            if lo_e is None or hi_e is None:
                notes.append(f"dropped Systematic={name!r}: norm requires Lo/Hi")
                unmapped.extend(_collect_unmapped(block=b, used_keys=used))
                continue
            used.add(_lower(lo_e.key))
            used.add(_lower(hi_e.key))
            out["lo"] = _as_f64(lo_e.value.raw)
            out["hi"] = _as_f64(hi_e.value.raw)
            mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].lo", _Source(b.kind, name, lo_e.key, lo_e.line, lo_e.value.raw)))
            mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].hi", _Source(b.kind, name, hi_e.key, hi_e.line, hi_e.value.raw)))

        elif typ == "weight":
            up_e = _last_any(b, ["WeightUp", "Up"])
            down_e = _last_any(b, ["WeightDown", "Down"])

            base_e = _last_any(b, ["WeightBase", "Weight"])
            up_suf_e = _last_any(b, ["WeightUpSuffix", "UpSuffix", "SuffixUp"])
            down_suf_e = _last_any(b, ["WeightDownSuffix", "DownSuffix", "SuffixDown"])

            if up_e is not None or down_e is not None:
                if up_e is None or down_e is None:
                    notes.append(f"dropped Systematic={name!r}: weight requires WeightUp and WeightDown")
                    unmapped.extend(_collect_unmapped(block=b, used_keys=used))
                    continue
                used.add(_lower(up_e.key))
                used.add(_lower(down_e.key))
                out["weight_up"] = up_e.value.raw.strip()
                out["weight_down"] = down_e.value.raw.strip()
                mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].weight_up", _Source(b.kind, name, up_e.key, up_e.line, up_e.value.raw)))
                mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].weight_down", _Source(b.kind, name, down_e.key, down_e.line, down_e.value.raw)))
            elif base_e is not None or up_suf_e is not None or down_suf_e is not None:
                if base_e is None or up_suf_e is None or down_suf_e is None:
                    notes.append(
                        f"dropped Systematic={name!r}: weight suffix expansion requires WeightBase/WeightUpSuffix/WeightDownSuffix"
                    )
                    unmapped.extend(_collect_unmapped(block=b, used_keys=used))
                    continue
                used.add(_lower(base_e.key))
                used.add(_lower(up_suf_e.key))
                used.add(_lower(down_suf_e.key))
                base = base_e.value.raw.strip()
                out["weight_up"] = f"{base}{up_suf_e.value.raw.strip()}"
                out["weight_down"] = f"{base}{down_suf_e.value.raw.strip()}"
                mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].weight_up", _Source(b.kind, name, base_e.key, base_e.line, base_e.value.raw)))
                mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].weight_down", _Source(b.kind, name, base_e.key, base_e.line, base_e.value.raw)))
            else:
                notes.append(f"dropped Systematic={name!r}: weight requires WeightUp/WeightDown")
                unmapped.extend(_collect_unmapped(block=b, used_keys=used))
                continue

        elif typ == "tree":
            up_e = _last_any(b, ["FileUp", "UpFile", "Up"])
            down_e = _last_any(b, ["FileDown", "DownFile", "Down"])
            if up_e is None or down_e is None:
                notes.append(f"dropped Systematic={name!r}: tree requires FileUp/FileDown")
                unmapped.extend(_collect_unmapped(block=b, used_keys=used))
                continue
            used.add(_lower(up_e.key))
            used.add(_lower(down_e.key))
            out["file_up"] = up_e.value.raw.strip()
            out["file_down"] = down_e.value.raw.strip()
            mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].file_up", _Source(b.kind, name, up_e.key, up_e.line, up_e.value.raw)))
            mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].file_down", _Source(b.kind, name, down_e.key, down_e.line, down_e.value.raw)))

            tn_e = _last_any(b, ["TreeName", "Tree"])
            if tn_e is not None and tn_e.value.raw.strip():
                used.add(_lower(tn_e.key))
                out["tree_name"] = tn_e.value.raw.strip()
                mapped.append(_Mapping("inputs.trex_config_yaml.systematics[].tree_name", _Source(b.kind, name, tn_e.key, tn_e.line, tn_e.value.raw)))

        unmapped.extend(_collect_unmapped(block=b, used_keys=used))
        systematics.append(out)

    if not regions:
        raise TrexConfigImportError("no usable Region blocks found")
    if not samples:
        raise TrexConfigImportError("no usable Sample blocks found")

    analysis_name = (
        (source_path.stem if source_path is not None else None) or "Generated TREx Analysis"
    )
    analysis_desc = (
        f"Generated from TRExFitter config: {source_path}" if source_path is not None else "Generated from TRExFitter config"
    )

    spec: dict[str, Any] = {
        "$schema": "https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json",
        "schema_version": "trex_analysis_spec_v0",
        "analysis": {
            "name": analysis_name,
            "description": analysis_desc,
            "tags": ["generated", "trex", "import"],
        },
        "inputs": {
            "mode": "trex_config_yaml",
            "trex_config_yaml": {
                "read_from": "NTUP",
                "base_dir": base_dir_value,
                "tree_name": tree_name,
                "measurement": measurement,
                "poi": poi,
                "regions": regions,
                "samples": samples,
                "systematics": systematics,
            },
        },
        "execution": {
            "determinism": {"threads": int(threads)},
            "import": {"enabled": True, "output_json": workspace_out},
            "fit": {"enabled": False, "output_json": "tmp/trex_fit.json"},
            "profile_scan": {
                "enabled": False,
                "start": 0.0,
                "stop": 5.0,
                "points": 21,
                "output_json": "tmp/trex_scan.json",
            },
            "report": {
                "enabled": False,
                "out_dir": "tmp/trex_report",
                "overwrite": True,
                "include_covariance": False,
                "histfactory_xml": None,
                "render": {"enabled": False, "pdf": None, "svg_dir": None, "python": None},
                "skip_uncertainty": False,
                "uncertainty_grouping": "prefix_1",
            },
        },
        "gates": {
            "baseline_compare": {
                "enabled": False,
                "baseline_dir": "tmp/baselines",
                "require_same_host": True,
                "max_slowdown": 1.3,
            }
        },
    }

    # Unmapped keys from global blocks too.
    used_globals: set[str] = set()
    for k in ("readfrom", "treename", "tree", "measurement", "poi", "poi"):
        used_globals.add(k)
    for gb in globals_blocks:
        unmapped.extend(_collect_unmapped(block=gb, used_keys=used_globals))

    report: dict[str, Any] = {
        "version": 1,
        "source_path": str(source_path) if source_path is not None else None,
        "out_path": str(out_path) if out_path is not None else None,
        "mapped": [
            {
                "target": m.target,
                "source": {
                    "kind": m.source.kind,
                    "name": m.source.name,
                    "key": m.source.key,
                    "line": m.source.line,
                    "value": m.source.value,
                },
            }
            for m in mapped
        ],
        "unmapped": [
            {"kind": u.kind, "name": u.name, "key": u.key, "line": u.line, "value": u.value}
            for u in unmapped
        ],
        "notes": notes,
    }
    return spec, report


def trex_config_file_to_analysis_spec_v0(
    config_path: str | Path,
    *,
    out_path: str | Path | None = None,
    threads: int = 1,
    workspace_out: str = "tmp/trex_workspace.json",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convenience wrapper: parse file -> convert -> (spec, report)."""
    from .parser import parse_trex_config_file

    cfg_path = Path(config_path)
    doc = parse_trex_config_file(cfg_path)
    outp = None if out_path is None else Path(out_path)
    return trex_doc_to_analysis_spec_v0(
        doc,
        source_path=cfg_path,
        out_path=outp,
        threads=threads,
        workspace_out=workspace_out,
    )


def dump_yaml(obj: Any) -> str:
    """Minimal YAML emitter (no external deps).

    Output is deterministic and double-quotes all strings to avoid YAML gotchas.
    """

    def scalar(x: Any) -> str:
        if x is None:
            return "null"
        if isinstance(x, bool):
            return "true" if x else "false"
        if isinstance(x, int):
            return str(x)
        if isinstance(x, float):
            # Use repr for round-trip stability.
            return repr(x)
        if isinstance(x, str):
            s = x.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            return f'"{s}"'
        raise TypeError(f"unsupported scalar type: {type(x)}")

    def emit_lines(x: Any, indent: int) -> list[str]:
        sp = "  " * indent
        if isinstance(x, dict):
            if not x:
                return [f"{sp}{{}}"]
            out: list[str] = []
            for k, v in x.items():
                if not isinstance(k, str):
                    raise TypeError("YAML emitter expects string keys")
                if isinstance(v, (dict, list)) and v:
                    out.append(f"{sp}{k}:")
                    out.extend(emit_lines(v, indent + 1))
                else:
                    if isinstance(v, (dict, list)) and not v:
                        # Empty containers in flow form.
                        out.append(f"{sp}{k}: {scalar(v) if v is None else ('[]' if isinstance(v, list) else '{}')}")
                    else:
                        out.append(f"{sp}{k}: {scalar(v)}")
            return out

        if isinstance(x, list):
            if not x:
                return [f"{sp}[]"]
            out = []
            for item in x:
                if isinstance(item, (dict, list)):
                    out.append(f"{sp}-")
                    out.extend(emit_lines(item, indent + 1))
                else:
                    out.append(f"{sp}- {scalar(item)}")
            return out

        return [f"{sp}{scalar(x)}"]

    return "\n".join(emit_lines(obj, 0)).rstrip() + "\n"
