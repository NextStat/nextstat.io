from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


class TrexConfigParseError(ValueError):
    def __init__(self, message: str, *, path: str | None, line: int, context: str, col: int | None = None):
        loc = f"{path or '<string>'}:{line}"
        caret = ""
        if col is not None and col >= 0:
            caret = f"\n  {' ' * col}^"
        super().__init__(f"{loc}: {message}\n  {context.rstrip()}{caret}")
        self.path = path
        self.line = line
        self.context = context
        self.col = col


@dataclass(frozen=True)
class TrexValue:
    raw: str
    items: list[str]


@dataclass(frozen=True)
class TrexConfigEntry:
    key: str
    value: TrexValue
    line: int  # 1-based


@dataclass
class TrexConfigBlock:
    kind: str
    name: str | None
    start_line: int  # 1-based
    _attrs: dict[str, list[TrexConfigEntry]]

    def entries(self, key: str) -> list[TrexConfigEntry]:
        return list(self._attrs.get(key.lower(), []))

    def last(self, key: str) -> TrexConfigEntry | None:
        items = self._attrs.get(key.lower())
        return items[-1] if items else None


@dataclass
class TrexConfigDoc:
    blocks: list[TrexConfigBlock]

    def find_blocks(self, kind: str) -> list[TrexConfigBlock]:
        k = kind.lower()
        return [b for b in self.blocks if b.kind.lower() == k]


_BLOCK_KEYS = {
    "job",
    "fit",
    "region",
    "sample",
    "systematic",
}


def _strip_comment(s: str, *, path: str | None, line: int) -> str:
    in_quote: str | None = None
    esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_quote is not None:
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            in_quote = ch
            continue
        if ch == "#":
            return s[:i]
        if ch == "/" and i + 1 < len(s) and s[i + 1] == "/":
            return s[:i]

    if in_quote is not None:
        raise TrexConfigParseError("unclosed quote", path=path, line=line, context=s)
    return s


def _split_kv(s: str, *, path: str | None, line: int) -> tuple[str, str]:
    in_quote: str | None = None
    esc = False
    for i, ch in enumerate(s):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if in_quote is not None:
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ("'", '"'):
            in_quote = ch
            continue
        if ch == ":":
            key = s[:i].strip()
            val = s[i + 1 :].strip()
            if not key:
                raise TrexConfigParseError("empty key before ':'", path=path, line=line, context=s, col=i)
            return key, val
    raise TrexConfigParseError("expected 'Key: Value' line", path=path, line=line, context=s)


def _unescape(s: str) -> str:
    # Conservative escapes: keep unknown sequences as-is.
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


def _parse_atom(s: str, *, path: str | None, line: int) -> str:
    s = s.strip()
    if not s:
        return ""
    if s[0] in ('"', "'"):
        q = s[0]
        if len(s) < 2 or s[-1] != q:
            raise TrexConfigParseError("unterminated quoted string", path=path, line=line, context=s)
        return _unescape(s[1:-1])
    return s


def _split_items(s: str, *, path: str | None, line: int) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    in_quote: str | None = None
    esc = False

    def flush() -> None:
        tok = "".join(buf).strip()
        buf.clear()
        if tok:
            items.append(_parse_atom(tok, path=path, line=line))

    for ch in s:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if ch == "\\":
            buf.append(ch)
            esc = True
            continue
        if in_quote is not None:
            buf.append(ch)
            if ch == in_quote:
                in_quote = None
            continue
        if ch in ('"', "'"):
            buf.append(ch)
            in_quote = ch
            continue
        if ch in (";", ","):
            flush()
            continue
        buf.append(ch)

    if in_quote is not None:
        raise TrexConfigParseError("unclosed quote", path=path, line=line, context=s)
    flush()
    return items


def _value_from_raw(raw: str, *, path: str | None, line: int) -> TrexValue:
    raw = raw.strip()
    items = _split_items(raw, path=path, line=line) if raw else []
    if raw and not items:
        # Non-empty raw should produce at least one item; if it doesn't, fall back to atom.
        items = [_parse_atom(raw, path=path, line=line)]
    if len(items) == 1:
        return TrexValue(raw=items[0], items=items)
    return TrexValue(raw=raw, items=items)


def parse_trex_config(text: str, *, path: str | None = None) -> TrexConfigDoc:
    blocks: list[TrexConfigBlock] = []
    global_block = TrexConfigBlock(kind="Global", name=None, start_line=1, _attrs={})
    current: TrexConfigBlock = global_block

    for idx, orig in enumerate(text.splitlines(), start=1):
        stripped = orig.strip()
        if not stripped:
            continue

        line_wo_comment = _strip_comment(orig, path=path, line=idx).strip()
        if not line_wo_comment:
            continue

        key, val = _split_kv(line_wo_comment, path=path, line=idx)
        key_norm = key.strip()
        val_norm = val.strip()

        if key_norm.lower() in _BLOCK_KEYS:
            # Start a new block.
            name = _parse_atom(val_norm, path=path, line=idx) if val_norm else None
            if current is not global_block:
                blocks.append(current)
            current = TrexConfigBlock(kind=key_norm, name=name, start_line=idx, _attrs={})
            continue

        entry = TrexConfigEntry(key=key_norm, value=_value_from_raw(val_norm, path=path, line=idx), line=idx)
        current._attrs.setdefault(key_norm.lower(), []).append(entry)

    # Flush.
    blocks_out = [global_block]
    if current is not global_block:
        blocks.append(current)
    blocks_out.extend(blocks)
    return TrexConfigDoc(blocks=blocks_out)


def parse_trex_config_file(path: str | Path) -> TrexConfigDoc:
    p = Path(path)
    return parse_trex_config(p.read_text(), path=str(p))


def iter_attrs(block: TrexConfigBlock) -> Iterable[TrexConfigEntry]:
    for items in block._attrs.values():
        for e in items:
            yield e

