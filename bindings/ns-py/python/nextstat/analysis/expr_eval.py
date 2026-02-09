"""Vectorized expression evaluator for TREx-style selections/weights.

This is a small, deterministic evaluator intended for:
- correctness contracts / baselines on small synthetic data
- future integration with columnar (NumPy/Awkward) event data

The grammar intentionally matches the Rust `ns-root` expression engine subset:
- arithmetic: + - * /
- comparisons: == != < <= > >=
- boolean: && || ! (truthiness: non-zero is true, including negatives and NaN)
- functions: abs/fabs, sqrt, log/log10, exp, sin/cos, pow/power, atan2, min, max
- ternary: cond ? a : b (right-associative)
- indexing: x[0] for per-event arrays (list-of-arrays). Indexing is evaluated lazily
  through ternary masks so missing indices can be avoided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


class ExprEvalError(ValueError):
    pass


@dataclass(frozen=True)
class Span:
    start: int
    end: int


def _line_col_1based(src: str, offset: int) -> tuple[int, int]:
    offset = max(0, min(len(src), offset))
    line = 1
    col = 1
    for ch in src[:offset]:
        if ch == "\n":
            line += 1
            col = 1
        else:
            col += 1
    return line, col


def _err(src: str, span: Span, msg: str) -> ExprEvalError:
    line, col = _line_col_1based(src, span.start)
    return ExprEvalError(f"line {line}, col {col}: {msg}")


class _K:
    NUM = "NUM"
    IDENT = "IDENT"
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    LPAREN = "("
    RPAREN = ")"
    COMMA = ","
    QUESTION = "?"
    COLON = ":"
    LBRACKET = "["
    RBRACKET = "]"
    EQ = "=="
    NE = "!="
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    AND = "&&"
    OR = "||"
    NOT = "!"


@dataclass(frozen=True)
class _Tok:
    kind: str
    value: Any
    span: Span


def _tokenize(src: str) -> list[_Tok]:
    if not src.isascii():
        raise ExprEvalError("expression must be ASCII (non-ASCII is not supported yet)")

    b = src.encode("ascii")
    out: list[_Tok] = []
    i = 0

    def emit(kind: str, value: Any, start: int, end: int) -> None:
        out.append(_Tok(kind=kind, value=value, span=Span(start=start, end=end)))

    while i < len(b):
        c = b[i]
        if chr(c).isspace():
            i += 1
            continue

        if i + 1 < len(b):
            two = src[i : i + 2]
            kind = {
                "&&": _K.AND,
                "||": _K.OR,
                "==": _K.EQ,
                "!=": _K.NE,
                "<=": _K.LE,
                ">=": _K.GE,
            }.get(two)
            if kind is not None:
                emit(kind, None, i, i + 2)
                i += 2
                continue

        start = i
        ch = chr(c)
        if ch in "+-*/(),?:[]<>!":
            kind = {
                "+": _K.PLUS,
                "-": _K.MINUS,
                "*": _K.STAR,
                "/": _K.SLASH,
                "(": _K.LPAREN,
                ")": _K.RPAREN,
                ",": _K.COMMA,
                "?": _K.QUESTION,
                ":": _K.COLON,
                "[": _K.LBRACKET,
                "]": _K.RBRACKET,
                "<": _K.LT,
                ">": _K.GT,
                "!": _K.NOT,
            }[ch]
            i += 1
            emit(kind, None, start, i)
            continue

        if ch.isdigit() or ch == ".":
            i += 1
            while i < len(b):
                cc = chr(b[i])
                if cc.isdigit() or cc in ".eE":
                    i += 1
                    continue
                if cc in "+-" and i > start and chr(b[i - 1]) in "eE":
                    i += 1
                    continue
                break
            s = src[start:i]
            try:
                n = float(s)
            except Exception:
                raise _err(src, Span(start, i), f"invalid number: '{s}'")
            emit(_K.NUM, n, start, i)
            continue

        if ch.isalpha() or ch == "_":
            i += 1
            while i < len(b):
                cc = chr(b[i])
                if cc.isalnum() or cc == "_" or cc == ".":
                    i += 1
                    continue
                # Allow C++-style namespace qualifier inside identifiers (e.g. `TMath::Abs`).
                if cc == ":" and i + 1 < len(b) and chr(b[i + 1]) == ":":
                    i += 2
                    continue
                break
            emit(_K.IDENT, src[start:i], start, i)
            continue

        raise _err(src, Span(i, i + 1), f"unexpected character: '{ch}'")

    return out


@dataclass(frozen=True)
class _Node:
    span: Span


@dataclass(frozen=True)
class _Num(_Node):
    value: float


@dataclass(frozen=True)
class _Var(_Node):
    name: str


@dataclass(frozen=True)
class _Unary(_Node):
    op: str
    expr: _Ast


@dataclass(frozen=True)
class _Bin(_Node):
    op: str
    left: _Ast
    right: _Ast


@dataclass(frozen=True)
class _Call(_Node):
    fn: str
    args: list[_Ast]


@dataclass(frozen=True)
class _Ternary(_Node):
    cond: _Ast
    then_expr: _Ast
    else_expr: _Ast


@dataclass(frozen=True)
class _Index(_Node):
    base: _Ast
    index: _Ast


_Ast = _Num | _Var | _Unary | _Bin | _Call | _Ternary | _Index


class _Parser:
    def __init__(self, src: str, toks: Sequence[_Tok]):
        self._src = src
        self._toks = toks
        self._pos = 0

    def _peek(self) -> _Tok | None:
        return self._toks[self._pos] if self._pos < len(self._toks) else None

    def _advance(self) -> _Tok | None:
        t = self._peek()
        if t is not None:
            self._pos += 1
        return t

    def _expect(self, kind: str) -> _Tok:
        t = self._advance()
        if t is None:
            raise _err(self._src, Span(len(self._src), len(self._src)), f"expected {kind}, got end of input")
        if t.kind != kind:
            raise _err(self._src, t.span, f"expected {kind}, got {t.kind}")
        return t

    def parse(self) -> _Ast:
        e = self._parse_ternary()
        if self._pos != len(self._toks):
            t = self._toks[self._pos]
            raise _err(self._src, t.span, f"unexpected token after expression: {t.kind}")
        return e

    def _parse_ternary(self) -> _Ast:
        cond = self._parse_or()
        t = self._peek()
        if t is not None and t.kind == _K.QUESTION:
            q = self._advance()
            assert q is not None
            then_expr = self._parse_ternary()
            self._expect(_K.COLON)
            else_expr = self._parse_ternary()
            span = Span(cond.span.start, else_expr.span.end)
            return _Ternary(span=span, cond=cond, then_expr=then_expr, else_expr=else_expr)
        return cond

    def _parse_or(self) -> _Ast:
        lhs = self._parse_and()
        while True:
            t = self._peek()
            if t is None or t.kind != _K.OR:
                return lhs
            op = self._advance()
            assert op is not None
            rhs = self._parse_and()
            lhs = _Bin(span=Span(lhs.span.start, rhs.span.end), op=_K.OR, left=lhs, right=rhs)

    def _parse_and(self) -> _Ast:
        lhs = self._parse_cmp()
        while True:
            t = self._peek()
            if t is None or t.kind != _K.AND:
                return lhs
            op = self._advance()
            assert op is not None
            rhs = self._parse_cmp()
            lhs = _Bin(span=Span(lhs.span.start, rhs.span.end), op=_K.AND, left=lhs, right=rhs)

    def _parse_cmp(self) -> _Ast:
        lhs = self._parse_add()
        t = self._peek()
        if t is None:
            return lhs
        if t.kind not in (_K.EQ, _K.NE, _K.LT, _K.LE, _K.GT, _K.GE):
            return lhs
        op = self._advance()
        assert op is not None
        rhs = self._parse_add()
        return _Bin(span=Span(lhs.span.start, rhs.span.end), op=op.kind, left=lhs, right=rhs)

    def _parse_add(self) -> _Ast:
        lhs = self._parse_mul()
        while True:
            t = self._peek()
            if t is None or t.kind not in (_K.PLUS, _K.MINUS):
                return lhs
            op = self._advance()
            assert op is not None
            rhs = self._parse_mul()
            lhs = _Bin(span=Span(lhs.span.start, rhs.span.end), op=op.kind, left=lhs, right=rhs)

    def _parse_mul(self) -> _Ast:
        lhs = self._parse_unary()
        while True:
            t = self._peek()
            if t is None or t.kind not in (_K.STAR, _K.SLASH):
                return lhs
            op = self._advance()
            assert op is not None
            rhs = self._parse_unary()
            lhs = _Bin(span=Span(lhs.span.start, rhs.span.end), op=op.kind, left=lhs, right=rhs)

    def _parse_unary(self) -> _Ast:
        t = self._peek()
        if t is not None and t.kind in (_K.MINUS, _K.NOT):
            op = self._advance()
            assert op is not None
            e = self._parse_unary()
            return _Unary(span=Span(op.span.start, e.span.end), op=op.kind, expr=e)
        return self._parse_postfix()

    def _parse_postfix(self) -> _Ast:
        e = self._parse_atom()
        while True:
            t = self._peek()
            if t is None or t.kind != _K.LBRACKET:
                return e
            lb = self._advance()
            assert lb is not None
            idx_expr = self._parse_ternary()
            rb = self._expect(_K.RBRACKET)
            e = _Index(span=Span(lb.span.start, rb.span.end), base=e, index=idx_expr)

    def _parse_atom(self) -> _Ast:
        t = self._advance()
        if t is None:
            raise _err(self._src, Span(len(self._src), len(self._src)), "expected expression, got end of input")

        if t.kind == _K.NUM:
            return _Num(span=t.span, value=float(t.value))

        if t.kind == _K.LPAREN:
            e = self._parse_ternary()
            self._expect(_K.RPAREN)
            return e

        if t.kind == _K.IDENT:
            name = str(t.value)
            if self._peek() is not None and self._peek().kind == _K.LPAREN:
                self._advance()  # '('
                args: list[_Ast] = []
                if self._peek() is not None and self._peek().kind != _K.RPAREN:
                    args.append(self._parse_ternary())
                    while self._peek() is not None and self._peek().kind == _K.COMMA:
                        self._advance()
                        args.append(self._parse_ternary())
                rp = self._expect(_K.RPAREN)
                return _Call(span=Span(t.span.start, rp.span.end), fn=name, args=args)
            return _Var(span=t.span, name=name)

        raise _err(self._src, t.span, f"expected number, identifier, or '(', got {t.kind}")


def _infer_n(env: Mapping[str, Any]) -> int:
    for v in env.values():
        if isinstance(v, np.ndarray) and v.ndim == 1:
            return int(v.shape[0])
        if isinstance(v, list):
            return len(v)
    return 1


def _as_f64(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("expected 1D array")
    return arr


def eval_expr(expr: str, env: Mapping[str, Any], *, n: int | None = None) -> np.ndarray:
    """Evaluate `expr` over `env` and return a float64 vector.

    `env` values can be:
    - 1D numpy arrays (scalars per event)
    - python scalars (broadcast)
    - per-event arrays: list[Sequence[float]] (only valid through indexing, e.g. jets_pt[0])
    """
    toks = _tokenize(expr)
    ast = _Parser(expr, toks).parse()
    n_total = int(_infer_n(env) if n is None else n)
    idxs = np.arange(n_total, dtype=np.int64)
    return _eval(ast, expr, env, idxs, strict_indexing=False)


def eval_expr_strict(expr: str, env: Mapping[str, Any], *, n: int | None = None) -> np.ndarray:
    """Like `eval_expr`, but out-of-range indexing is an error (better diagnostics in tests)."""
    toks = _tokenize(expr)
    ast = _Parser(expr, toks).parse()
    n_total = int(_infer_n(env) if n is None else n)
    idxs = np.arange(n_total, dtype=np.int64)
    return _eval(ast, expr, env, idxs, strict_indexing=True)


def _truthy(x: np.ndarray) -> np.ndarray:
    # ROOT/TTreeFormula truthiness: any non-zero is true (including negatives and NaN).
    return x != 0.0


def _eval(
    ast: _Ast,
    src: str,
    env: Mapping[str, Any],
    idxs: np.ndarray,
    *,
    strict_indexing: bool,
) -> np.ndarray:
    n = int(idxs.shape[0])

    if isinstance(ast, _Num):
        return np.full(n, ast.value, dtype=np.float64)

    if isinstance(ast, _Var):
        v = env.get(ast.name)
        if v is None:
            raise _err(src, ast.span, f"unknown variable: '{ast.name}'")
        if isinstance(v, (int, float, np.floating)):
            return np.full(n, float(v), dtype=np.float64)
        if isinstance(v, np.ndarray):
            a = _as_f64(v)
            return a[idxs]
        if isinstance(v, list):
            raise _err(src, ast.span, f"array-valued variable '{ast.name}' requires indexing (e.g. {ast.name}[0])")
        raise _err(src, ast.span, f"unsupported variable type for '{ast.name}': {type(v).__name__}")

    if isinstance(ast, _Index):
        if not isinstance(ast.base, _Var):
            raise _err(src, ast.span, "indexing is only supported on variables in v1")
        name = ast.base.name
        v = env.get(name)
        if v is None:
            raise _err(src, ast.span, f"unknown variable: '{name}'")
        if not isinstance(v, list):
            raise _err(src, ast.span, f"variable '{name}' is not an array; cannot index with [...]")

        idx_vals = _eval(ast.index, src, env, idxs, strict_indexing=strict_indexing)
        out = np.empty(n, dtype=np.float64)
        for j, event_idx in enumerate(idxs.tolist()):
            row = v[event_idx]
            idx_f = float(idx_vals[j])
            if not np.isfinite(idx_f):
                idx_i = -1
            else:
                # Match ROOT-like behavior: truncate float to int.
                idx_i = int(np.trunc(idx_f))

            if idx_i < 0 or idx_i >= len(row):
                if strict_indexing:
                    raise _err(
                        src,
                        ast.span,
                        f"index out of bounds: {name}[{idx_i}] for event {event_idx}",
                    )
                out[j] = 0.0
                continue
            out[j] = float(row[idx_i])
        return out

    if isinstance(ast, _Unary):
        a = _eval(ast.expr, src, env, idxs, strict_indexing=strict_indexing)
        if ast.op == _K.MINUS:
            return -a
        if ast.op == _K.NOT:
            return np.where(_truthy(a), 0.0, 1.0).astype(np.float64)
        raise _err(src, ast.span, f"unknown unary op: {ast.op}")

    if isinstance(ast, _Bin):
        lhs = _eval(ast.left, src, env, idxs, strict_indexing=strict_indexing)
        rhs = _eval(ast.right, src, env, idxs, strict_indexing=strict_indexing)
        op = ast.op
        if op == _K.PLUS:
            return lhs + rhs
        if op == _K.MINUS:
            return lhs - rhs
        if op == _K.STAR:
            return lhs * rhs
        if op == _K.SLASH:
            return lhs / rhs
        if op == _K.EQ:
            eps = np.finfo(np.float64).eps
            return np.where(np.abs(lhs - rhs) < eps, 1.0, 0.0)
        if op == _K.NE:
            eps = np.finfo(np.float64).eps
            return np.where(np.abs(lhs - rhs) >= eps, 1.0, 0.0)
        if op == _K.LT:
            return np.where(lhs < rhs, 1.0, 0.0)
        if op == _K.LE:
            return np.where(lhs <= rhs, 1.0, 0.0)
        if op == _K.GT:
            return np.where(lhs > rhs, 1.0, 0.0)
        if op == _K.GE:
            return np.where(lhs >= rhs, 1.0, 0.0)
        if op == _K.AND:
            return np.where(_truthy(lhs) & _truthy(rhs), 1.0, 0.0)
        if op == _K.OR:
            return np.where(_truthy(lhs) | _truthy(rhs), 1.0, 0.0)
        raise _err(src, ast.span, f"unknown binary op: {op}")

    if isinstance(ast, _Ternary):
        # Lazy per-mask evaluation to match row-wise semantics.
        cond = _eval(ast.cond, src, env, idxs, strict_indexing=strict_indexing)
        mask = _truthy(cond)
        out = np.empty(n, dtype=np.float64)
        if bool(mask.any()):
            out[mask] = _eval(ast.then_expr, src, env, idxs[mask], strict_indexing=strict_indexing)
        if bool((~mask).any()):
            out[~mask] = _eval(ast.else_expr, src, env, idxs[~mask], strict_indexing=strict_indexing)
        return out

    if isinstance(ast, _Call):
        fn_raw = ast.fn
        leaf = fn_raw.rsplit("::", 1)[-1].strip().lower()
        args = [_eval(a, src, env, idxs, strict_indexing=strict_indexing) for a in ast.args]

        if leaf in ("abs", "fabs") and len(args) == 1:
            return np.abs(args[0])
        if leaf == "sqrt" and len(args) == 1:
            return np.sqrt(args[0])
        if leaf == "log" and len(args) == 1:
            return np.log(args[0])
        if leaf == "log10" and len(args) == 1:
            return np.log10(args[0])
        if leaf == "exp" and len(args) == 1:
            return np.exp(args[0])
        if leaf == "sin" and len(args) == 1:
            return np.sin(args[0])
        if leaf == "cos" and len(args) == 1:
            return np.cos(args[0])
        if leaf in ("pow", "power") and len(args) == 2:
            return np.power(args[0], args[1])
        if leaf == "atan2" and len(args) == 2:
            return np.arctan2(args[0], args[1])
        if leaf == "min" and len(args) == 2:
            return np.minimum(args[0], args[1])
        if leaf == "max" and len(args) == 2:
            return np.maximum(args[0], args[1])

        raise _err(src, ast.span, f"unknown function or invalid arity: {fn_raw}({len(args)})")

    raise _err(src, ast.span, "internal error: unknown AST node")
