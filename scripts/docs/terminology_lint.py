#!/usr/bin/env python3
"""
Terminology lint for public-facing docs.

Goal: keep "general audience" docs free of unexplained HEP jargon/acronyms.

Scope (by default):
- docs/README.md
- docs/personas/*.md
- docs/quickstarts/**/*.md

Rule (heuristic):
- If a term appears in a file, it must be "explained" in that file.
- Explanation is either an inline expansion like `TERM (expanded ...)` or the
  presence of the expansion phrase somewhere in the prose.

Per-file escape hatch:
  <!-- terminology-lint: disable -->
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class TermRule:
    term: str
    # If any of these regexes match, the term is considered "explained".
    explain_regexes: Tuple[re.Pattern[str], ...]
    suggestion: str


def _compile_rules() -> List[TermRule]:
    def ci(p: str) -> re.Pattern[str]:
        return re.compile(p, re.IGNORECASE)

    def cs(p: str) -> re.Pattern[str]:
        return re.compile(p)

    # Terms we want to avoid in general docs unless explained.
    # Keep the list small and focused; this is a guardrail, not a full glossary.
    base = [
        (
            "NLL",
            (cs(r"\bNLL\s*\("), ci(r"\bnegative log-?likelihood\b")),
            "Write `NLL (negative log-likelihood)` on first use.",
        ),
        (
            "POI",
            (cs(r"\bPOI\s*\("), ci(r"\bparameter of interest\b")),
            "Write `POI (parameter of interest)` on first use.",
        ),
        (
            "NP",
            (cs(r"\bNPs?\s*\("), ci(r"\bnuisance parameter(s)?\b")),
            "Write `NP (nuisance parameter)` (or avoid the acronym).",
        ),
        (
            "CLs",
            (cs(r"\bCLs\s*\("), ci(r"\bconfidence level(s)?\b"), ci(r"\bmodified frequentist\b")),
            "Write `CLs (modified frequentist confidence level)` on first use.",
        ),
        (
            "Asimov",
            (ci(r"\bAsimov\s+dataset\b"), ci(r"\bexpected dataset\b"), ci(r"\bAsimov\s+data\b")),
            "Prefer `expected dataset` or write `Asimov dataset (expected dataset)` on first use.",
        ),
        (
            "HistFactory",
            (cs(r"\bHistFactory\s*\("), ci(r"\bbinned likelihood\b"), ci(r"\bworkspace\b")),
            "Add a short gloss: `HistFactory (a binned-likelihood workspace format)`.",
        ),
        (
            "pyhf",
            (cs(r"\bpyhf\s*\("), ci(r"\bpython\b"), ci(r"\bhistfactory\b")),
            "Add a short gloss: `pyhf (Python reference implementation for HistFactory)`.",
        ),
        (
            "RooFit",
            (cs(r"\bRooFit\s*\("), ci(r"\bROOT\b")),
            "If mentioned, briefly identify it: `RooFit (ROOT's statistical modeling toolkit)`.",
        ),
        (
            "TREx",
            (cs(r"\bTREx(Fitter)?\s*\("), ci(r"\bTRExFitter\b")),
            "Avoid in general docs; if needed, write `TRExFitter (a HEP analysis framework)`.",
        ),
    ]

    rules: List[TermRule] = []
    for term, explainers, suggestion in base:
        rules.append(TermRule(term=term, explain_regexes=tuple(explainers), suggestion=suggestion))
    return rules


def _strip_markdown_code(text: str) -> str:
    # Remove fenced code blocks and inline code spans to avoid false positives
    # from code samples and JSON keys.
    out_lines: List[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        out_lines.append(line)
    s = "\n".join(out_lines)
    # Inline code `...`
    s = re.sub(r"`[^`]*`", "", s)
    return s


def _iter_md_files(paths: Iterable[Path]) -> Iterable[Path]:
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".md":
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.md"))


def _find_issues(path: Path, rules: List[TermRule]) -> List[str]:
    raw = path.read_text(encoding="utf-8")
    if "terminology-lint: disable" in raw:
        return []

    text = _strip_markdown_code(raw)

    issues: List[str] = []
    for rule in rules:
        # Term presence check (case-sensitive for acronyms; allow "Asimov" capitalized).
        if rule.term in ("pyhf",):
            present = re.search(r"\bpyhf\b", text) is not None
        elif rule.term in ("Asimov", "HistFactory", "RooFit"):
            present = re.search(rf"\b{re.escape(rule.term)}\b", text) is not None
        else:
            present = re.search(rf"\b{re.escape(rule.term)}s?\b", text) is not None
        if not present:
            continue

        explained = any(r.search(text) is not None for r in rule.explain_regexes)
        if explained:
            continue

        issues.append(f"{path}: uses `{rule.term}` without explanation. {rule.suggestion}")
    return issues


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Terminology lint for public-facing docs.")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any issues are found (CI mode).",
    )
    ap.add_argument(
        "--paths",
        nargs="*",
        default=[
            "docs/README.md",
            "docs/personas",
            "docs/quickstarts",
        ],
        help="Paths to scan (files or directories).",
    )
    args = ap.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    paths = [repo / p for p in args.paths]
    rules = _compile_rules()

    all_issues: List[str] = []
    for md in _iter_md_files(paths):
        all_issues.extend(_find_issues(md, rules))

    if all_issues:
        for s in all_issues:
            print(s)
        if args.check:
            return 1
        return 0

    if os.environ.get("CI"):
        print("terminology-lint: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

