from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def assert_doc_contains_strings(doc_path: Path, required_strings: Sequence[str]) -> None:
    doc = doc_path.read_text(encoding="utf-8")
    missing = [entry for entry in required_strings if entry not in doc]
    assert not missing, f"{doc_path} missing contract references: {missing}"
