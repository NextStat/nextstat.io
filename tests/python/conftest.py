"""Pytest config for Python regression tests.

We keep helper modules (like `_tolerances.py`) alongside tests and ensure they
are importable regardless of how pytest is invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `tests/python` importable as a top-level module path.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
