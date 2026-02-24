#!/usr/bin/env python3
"""Competitor benchmark wrappers for pharma suite.

Each wrapper:
1. Checks if the competitor tool is installed
2. Runs the model with the same data
3. Returns standardized result dict

Supported competitors:
- nlmixr2 (R package, via subprocess)
- Stan/Torsten baseline (cmdstanr R package)
- saemix (R package)
- Pharmpy (Python package)
- MaS (Python package)
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional


def _r_pkg_available(pkg: str) -> bool:
    if not shutil.which("Rscript"):
        return False
    try:
        proc = subprocess.run(
            ["Rscript", "-e", f"cat(requireNamespace('{pkg}', quietly=TRUE))"],
            capture_output=True, text=True, timeout=30, env=os.environ.copy(),
        )
        return proc.returncode == 0 and proc.stdout.strip().lower() == "true"
    except Exception:
        return False


def _r_pkg_version(pkg: str) -> Optional[str]:
    if not shutil.which("Rscript"):
        return None
    try:
        proc = subprocess.run(
            ["Rscript", "-e", f"if(requireNamespace('{pkg}', quietly=TRUE)) cat(as.character(utils::packageVersion('{pkg}')))"],
            capture_output=True, text=True, timeout=30, env=os.environ.copy(),
        )
        if proc.returncode == 0:
            out = proc.stdout.strip()
            return out if out else None
    except Exception:
        pass
    return None


def _py_mod_available(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# nlmixr2 (R)
# ---------------------------------------------------------------------------

def check_nlmixr2() -> bool:
    """Check if nlmixr2 is available via R."""
    return _r_pkg_available("nlmixr2")


def nlmixr2_version() -> Optional[str]:
    """Return nlmixr2 version string, or None if not available."""
    return _r_pkg_version("nlmixr2")


def check_torsten_cmdstanr() -> bool:
    """Check if Stan baseline tooling (cmdstanr) is available in R."""
    return _r_pkg_available("cmdstanr")


def torsten_cmdstanr_version() -> Optional[str]:
    return _r_pkg_version("cmdstanr")


def check_saemix() -> bool:
    return _r_pkg_available("saemix")


def saemix_version() -> Optional[str]:
    return _r_pkg_version("saemix")


def check_pharmpy() -> bool:
    return _py_mod_available("pharmpy")


def pharmpy_version() -> Optional[str]:
    if not check_pharmpy():
        return None
    try:
        proc = subprocess.run(
            ["python3", "-c", "import pharmpy; print(getattr(pharmpy, '__version__', 'unknown'))"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            out = proc.stdout.strip()
            return out if out else "installed"
    except Exception:
        pass
    return "installed"


def check_mas() -> bool:
    # Common python module names seen in MaS distributions.
    return _py_mod_available("mas") or _py_mod_available("MaS")


def mas_version() -> Optional[str]:
    if not check_mas():
        return None
    return "installed"


def run_nlmixr2_foce(
    r_script: str,
    timeout: int = 600,
) -> Optional[dict[str, Any]]:
    """Run an nlmixr2 R script and parse JSON output.

    Args:
        r_script: Complete R script that prints JSON to stdout.
        timeout: Max execution time in seconds.

    Returns:
        Parsed dict from JSON output, or None on failure.
    """
    if not check_nlmixr2():
        print("SKIP: nlmixr2 not installed")
        return None

    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
        f.write(r_script)
        r_path = f.name

    try:
        proc = subprocess.run(
            ["Rscript", r_path],
            capture_output=True, text=True, timeout=timeout, env=os.environ.copy(),
        )
        if proc.returncode != 0:
            print(f"SKIP: nlmixr2 failed (exit {proc.returncode}): {proc.stderr[:300]}")
            return None

        # Parse the last line of stdout as JSON (nlmixr2 may print progress)
        stdout = proc.stdout.strip()
        # Find JSON object/array in output
        json_start = stdout.rfind("{")
        if json_start < 0:
            print(f"SKIP: nlmixr2 no JSON in output")
            return None
        json_str = stdout[json_start:]
        return json.loads(json_str)
    except subprocess.TimeoutExpired:
        print(f"SKIP: nlmixr2 timed out after {timeout}s")
        return None
    except json.JSONDecodeError as e:
        print(f"SKIP: nlmixr2 JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"SKIP: nlmixr2 error: {e}")
        return None
    finally:
        Path(r_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Competitor availability summary
# ---------------------------------------------------------------------------

def list_available() -> dict[str, Optional[str]]:
    """Return dict of competitor names and their versions (None if not installed)."""
    competitors = {}

    if check_nlmixr2():
        competitors["nlmixr2"] = nlmixr2_version()
    else:
        competitors["nlmixr2"] = None

    if check_torsten_cmdstanr():
        competitors["torsten"] = torsten_cmdstanr_version()
    else:
        competitors["torsten"] = None

    if check_saemix():
        competitors["saemix"] = saemix_version()
    else:
        competitors["saemix"] = None

    if check_pharmpy():
        competitors["pharmpy"] = pharmpy_version()
    else:
        competitors["pharmpy"] = None

    if check_mas():
        competitors["mas"] = mas_version()
    else:
        competitors["mas"] = None

    return competitors
