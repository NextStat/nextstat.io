#!/usr/bin/env python3
"""Competitor benchmark wrappers for pharma suite.

Each wrapper:
1. Checks if the competitor tool is installed
2. Runs the model with the same data
3. Returns standardized result dict

Supported competitors:
- nlmixr2 (R package, via subprocess)
- Pumas.jl (Julia package, via subprocess)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# nlmixr2 (R)
# ---------------------------------------------------------------------------

def check_nlmixr2() -> bool:
    """Check if nlmixr2 is available via R."""
    if not shutil.which("Rscript"):
        return False
    try:
        proc = subprocess.run(
            ["Rscript", "-e", "library(nlmixr2); cat(packageVersion('nlmixr2'))"],
            capture_output=True, text=True, timeout=30,
        )
        return proc.returncode == 0 and len(proc.stdout.strip()) > 0
    except Exception:
        return False


def nlmixr2_version() -> Optional[str]:
    """Return nlmixr2 version string, or None if not available."""
    if not shutil.which("Rscript"):
        return None
    try:
        proc = subprocess.run(
            ["Rscript", "-e", "cat(as.character(packageVersion('nlmixr2')))"],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return None


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
            capture_output=True, text=True, timeout=timeout,
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
# Pumas.jl (Julia)
# ---------------------------------------------------------------------------

def check_pumas() -> bool:
    """Check if Pumas.jl is available."""
    if not shutil.which("julia"):
        return False
    try:
        proc = subprocess.run(
            ["julia", "-e", "using Pumas; println(pkgversion(Pumas))"],
            capture_output=True, text=True, timeout=120,
        )
        return proc.returncode == 0 and len(proc.stdout.strip()) > 0
    except Exception:
        return False


def pumas_version() -> Optional[str]:
    """Return Pumas.jl version string, or None if not available."""
    if not shutil.which("julia"):
        return None
    try:
        proc = subprocess.run(
            ["julia", "-e", "using Pumas; println(pkgversion(Pumas))"],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return None


def run_pumas(
    julia_script: str,
    timeout: int = 600,
) -> Optional[dict[str, Any]]:
    """Run a Pumas Julia script and parse JSON output.

    Args:
        julia_script: Complete Julia script that prints JSON to stdout.
        timeout: Max execution time in seconds.

    Returns:
        Parsed dict from JSON output, or None on failure.
    """
    if not check_pumas():
        print("SKIP: Pumas.jl not installed")
        return None

    with tempfile.NamedTemporaryFile(suffix=".jl", mode="w", delete=False) as f:
        f.write(julia_script)
        jl_path = f.name

    try:
        proc = subprocess.run(
            ["julia", jl_path],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            print(f"SKIP: Pumas failed (exit {proc.returncode}): {proc.stderr[:300]}")
            return None

        stdout = proc.stdout.strip()
        json_start = stdout.rfind("{")
        if json_start < 0:
            print(f"SKIP: Pumas no JSON in output")
            return None
        json_str = stdout[json_start:]
        return json.loads(json_str)
    except subprocess.TimeoutExpired:
        print(f"SKIP: Pumas timed out after {timeout}s")
        return None
    except json.JSONDecodeError as e:
        print(f"SKIP: Pumas JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"SKIP: Pumas error: {e}")
        return None
    finally:
        Path(jl_path).unlink(missing_ok=True)


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

    if check_pumas():
        competitors["pumas"] = pumas_version()
    else:
        competitors["pumas"] = None

    return competitors
