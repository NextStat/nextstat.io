"""IQ (Installation Qualification) test cases.

Implements IQ test cases from the IQ/OQ/PQ protocol (NS-VAL-001 v2.0.0,
Section 2).  Each test checks installation, imports, dependencies, etc.
"""
from __future__ import annotations

import time
from typing import Any


def _make_result(
    test_id: str,
    section: str,
    title: str,
    ok: bool | None,
    observed: dict[str, Any],
    acceptance: str,
    deviation: str | None,
    wall_s: float,
) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "category": "IQ",
        "section": section,
        "title": title,
        "ok": ok,
        "observed": observed,
        "acceptance": acceptance,
        "deviation": deviation,
        "wall_s": round(wall_s, 6),
    }


# ------------------------------------------------------------------
# Individual test functions
# ------------------------------------------------------------------

def _iq_inst_002() -> dict[str, Any]:
    """Verify import -- import nextstat; nextstat.__version__."""
    t0 = time.monotonic()
    try:
        import nextstat  # type: ignore[import-untyped]

        version = getattr(nextstat, "__version__", None)
        ok = version is not None
        observed = {"version": version}
        deviation = None if ok else "__version__ attribute missing"
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Import failed: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="IQ-INST-002",
        section="2.1",
        title="Verify import -- import nextstat; nextstat.__version__",
        ok=ok,
        observed=observed,
        acceptance="Import succeeds, version printed",
        deviation=deviation,
        wall_s=wall_s,
    )


def _iq_inst_003() -> dict[str, Any]:
    """Verify pharmacometrics API imports."""
    t0 = time.monotonic()
    names = ["nlme_foce", "nlme_saem", "pk_vpc", "pk_gof", "read_nonmem"]
    found: list[str] = []
    missing: list[str] = []
    error: str | None = None
    try:
        import nextstat._core as _core  # type: ignore[import-untyped]

        for name in names:
            if hasattr(_core, name):
                found.append(name)
            else:
                missing.append(name)
        ok = len(missing) == 0
        observed = {"found": found, "missing": missing}
        deviation = f"Missing: {', '.join(missing)}" if missing else None
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Import failed: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="IQ-INST-003",
        section="2.2",
        title="Verify pharmacometrics API (nlme_foce, nlme_saem, pk_vpc, pk_gof, read_nonmem)",
        ok=ok,
        observed=observed,
        acceptance="All five names importable from nextstat._core",
        deviation=deviation,
        wall_s=wall_s,
    )


def _iq_inst_004() -> dict[str, Any]:
    """Verify PK model classes."""
    t0 = time.monotonic()
    model_names = [
        "OneCompartmentOralPkModel",
        "TwoCompartmentIvPkModel",
        "TwoCompartmentOralPkModel",
        "ThreeCompartmentIvPkModel",
        "ThreeCompartmentOralPkModel",
    ]
    found: list[str] = []
    missing: list[str] = []
    try:
        import nextstat._core as _core  # type: ignore[import-untyped]

        for name in model_names:
            if hasattr(_core, name):
                found.append(name)
            else:
                missing.append(name)
        ok = len(missing) == 0
        observed = {"found": found, "missing": missing}
        deviation = f"Missing: {', '.join(missing)}" if missing else None
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Import failed: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="IQ-INST-004",
        section="2.3",
        title="Verify PK model classes importable from nextstat._core",
        ok=ok,
        observed=observed,
        acceptance="All five PK model classes importable",
        deviation=deviation,
        wall_s=wall_s,
    )


def _iq_dep_001() -> dict[str, Any]:
    """Verify numpy -- import numpy; numpy.__version__ >= 1.21."""
    t0 = time.monotonic()
    try:
        import numpy as np  # type: ignore[import-untyped]

        version_str: str = np.__version__
        parts = version_str.split(".")
        major, minor = int(parts[0]), int(parts[1])
        ok = (major, minor) >= (1, 21)
        observed = {"numpy_version": version_str}
        deviation = (
            None
            if ok
            else f"numpy {version_str} < 1.21 minimum requirement"
        )
    except Exception as exc:
        ok = False
        observed = {"error": str(exc)}
        deviation = f"Import failed: {exc}"
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="IQ-DEP-001",
        section="2.4",
        title="Verify numpy >= 1.21",
        ok=ok,
        observed=observed,
        acceptance="numpy importable with version >= 1.21",
        deviation=deviation,
        wall_s=wall_s,
    )


def _iq_conf_001() -> dict[str, Any]:
    """BLAS/LAPACK backend -- check numpy config (informational, always pass)."""
    t0 = time.monotonic()
    try:
        import numpy as np  # type: ignore[import-untyped]

        # numpy >= 1.24 exposes show_config(mode="dicts")
        config_info: dict[str, Any] = {}
        if hasattr(np, "show_config"):
            try:
                info = np.show_config(mode="dicts")  # type: ignore[call-arg]
                if isinstance(info, dict):
                    blas = info.get("Build Dependencies", {}).get("blas", {})
                    lapack = info.get("Build Dependencies", {}).get("lapack", {})
                    config_info = {"blas": blas, "lapack": lapack}
                else:
                    config_info = {"raw": str(info)[:500]}
            except TypeError:
                # Older numpy without mode= kwarg
                config_info = {"note": "numpy.show_config(mode='dicts') not supported"}
        ok = True  # Informational -- always passes
        observed = {"numpy_version": np.__version__, "config": config_info}
        deviation = None
    except Exception as exc:
        ok = True  # Still pass -- informational only
        observed = {"error": str(exc)}
        deviation = None
    wall_s = time.monotonic() - t0
    return _make_result(
        test_id="IQ-CONF-001",
        section="2.5",
        title="BLAS/LAPACK backend (informational)",
        ok=ok,
        observed=observed,
        acceptance="Report BLAS/LAPACK config (always pass)",
        deviation=deviation,
        wall_s=wall_s,
    )


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def run_iq_tests() -> list[dict[str, Any]]:
    """Run all IQ tests. Returns list of test results."""
    return [
        _iq_inst_002(),
        _iq_inst_003(),
        _iq_inst_004(),
        _iq_dep_001(),
        _iq_conf_001(),
    ]
