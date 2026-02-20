#!/usr/bin/env python3
"""Theophylline 1-cpt oral PK benchmark.

Reference: Boeckmann AJ, Sheiner LB, Beal SL. NONMEM Users Guide, Part V.
Dataset: 12 subjects, oral dosing, plasma concentrations.
Model: 1-compartment oral, FOCE, proportional error.

The theophylline dataset is the canonical NONMEM Example 1 dataset. Data
below is the publicly available version from Boeckmann et al. (1994).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Theophylline pharmacokinetic data -- 12 subjects, oral administration.
# Each entry: id, weight (kg), dose (mg/kg), sampling times (h), concentrations (mg/L).
# Source: Boeckmann et al. NONMEM Users Guide (1994), Example 1.
THEO_DATA = [
    {"id": 1,  "wt": 79.6, "dose": 4.02, "times": [0.00, 0.25, 0.57, 1.12, 2.02, 3.82, 5.10, 7.03,  9.05, 12.12, 24.37], "conc": [0.74, 2.84, 6.57, 10.50, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28]},
    {"id": 2,  "wt": 72.4, "dose": 4.40, "times": [0.00, 0.27, 0.52, 1.00, 1.92, 3.50, 5.02, 7.03,  9.00, 12.00, 24.30], "conc": [0.00, 1.72, 7.91, 8.31, 8.33, 6.85, 6.08, 5.40, 4.55, 3.01, 0.90]},
    {"id": 3,  "wt": 70.5, "dose": 4.53, "times": [0.00, 0.27, 0.58, 1.02, 2.02, 3.62, 5.08, 7.07,  9.00, 12.15, 24.17], "conc": [0.00, 4.40, 6.90, 8.20, 7.80, 7.50, 6.20, 5.30, 4.90, 3.70, 1.05]},
    {"id": 4,  "wt": 72.7, "dose": 4.40, "times": [0.00, 0.35, 0.60, 1.07, 2.13, 3.50, 5.02, 7.02,  9.02, 12.00, 24.65], "conc": [0.00, 1.89, 4.60, 8.60, 8.38, 7.54, 6.88, 5.78, 5.33, 4.19, 1.15]},
    {"id": 5,  "wt": 54.6, "dose": 5.86, "times": [0.00, 0.30, 0.52, 1.00, 2.02, 3.50, 5.02, 7.02,  9.10, 12.00, 24.35], "conc": [0.00, 2.02, 5.63, 11.40, 9.33, 8.74, 7.56, 7.09, 5.90, 4.37, 1.57]},
    {"id": 6,  "wt": 80.0, "dose": 4.00, "times": [0.00, 0.27, 0.58, 1.15, 2.03, 3.57, 5.00, 7.00,  9.22, 12.10, 23.85], "conc": [0.00, 1.29, 3.08, 6.44, 6.32, 5.53, 4.94, 4.02, 3.46, 2.78, 0.92]},
    {"id": 7,  "wt": 64.6, "dose": 4.95, "times": [0.00, 0.25, 0.50, 0.98, 1.98, 3.60, 5.02, 7.17,  9.00, 12.12, 24.08], "conc": [0.15, 0.85, 2.35, 5.02, 6.58, 7.09, 6.66, 5.25, 4.39, 3.53, 1.15]},
    {"id": 8,  "wt": 70.5, "dose": 4.53, "times": [0.00, 0.25, 0.52, 0.98, 2.02, 3.53, 5.05, 7.15,  9.00, 12.12, 24.22], "conc": [0.00, 3.05, 3.05, 7.31, 7.56, 6.59, 5.88, 4.73, 4.57, 3.00, 1.25]},
    {"id": 9,  "wt": 86.4, "dose": 3.70, "times": [0.00, 0.30, 0.63, 1.05, 2.02, 3.53, 5.02, 7.17,  9.00, 12.10, 24.12], "conc": [0.00, 7.37, 9.03, 7.14, 6.33, 5.66, 5.67, 4.24, 4.11, 3.16, 1.12]},
    {"id": 10, "wt": 58.2, "dose": 5.50, "times": [0.00, 0.37, 0.77, 1.02, 2.05, 3.55, 5.05, 7.08,  9.00, 12.10, 24.22], "conc": [0.24, 2.89, 5.22, 6.41, 7.83, 10.21, 9.18, 8.02, 7.14, 5.68, 2.42]},
    {"id": 11, "wt": 65.0, "dose": 4.92, "times": [0.00, 0.25, 0.50, 0.98, 1.98, 3.60, 5.02, 7.03,  9.03, 12.12, 24.08], "conc": [0.00, 4.86, 7.24, 8.00, 6.81, 5.87, 5.22, 4.45, 3.62, 2.69, 0.86]},
    {"id": 12, "wt": 60.5, "dose": 5.30, "times": [0.00, 0.25, 0.50, 1.00, 2.00, 3.52, 5.07, 7.07,  9.03, 12.05, 24.15], "conc": [0.00, 1.25, 3.96, 7.82, 9.72, 9.75, 8.57, 6.59, 6.11, 4.57, 1.17]},
]

# Reference parameter estimates from NONMEM 7.5 Example 1.
# 1-compartment oral model: CL/F, V/F, Ka.
# OFV is -2LL (objective function value).
REFERENCE = {
    "tool": "NONMEM 7.5",
    "method": "FOCE",
    "theta": {"CL_F": 0.0309, "V_F": 0.456, "Ka": 1.51},
    "omega": {"CL_F": 0.0270, "V_F": 0.0138},
    "sigma": 0.0164,
    "ofv": -115.3,
}


def _flatten_theo_data():
    """Flatten THEO_DATA into arrays for nlme_foce.

    Returns:
        times, y, subject_idx, n_subjects, dose
    """
    times = []
    y = []
    subject_idx = []

    for i, subj in enumerate(THEO_DATA):
        for t, c in zip(subj["times"], subj["conc"]):
            if t == 0.0 and c <= 0.0:
                # Skip time=0 pre-dose records with zero/negative conc
                continue
            times.append(t)
            y.append(c)
            subject_idx.append(i)

    n_subjects = len(THEO_DATA)
    # Use weight-normalized dose: mean dose across subjects
    # NONMEM Example 1 uses per-subject dose, but nlme_foce takes scalar dose.
    # We use mean dose * mean weight for total dose (mg).
    mean_dose_per_kg = sum(s["dose"] for s in THEO_DATA) / n_subjects
    mean_wt = sum(s["wt"] for s in THEO_DATA) / n_subjects
    dose = mean_dose_per_kg * mean_wt  # total mg

    return times, y, subject_idx, n_subjects, dose


def bench_nextstat(seed=42, n_repeats=3):
    """Run NextStat nlme_foce on theophylline data.

    Returns:
        dict with keys: theta, omega, ofv, converged, wall_s (median), wall_s_all
    """
    REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(REPO_ROOT / "bindings" / "ns-py" / "python"))
    import nextstat

    times, y, subject_idx, n_subjects, dose = _flatten_theo_data()

    # Initial estimates (close to NONMEM reference)
    theta_init = [0.04, 0.5, 1.0]  # CL_F, V_F, Ka
    omega_init = [0.04, 0.04]       # omega_CL, omega_V (diagonal)

    wall_times = []
    result = None

    for rep in range(n_repeats):
        t0 = time.perf_counter()
        result = nextstat.nlme_foce(
            times, y, subject_idx, n_subjects,
            dose=dose,
            bioavailability=1.0,
            error_model="proportional",
            sigma=0.1,
            theta_init=theta_init,
            omega_init=omega_init,
            max_outer_iter=100,
            max_inner_iter=50,
            tol=1e-6,
            interaction=False,
        )
        wall = time.perf_counter() - t0
        wall_times.append(wall)

    wall_times.sort()
    median_wall = wall_times[len(wall_times) // 2]

    return {
        "tool": f"NextStat {nextstat.__version__}",
        "method": "FOCE",
        "theta": dict(zip(["CL_F", "V_F", "Ka"], result["theta"])),
        "omega": result["omega"],
        "omega_matrix": result["omega_matrix"],
        "ofv": result["ofv"],
        "converged": result["converged"],
        "n_iter": result["n_iter"],
        "wall_s": median_wall,
        "wall_s_all": wall_times,
        "seed": seed,
    }


def bench_nlmixr2(seed=42, n_repeats=3):
    """Run nlmixr2 FOCE on theophylline data (requires R + nlmixr2).

    Returns:
        dict with results, or None if nlmixr2 not available.
    """
    import shutil
    import subprocess
    import json
    import tempfile

    if not shutil.which("Rscript"):
        print("SKIP: Rscript not found")
        return None

    # Check nlmixr2 availability
    check = subprocess.run(
        ["Rscript", "-e", "library(nlmixr2); cat('ok')"],
        capture_output=True, text=True, timeout=30,
    )
    if check.returncode != 0 or "ok" not in check.stdout:
        print("SKIP: nlmixr2 not installed")
        return None

    # Build data frame as CSV
    rows = []
    for subj in THEO_DATA:
        total_dose = subj["dose"] * subj["wt"]
        # Dosing record
        rows.append(f'{subj["id"]},0,0,{total_dose:.2f},1')
        for t, c in zip(subj["times"], subj["conc"]):
            if t == 0.0 and c <= 0.0:
                continue
            rows.append(f'{subj["id"]},{t},{c},0,0')
    csv_data = "ID,TIME,DV,AMT,EVID\\n" + "\\n".join(rows)

    r_script = f'''
library(nlmixr2)
library(jsonlite)

d <- read.csv(textConnection("{csv_data}"))

one.cmt <- function() {{
  ini({{
    tka <- log(1.0)
    tcl <- log(0.04)
    tv  <- log(0.5)
    eta.cl ~ 0.04
    eta.v  ~ 0.04
    prop.sd <- 0.1
  }})
  model({{
    ka <- exp(tka)
    cl <- exp(tcl + eta.cl)
    v  <- exp(tv  + eta.v)
    linCmt() ~ prop(prop.sd)
  }})
}}

times <- numeric({n_repeats})
result <- NULL
for (i in 1:{n_repeats}) {{
  t0 <- proc.time()
  result <- nlmixr2(one.cmt, d, est="foce", control=foceiControl(print=0))
  times[i] <- (proc.time() - t0)[3]
}}

out <- list(
  tool = paste0("nlmixr2 ", packageVersion("nlmixr2")),
  method = "FOCE",
  theta = list(
    CL_F = as.numeric(result$theta["tcl"]),
    V_F  = as.numeric(result$theta["tv"]),
    Ka   = as.numeric(result$theta["tka"])
  ),
  ofv = as.numeric(result$objective),
  wall_s = median(times),
  wall_s_all = times
)
cat(toJSON(out, auto_unbox=TRUE))
'''

    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
        f.write(r_script)
        r_path = f.name

    try:
        proc = subprocess.run(
            ["Rscript", r_path],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            print(f"SKIP: nlmixr2 failed: {proc.stderr[:200]}")
            return None
        return json.loads(proc.stdout)
    except Exception as e:
        print(f"SKIP: nlmixr2 error: {e}")
        return None
    finally:
        Path(r_path).unlink(missing_ok=True)


def run(seed=42, n_repeats=3) -> dict[str, Any]:
    """Run all available benchmarks and return results dict.

    Returns:
        dict with keys: model, reference, nextstat, nlmixr2 (if available)
    """
    results: dict[str, Any] = {
        "model": "theophylline_1cpt_oral",
        "description": "Theophylline 1-cpt oral PK, 12 subjects, FOCE",
        "reference": REFERENCE,
    }

    print(f"\n  [Theophylline] NextStat FOCE ({n_repeats} repeats)...")
    ns_result = bench_nextstat(seed=seed, n_repeats=n_repeats)
    results["nextstat"] = ns_result
    print(f"    OFV={ns_result['ofv']:.2f}  wall={ns_result['wall_s']*1000:.1f}ms  converged={ns_result['converged']}")

    print(f"  [Theophylline] nlmixr2 FOCE...")
    nlmixr2_result = bench_nlmixr2(seed=seed, n_repeats=n_repeats)
    if nlmixr2_result is not None:
        results["nlmixr2"] = nlmixr2_result
        print(f"    OFV={nlmixr2_result['ofv']:.2f}  wall={nlmixr2_result['wall_s']*1000:.1f}ms")

    return results
