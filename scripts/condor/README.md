# HTCondor templates (Apex2 ROOT/TRExFitter parity)

These files are **templates** intended for cluster runs (e.g. CERN lxplus/HTCondor).
They assume you have:
- a shared filesystem visible to worker nodes (recommended), and
- a Python environment where `nextstat` is importable, plus ROOT/HistFactory tools (`root`, `hist2workspace`).

Files:
- `apex2_root_suite_single.sub`: one job runs the full ROOT parity suite (all cases sequentially).
- `apex2_root_suite_array.sub`: job-array mode (one job per case index).
- `run_apex2_root_suite_single.sh`: wrapper for the single-job submit file.
- `run_apex2_root_suite_case.sh`: wrapper for the job-array submit file.
- Render a `.sub` with correct `queue N` from a cases JSON:
  - `scripts/condor/render_apex2_root_suite_array_sub.py`
- Aggregate per-case JSONs back into one suite report:
  - `tests/aggregate_apex2_root_suite_reports.py`
- Optional: compare aggregated perf vs baseline:
  - `tests/compare_apex2_root_suite_to_baseline.py`

Note: the `.sub` templates use `executable = /bin/bash` and pass the wrapper script as an argument, so the
wrapper `.sh` files do not need executable permissions.

Before submitting:
- edit `initialdir = ...` to point at your checkout of this repo
- export the variables noted in each `.sub` file (e.g. `APEX2_ROOT_CASES_JSON`, `APEX2_TREX_SEARCH_DIR`)

Optional helper (avoid manual edit of `queue <N>`):

```bash
python3 scripts/condor/render_apex2_root_suite_array_sub.py \
  --cases /abs/path/to/apex2_root_cases.json \
  --initialdir /abs/path/to/nextstat.io \
  --out apex2_root_suite_array.sub
```

Submit:

```bash
mkdir -p scripts/condor/logs
condor_submit scripts/condor/apex2_root_suite_single.sub
```

or:

```bash
mkdir -p scripts/condor/logs
condor_submit scripts/condor/apex2_root_suite_array.sub
```
