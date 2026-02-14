# Benchmark Protocol — Strict Agent Checklist

> **This is a MANDATORY protocol for any agent running benchmarks.**
> Read this BEFORE writing any benchmark code or reporting results.

---

## 1. PROHIBITED Actions

**NEVER do any of the following:**

- Qualitative assessments: "modest", "impressive", "needs context", "reasonable", "competitive"
- Suggestions: "what could be done", "it would be interesting to", "next steps"
- Running timing benchmarks WITHOUT parity verification first
- Reporting `warn` status without a 1-line root cause
- Submitting a report without a numbers table (see Section 2)
- Rounding speedups to hide poor results — report exact ratios
- Reporting "N/A" without explaining why the case was skipped
- Commentary, analysis, or editorializing — numbers only

---

## 2. Mandatory Report Format

Every benchmark report MUST end with this exact table format:

```
| Case | NS (median) | Competitor (median) | Speedup | Parity | Status |
|------|-------------|---------------------|---------|--------|--------|
| case_name | 5.2ms | 42.1ms | 8.1x | ok (Δcoef<1e-6) | pass |
| case_name | 12.0ms | 8.3ms | 0.7x | ok (Δnll<1e-8) | fail:slower |
```

**Required columns:**

| Column | Content | Example |
|--------|---------|---------|
| Case | Exact case name from suite | `glm_gaussian_10k` |
| NS (median) | Median time, unit included | `5.2ms` |
| Competitor (median) | Median time, unit included | `42.1ms` |
| Speedup | NS speed / competitor speed, 1 decimal | `8.1x` |
| Parity | `ok` or `fail` + metric + tolerance | `ok (Δcoef<1e-6)` |
| Status | `pass` / `fail` / `warn` + reason | `pass` |

**Status rules:**
- `pass` — NS faster AND parity ok
- `fail:slower` — NS slower than competitor
- `fail:parity` — parity check failed
- `warn:X` — edge case, must include 1-line reason (e.g. `warn:competitor_timeout`)

---

## 2b. GPU Timing Synchronization (mandatory for GPU benchmarks)

**GPU operations are asynchronous.** Without synchronization, `time.perf_counter()` measures dispatch time, NOT execution time. This gives falsely fast numbers.

**Rules:**
- **JAX/BlackJAX:** MUST call `.block_until_ready()` on output tensor BEFORE stopping the timer
- **NS LAPS (PyO3):** Implicitly synchronized — Rust `stream.synchronize()` runs inside `download_*()` before returning to Python. Wall time via `time.perf_counter()` around `sample_laps()` is correct.
- **PyTorch:** MUST call `torch.cuda.synchronize()` before stopping the timer
- **CuPy:** MUST call `cp.cuda.Stream.null.synchronize()` before stopping the timer

**Audit checklist for any new GPU benchmark script:**
1. Identify ALL `time.perf_counter()` stop points
2. For each stop point, verify sync call exists between last GPU op and timer stop
3. If sync is missing — **the benchmark is invalid, fix before reporting**

---

## 3. Pre-Run Checklist

Execute these steps IN ORDER before running any benchmark suite:

```bash
# 1. Verify NS Python API is importable
python -c "import nextstat; print(dir(nextstat))"

# 2. Verify competitor is installed + version
python -c "import <competitor>; print(<competitor>.__version__)"

# 3. Test NS API call — check return keys
python -c "
import nextstat
result = nextstat.<api_call>(...)
print(type(result))
print(list(result.keys()) if hasattr(result, 'keys') else dir(result))
"

# 4. Test competitor API call — check return format
python -c "
import <competitor>
result = <competitor>.<api_call>(...)
print(type(result))
"

# 5. Parity check on small data (N=100)
# Compare NS result vs competitor result, print delta
```

**If any step fails — STOP. Fix it before proceeding.**

---

## 4. Run Checklist

```bash
# 1. Run the suite
python benchmarks/nextstat-public-benchmarks/suites/<vertical>/run.py \
    --out-dir /tmp/bench_blitz_<vertical> --seed 42

# 2. Check exit code
echo $?  # Must be 0

# 3. Read JSON artifacts
cat /tmp/bench_blitz_<vertical>/results.json

# 4. Extract numbers into table (Section 2 format)
```

**If exit code ≠ 0 — report the error, do NOT fabricate results.**

---

## 5. Post-Run Checklist

1. **Numbers table** in mandatory format (Section 2) — NO EXCEPTIONS
2. **Parity status** for each case — metric name + delta value
3. **If warn/fail** — exactly 1 line explaining why
4. **NO commentary, advice, analysis, or suggestions**

The report is ONLY the table. Nothing else.

---

## 6. Infrastructure

| Resource | Value |
|----------|-------|
| Benchmark server | `ssh nextstat-bench` (EPYC 7502P 32C/64T, 128GB) |
| Repo path | `/data/nextstat` |
| Venv | `/data/nextstat/.venv` |
| Build | `.venv/bin/maturin develop --release` |
| Rsync to server | `rsync -az --exclude '.venv/' --exclude 'target/' --exclude 'target-*/' . nextstat-bench:/data/nextstat/` |
| Activate venv | `source /data/nextstat/.venv/bin/activate` |
| Cargo target | `/data/cargo-target` (set via CARGO_TARGET_DIR) |

**Before running on server:**
1. rsync latest code
2. Build: `cd /data/nextstat && .venv/bin/maturin develop --release`
3. Verify build: `python -c "import nextstat; print('ok')"`

---

## 6b. Mandatory Environment Snapshot

**EVERY benchmark report MUST begin with this environment snapshot.**
No exceptions. Run these commands on the bench machine and paste the output.

### CPU benchmarks (nextstat-bench):
```bash
python -c "import nextstat; print('nextstat', nextstat.__version__)"
python --version
rustc --version
uname -r
lscpu | grep 'Model name'
```

### GPU benchmarks (A100/any GPU server):
```bash
# GPU hardware + driver
nvidia-smi --query-gpu=name,driver_version,cuda_version,compute_mode,power.limit --format=csv,noheader

# CUDA toolkit
nvcc --version | grep release

# Python + key packages
python --version
python -c "import nextstat; print('nextstat', nextstat.__version__)"
python -c "import jax; print('jax', jax.__version__); import jaxlib; print('jaxlib', jaxlib.__version__); print('backend:', jax.devices()[0].platform, jax.devices()[0].device_kind)"
python -c "import blackjax; print('blackjax', blackjax.__version__)"
rustc --version

# JAX compilation cache (affects warm timings)
echo "JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-unset}"
python -c "import jax; cc = jax.config.jax_compilation_cache_dir; print('jax_compilation_cache_dir:', cc if cc else 'disabled')"

# XLA flags (affects codegen)
echo "XLA_FLAGS=${XLA_FLAGS:-unset}"
echo "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-unset}"
```

### Report format:
Paste the raw output at the top of your report, inside a `## Environment` section:
```
## Environment
- GPU: NVIDIA A100-SXM4-80GB, driver 535.129.03, CUDA 12.2, power 400W
- nvcc: 12.2.140
- Python: 3.12.1
- nextstat: 0.9.3
- jax: 0.4.35, jaxlib: 0.4.35+cuda12, backend: gpu NVIDIA A100-SXM4-80GB
- blackjax: 1.3.0
- rustc: 1.93.0
- JAX compilation cache: disabled
- XLA_FLAGS: unset
```

**If you cannot run these commands — STOP. Fix the environment first.**

---

## 7. Executive Summary Table (mandatory at end of multi-suite runs)

When running multiple suites, end with this summary:

```
| Vertical | Best Competitor | Speedup Range | Parity |
|----------|----------------|---------------|--------|
| GLM | statsmodels | 2.1-7.3x | ok |
| Bayesian | CmdStan | 0.8-1.4x | ok |
| Panel | linearmodels | 4.2-10.8x | ok |
```

---

## 8. Examples

### CORRECT report:

```
| Case | NS (median) | Competitor (median) | Speedup | Parity | Status |
|------|-------------|---------------------|---------|--------|--------|
| panel_fe_16k | 12.8ms | 137.6ms | 10.8x | ok (Δcoef<1e-8) | pass |
| did_twfe_16k | 19.9ms | 197.0ms | 9.9x | ok (Δcoef<1e-7) | pass |
| iv_2sls_16k | 5.8ms | 24.1ms | 4.2x | ok (Δcoef<1e-6) | pass |
```

### WRONG report (every line violates the protocol):

```
The results are quite impressive for panel fixed effects, showing a solid
10x improvement. The IV case is more modest but still competitive.
It would be worth investigating whether larger datasets show even better
scaling. Overall, NextStat performs well in this vertical.
```

**Why it's wrong:** No table, no numbers, qualitative adjectives ("impressive", "modest", "competitive"), suggestions ("worth investigating"), commentary ("overall... performs well").
