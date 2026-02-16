# Release Gates (Apex2, no cluster)

This document describes the minimal **pre-release gate** based on the Apex2 baseline workflow.

## What it checks

- **pyhf parity** vs the pyhf reference implementation
- **P6 GLM** end-to-end fit/predict stability vs a recorded baseline
- **Performance regressions** vs the latest baseline manifest on the same machine
- **ROOT suite** (optional): runs the recorded ROOT/HistFactory parity case pack and gates on *regressions vs the recorded baseline* (baseline failures are treated as expected)

Artifacts:
- baselines: `tmp/baselines/`
- compare report: `tmp/baseline_compare_report.json`

## Step-by-step (recommended)

1) Record baselines once on a reference machine:

```bash
make apex2-baseline-record
```

2) Before cutting a release, run the pre-release gate:

```bash
make apex2-baseline-compare COMPARE_ARGS="--require-same-host --p6-attempts 2"
```

or:

```bash
make apex2-pre-release-gate
```

This runs:
- `cargo build --workspace --release`
- `cargo test --workspace`
- `maturin develop --release` (Python bindings)
- `pytest -m "not slow" tests/python`
- `tests/compare_with_latest_baseline.py --require-same-host --p6-attempts 2` (P6 retried up to N times; whole compare retried once if it fails with `rc=2`)
- If `tmp/baselines/latest_root_manifest.json` exists: the gate also runs the ROOT suite case pack recorded there and compares it to the baseline (expected ROOT divergences recorded as baseline failures do not gate).

Optional (pytest timing breakdown):
- CLI flag: `pytest -m "not slow" tests/python --ns-test-timings`
- Env var: `NS_TEST_TIMINGS=1 pytest -m "not slow" tests/python`
- JSON output: `--ns-test-timings-json tmp/pytest_timings.json` (or `NS_TEST_TIMINGS_JSON=tmp/pytest_timings.json`)

Optional (GPU backends):
- CUDA (requires `nvcc`): `APEX2_CARGO_TEST_ARGS="--workspace --all-features" make apex2-pre-release-gate`

Optional (TREx analysis spec):
- Record once: `make trex-spec-baseline-record TREX_SPEC=docs/specs/trex/canonical/histfactory_fixture_baseline.yaml`
- Compare before release: `make trex-spec-baseline-compare TREX_COMPARE_ARGS="--require-same-host"`
- If `tmp/baselines/latest_trex_analysis_spec_manifest.json` exists, `make apex2-pre-release-gate` also runs this compare.
- Set `APEX2_SKIP_TREX_SPEC=1` to skip, or override args with `APEX2_TREX_COMPARE_ARGS="--require-same-host"`.

Notes:

- `maturin develop` installs the compiled extension into the active venv. The built binary is **not** committed to git.

Exit codes:
- `0`: OK (parity OK and within slowdown thresholds)
- `2`: FAIL (parity failure or slowdown threshold exceeded)
- `3`: baseline manifest missing/invalid
- `4`: runner error (missing deps, crash, etc.)
- `5`: git working tree dirty (set `APEX2_ALLOW_DIRTY=1` to override)

## If it fails

- Open `tmp/baseline_compare_report.json` and check:
  - `pyhf.compare.cases[*].ok` for perf regressions
  - `p6_glm.attempts[*].status` to see retry outcomes
  - `p6_glm.compare.compare.cases[*].ok` for fit/predict regressions (selected attempt)
- Note: by default, very small baseline timings are **not** gated (to avoid timer noise):
  - pyhf: `--pyhf-min-baseline-s` (default `1e-5`)
  - P6: `--p6-min-baseline-fit-s` and `--p6-min-baseline-predict-s` (default `1e-2`)
  Override via `COMPARE_ARGS="--require-same-host"` if you want stricter gating on a dedicated benchmark host.
- If the baseline is stale (e.g. after a known perf improvement), record a new baseline and re-run the gate.

## Cluster notes (ROOT/TRExFitter)

ROOT/HistFactory parity baselines are recorded separately on a cluster environment (e.g. lxplus) via:
- `PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_baseline.py --only root --root-search-dir /abs/path/to/trex/output`

TREx replacement baselines (numbers-only: fit + expected_data surfaces) can also be recorded from a TRExFitter/HistFactory export dir via:
- `PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_trex_baseline.py --export-dir tests/fixtures/trex_exports/tttt-prod`

Typical cluster gate workflow:
- Record a ROOT baseline once (on the cluster machine where ROOT/TRExFitter are available).
- Run the ROOT suite via HTCondor (single job or array), then aggregate JSON outputs.
- Optionally compare perf vs the recorded baseline (JSON-only, no ROOT needed for the compare step).

Notes:
- For large suites, it can be faster to record the baseline via HTCondor array + aggregation, then register it with:
  `PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/record_baseline.py --only root --root-suite-existing tmp/apex2_root_suite_aggregate.json --root-cases-existing tests/fixtures/trex_parity_pack/cases_trex_exports.json --out-dir tmp/baselines`.

See `docs/tutorials/root-trexfitter-parity.md` for HTCondor job-array workflow, baseline registration, aggregation, and perf compare.

Manual refresh (GitHub Actions, external runner):
- Workflow: `.github/workflows/trex-baseline-refresh.yml` (workflow_dispatch only; requires a self-hosted runner with label `trex` and ROOT installed).

---

## RC-2: LLM Agent Surface Audit

Before each release, verify that **all** verticals in the codebase are exposed to AI agents.

### Checklist

1. **tools.py** — every vertical has at least one tool definition + execution dispatch.
   ```bash
   python -c "from nextstat.tools import get_tool_names; names = get_tool_names(); print(f'{len(names)} tools: {names}')"
   ```
   Expected: tool count matches the number documented in `AgenticTools.tsx`.

2. **llms-full.txt (EN)** — all verticals with real API examples, version header updated.
   ```bash
   grep -c "^##" public/llms-full.txt   # section count
   head -5 public/llms-full.txt         # verify version/date
   ```

3. **llms.txt (EN)** — summary mentions all verticals, version in "Key facts" matches.
   ```bash
   grep "Version:" public/llms.txt
   grep "Agentic tools:" public/llms.txt
   ```

4. **AgenticTools.tsx** — tool table row count matches `tools.py`.
   ```bash
   grep -c "nextstat_" src/pages/docs/AgenticTools.tsx
   ```

5. **Sync RU copies**:
   ```bash
   cp public/llms-full.txt /path/to/nextstat.ru/public/llms-full.txt
   cp public/llms.txt /path/to/nextstat.ru/public/llms.txt
   ```

### When to run

Every release that adds or changes a vertical, CLI subcommand, or Python API function.

---

## RC-3: Version Bump

Bump version in **all** of these locations (must match):

| File | Field |
|------|-------|
| `Cargo.toml` (workspace) | `version = "X.Y.Z"` |
| `bindings/ns-py/pyproject.toml` | `version = "X.Y.Z"` |
| `llms.txt` (EN + RU) | `- Version: X.Y.Z` |
| `llms-full.txt` (EN + RU) | header comment |

Verify:
```bash
grep '^version' Cargo.toml | head -1
grep '^version' bindings/ns-py/pyproject.toml
grep 'Version:' nextstat.io_web/ref/app/public/llms.txt
```

---

## RC-4: Changelog Update

Update `CHANGELOG.md` following the `/changelog` workflow:

- New verticals / tools added
- Breaking API changes
- Performance improvements (with numbers if available)
- Bug fixes
- Dependency updates

---

## RC-5: Website Rebuild + Deploy (EN + RU)

### Build

```bash
# EN
cd /path/to/nextstat.io_web/ref/app
npm run build
# Expect: "Prerendered N/N pages (0 failed)"

# RU
cd /path/to/nextstat.ru
npm run build
# Expect: "Prerendered N/N pages (0 failed)"
```

### Verify prerendered output

```bash
# New tools appear in prerendered HTML
grep -c "nextstat_glm_fit\|nextstat_bayesian_sample\|nextstat_survival_fit" \
  dist/docs/agentic-tools/index.html

# LLM files present in dist
wc -l dist/llms.txt dist/llms-full.txt
```

### Deploy

```bash
# EN — Railway
cd /path/to/nextstat.io_web/ref/app && railway up

# RU — Railway
cd /path/to/nextstat.ru && railway up
```

### Verify live

```bash
curl -s https://nextstat.io/llms.txt | head -10
curl -s https://nextstat.ru/llms.txt | head -10
```

---

## RC-6: PyPI Wheel Matrix & Publish (CI-automated)

PyPI publishing is **fully automated** via `.github/workflows/release.yml`. Pushing a `v*` tag triggers the entire pipeline. **Never run `maturin publish` manually.**

### Wheel Build Matrix (canonical)

| Runner | Target | Interpreter source | manylinux |
|--------|--------|-------------------|-----------|
| `ubuntu-latest` | `x86_64-unknown-linux-gnu` | manylinux Docker (pre-installed) | `2_17` |
| `ubuntu-latest` | `aarch64-unknown-linux-gnu` | manylinux Docker (pre-installed) | `2_17` |
| `macos-14` (ARM) | `aarch64-apple-darwin` | `setup-python` 3.11/3.12/3.13 | — |
| `macos-14` (ARM) | `x86_64-apple-darwin` | `setup-python` 3.11/3.12/3.13 | — |
| `windows-latest` | `x86_64-pc-windows-msvc` | `setup-python` 3.11/3.12/3.13 | — |

Expected output: **~15 wheels** (5 targets × 3 Python versions) + 1 sdist.

### Critical rules (lessons from v0.9.5 incident)

1. **Linux MUST use `manylinux: "2_17"`** — without it, maturin builds on the bare runner (glibc 2.35+) and produces wheels incompatible with most Linux distros, Docker images, and CI environments. The `manylinux_2_17` Docker container has all Python interpreters pre-installed AND old glibc.

2. **macOS Intel uses `macos-14` (cross-compile), NOT `macos-13`** — GitHub deprecated `macos-13` runners. Intel Mac wheels are cross-compiled from ARM via `--target x86_64-apple-darwin`. maturin-action handles this transparently.

3. **`setup-python` is required for macOS/Windows** — Linux manylinux Docker has interpreters pre-installed, but macOS/Windows runners only have one Python. Without `setup-python` installing 3.11/3.12/3.13, `--find-interpreter` finds only one → only one wheel per platform.

4. **Linux does NOT need `setup-python`** — the manylinux Docker container overrides the runner environment. Adding `setup-python` for Linux is harmless but unnecessary (the `if: runner.os != 'Linux'` guard skips it).

5. **Validation Pack must match interpreter** — the `find dist -name '*.whl'` glob can pick up a `cp314` wheel on a Python 3.12 runner. Always filter: `find dist -name '*cp312*.whl'`.

6. **CLI cross-compile needs `--target` flag** — when cross-compiling CLI binaries (e.g. Intel from ARM runner), `cargo build --release -p ns-cli` builds for the host arch. Must use `cargo build --release -p ns-cli --target ${{ matrix.target }}` and look for binary in `target/<target>/release/`.

### Pipeline flow

```
tag v* pushed
  → test (fmt + clippy + tests)
  → wheels (5 targets × 3 Pythons)
  → sdist
  → cli (3 targets)
  → whitepaper PDF
  → crates.io publish
  → validation pack (install cp312 wheel + pytest suite)
  → GitHub Release (attach all artifacts)
  → PyPI publish (Trusted Publisher, OIDC)
```

### Post-publish verification

```bash
# 1. Check PyPI has all wheels
pip install nextstat==X.Y.Z --dry-run --verbose 2>&1 | grep "Found link"

# 2. Verify installation on clean venv
python -m venv /tmp/ns-test && source /tmp/ns-test/bin/activate
pip install nextstat==X.Y.Z
python -c "import nextstat; print(nextstat.__version__)"
deactivate && rm -rf /tmp/ns-test

# 3. Check wheel tags on PyPI (expect ~15 wheels + 1 sdist)
pip index versions nextstat 2>&1 | head -5
```

### Post-mortem: v0.9.4 → v0.9.5 wheel fix

**Symptoms:** `pip install nextstat` failed on Linux x86_64 (most common platform) and macOS Intel. Users saw maturin trying to build from source, failing because sdist lacked complete Cargo workspace.

**Root causes:**
1. Linux x86_64 built on bare `ubuntu-latest` (glibc 2.35) → produced `manylinux_2_35` tag → incompatible with most systems
2. Only 1 Python found on bare runner → only 1 wheel (cp312) instead of 3
3. macOS Intel (`x86_64-apple-darwin`) target entirely missing from matrix
4. macOS/Windows had no `setup-python` → only 1 wheel per platform

**Fix:** Added `manylinux: "2_17"`, added Intel Mac target on `macos-14` (cross-compile), added `setup-python@v5` for non-Linux. Result: full wheel coverage across all platforms.

**Prevention:** This document. Any change to `release.yml` wheel matrix MUST be verified against this canonical table.

---

## RC-7: WASM Playground Build

### Build

```bash
bash scripts/playground_build_wasm.sh
```

This runs:
1. `RUSTFLAGS="-C target-feature=+reference-types" cargo build -p ns-wasm --target wasm32-unknown-unknown --profile release-wasm`
2. `wasm-bindgen` → `playground/pkg/`

The `reference-types` target feature is required for Rust 1.93+ so that wasm-bindgen's
externref-table implementation works correctly during the bindgen transform step.

### Critical: use `--profile release-wasm`, NOT `--release`

| Parameter | `--release` (WRONG) | `--profile release-wasm` (CORRECT) |
|-----------|--------------------|------------------------------------|
| opt-level | 3 (speed) | **z** (size) |
| LTO | thin | **fat** (better DCE) |
| strip | false (1.8 MB debug info!) | **true** |

The `release-wasm` profile is defined in the workspace `Cargo.toml`:

```toml
[profile.release-wasm]
inherits = "release"
opt-level = "z"
lto = "fat"
strip = true
```

Using `--release` leaves all debug sections (`.debug_str`, `.debug_info`, `.debug_line` = 1.8 MB)
and uses `opt-level = 3` which optimizes for speed, not size. Fat LTO is essential for
dead-code elimination of unused ns-inference modules (survival, econometrics, timeseries, etc.).

### wasm-opt post-processing (required)

```bash
wasm-opt -Oz --all-features \
  playground/pkg/ns_wasm_bg.wasm \
  -o playground/pkg/ns_wasm_bg.wasm
```

**`--all-features`** enables all WASM features (bulk-memory, nontrapping-float-to-int,
reference-types, etc.) which Rust 1.93+ emits. Do NOT use individual `--enable-*` flags —
`--all-features` produces smaller output (~874 KB vs ~910 KB with manual flags).

Without `--all-features`, wasm-opt fails with "unexpected false: memory.copy operations
require bulk memory operations".

### Binary size reference (v0.9.4, wasm-opt v125)

A/B tested by disabling HS3 branch in `ns-wasm/src/lib.rs`:

| Variant | raw cargo | post-bindgen | post-wasm-opt |
|---------|-----------|-------------|---------------|
| pyhf-only | 1,098,710 B | 845,081 B | **728,133 B** |
| hs3+pyhf | 1,283,604 B | 1,014,606 B | **874,862 B** |
| **HS3 delta** | +184,894 B | +169,525 B | **+146,729 B** |

The HS3 delta is code (parser/resolver/converter from `ns-translate`), not data.
By twiggy: retained items 5315 → 6008 (+693), `__wasm_bindgen_unstable` unchanged (87,981 B),
`data[0]` grew only ~5 KB — almost all growth is HS3 logic.

### Copy to website repos

```bash
for SITE in /path/to/nextstat.io_web/ref/app /path/to/nextstat.ru; do
  cp playground/pkg/ns_wasm_bg.wasm "$SITE/public/wasm/ns_wasm_bg.wasm"
  cp playground/pkg/ns_wasm.js      "$SITE/public/wasm/ns_wasm.js"
done
```

The `worker.js` dispatch table in `public/wasm/worker.js` must include handlers for any
new WASM exports (e.g. `bayes_sample(msg) { return wasm.run_bayes_sample(msg.inputJson); }`).

### Verify deployed size

```bash
curl -sI https://nextstat.io/wasm/ns_wasm_bg.wasm | grep content-length
curl -sI https://nextstat.ru/wasm/ns_wasm_bg.wasm | grep content-length
```

### Debugging binary size

```bash
cargo install twiggy

# Analyze the PRE-wasm-bindgen binary (the one cargo produces)
twiggy top target/wasm32-unknown-unknown/release-wasm/ns_wasm.wasm -n 30

# IMPORTANT: Do NOT analyze the post-wasm-bindgen binary (playground/pkg/ns_wasm_bg.wasm)
# wasm-bindgen injects JS glue that inflates size and twiggy will show
# phantom symbols (e.g. arrow_cast, parquet) that are NOT in the actual code.
```

### Feature isolation

`ns-translate` has optional Arrow/Parquet support behind the `arrow-io` feature.
Both `ns-wasm` and `ns-inference` depend on `ns-translate` with `default-features = false`,
which disables `root-io` and `arrow-io`. Verify no feature leaks:

```bash
cargo tree -p ns-wasm --target wasm32-unknown-unknown -e normal -i parquet 2>&1
# Expected: "package ID specification `parquet` did not match any packages"
```

### Supported WASM exports

| Function | Purpose |
|----------|---------|
| `run_asymptotic_upper_limits` | Brazil band CLs limits |
| `run_fit` | MLE fit (pyhf + HS3 auto-detect) |
| `run_profile_scan` | Profile likelihood scan |
| `run_hypotest` | Single-point CLs test |
| `run_glm` | GLM regression (linear/logistic/Poisson) |
| `run_bayes_sample` | NUTS/MAMS Bayesian sampling |
| `run_pop_pk_saem` | Population PK SAEM estimation |
| `workspace_from_histogram_rows_json` | Histogram → pyhf workspace |

---

## Full Pre-Release Checklist (summary)

| # | Gate | Command / Action | Blocking? |
|---|------|-----------------|-----------|
| RC-1 | Apex2 Baseline Gate | `make apex2-pre-release-gate` | YES |
| RC-2 | LLM Agent Surface Audit | Manual: tools.py ↔ llms*.txt ↔ AgenticTools.tsx | YES |
| RC-3 | Version Bump | Cargo.toml + pyproject.toml + llms*.txt | YES |
| RC-4 | Changelog | Update CHANGELOG.md | YES |
| RC-5 | Website Rebuild + Deploy | `npm run build` + `railway up` (EN + RU) | YES |
| RC-6 | PyPI Wheel Matrix & Publish | CI-automated (push `v*` tag) — verify wheel count post-publish | YES |
| RC-7 | WASM Playground Build | `make playground-build-wasm` + copy to sites | If playground changed |

**BMCP epic:** `eb8c74ad-5b16-4d95-b523-f9a495fac8da` (Pre-Release Checklist — Recurring Gate)
