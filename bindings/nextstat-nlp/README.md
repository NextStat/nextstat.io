# nextstat-nlp 0.2.1

`nextstat-nlp` is an optional, Python-only plugin that turns unstructured clinical text
into strict, validated tabular inputs for NextStat: survival datasets, prior candidates,
and PK dosing regimens.

This package is intentionally separate from `nextstat`:

- `nextstat` stays a minimal inference engine (Rust + PyO3).
- `nextstat-nlp` carries optional NLP dependencies (GLiNER2 / ONNX Runtime).

## Install

```bash
# Core (heuristic backend only — zero ML dependencies)
pip install -e bindings/nextstat-nlp

# With GLiNER2 Torch backend
pip install -e 'bindings/nextstat-nlp[gliner2]'

# With GLiNER2 ONNX backend
pip install -e 'bindings/nextstat-nlp[onnx]'
```

## Pipelines

### Survival extraction

```python
from nextstat_nlp import extract_survival_records

texts = [
    "Subject 001: progressed at day 84. Age 63. Dose 20 mg daily.",
    "Subject 002: lost to follow-up at 12 weeks. Age 58. Dose 10 mg daily.",
]

ds = extract_survival_records(texts, backend="heuristic")
time, event, X, features = ds.to_design_matrix()
# time    = [84.0, 84.0]
# event   = [1, 0]
# X       = [[63.0, 20.0], [58.0, 10.0]]
# features = ['age', 'dose']
```

Supported time formats: `"84 days"`, `"day 84"`, `"12 weeks"`, `"6 months"`, `"2 years"`, `"48 hours"`, decimals.

Covariates extracted: `age`, `dose`, `sex`, `stage`, `ecog_ps`, `weight`.

### Prior candidate extraction

```python
from nextstat_nlp.priors import extract_prior_candidates

texts = [
    "CL was estimated with a lognormal prior, mean 3.5 SD 1.2 L/h.",
    "V1 followed a normal distribution, mean 45.0 SD 12.0 L.",
]

bundle = extract_prior_candidates(texts, backend="heuristic")
for c in bundle.candidates:
    print(f"{c.param_name}: {c.dist}({c.location}, {c.scale}) {c.units}  [{c.status}]")
# CL: lognormal(3.5, 1.2) L/h  [candidate]
# V1: normal(45.0, 12.0) L  [candidate]
```

All candidates are `status="candidate"` — requires explicit human approval.

### Regimen extraction

```python
from nextstat_nlp.regimens import extract_regimens

texts = [
    "Patient 101: 500 mg IV infusion, QD dosing.",
    "Patient 102: 250 mg oral twice daily.",
]

table = extract_regimens(texts, backend="heuristic")
for r in table.records:
    print(f"{r.subject_id}: {r.dose} {r.amount_units} {r.route} {r.frequency}")
# 101: 500.0 mg IV QD
# 102: 250.0 mg oral BID
```

Notes:
- `RegimenRecord.duration` is *course duration* (e.g. "for 14 days").
- `RegimenRecord.infusion_duration` is *infusion duration* (e.g. "over 2 hours" -> `2/24` days).

## Backends

| Backend | Install | Model download | Use case |
|---------|---------|----------------|----------|
| `heuristic` | None | No | CI, testing, regex-only |
| `gliner2` | `pip install gliner2` | Yes (~300MB) | Production NER |
| `onnx` | `pip install gliner2-onnx onnxruntime` | Yes | CPU-optimised production |
| `mlx` | Swift CLI | Yes | Apple Silicon (MLX/Metal) |

### macOS acceleration notes

- `gliner2` (Torch): you can request Metal (MPS) by passing `device="mps"` to the backend factory. This is best-effort: it depends on your `torch` build and whether GLiNER2 exposes a movable module.
- `onnx`: you can pass ONNX Runtime providers explicitly. On macOS, if your `onnxruntime` build includes CoreML EP, try:
  `providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]`.
- `mlx`: GLiNER2-on-MLX via an optional Swift CLI.

  1. Build the CLI:
     ```bash
     cd bindings/nextstat-nlp/tools/gliner2_mlx_cli
     # Optional: set XDG_CACHE_HOME if your environment restricts writing to ~/Library/Caches
     XDG_CACHE_HOME="$PWD/../../../.tmp/swiftpm_cache" swift build -c release
     ```
  2. Export the CLI path:
     ```bash
     export NEXTSTAT_GLINER2_MLX_CLI="$PWD/.build/release/gliner2_mlx_cli"
     ```
  3. Build `mlx.metallib` next to the CLI (required by MLX):
     ```bash
     ./build_mlx_metallib.sh
     ```
     If `xcrun metal` is missing in your Xcode install, you can use:
     ```bash
     ./bootstrap_metallib.sh
     ```
  4. Run any pipeline with `backend="mlx"`.

  Notes:
  - The default MLX backend model is `fastino/gliner2-base-v1` (public).
  - Some MacPaw model repos are gated on Hugging Face (HTTP 401 without auth). Set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`
    to enable access in the Swift Hub client.

This plugin intentionally keeps all accelerators optional and pluggable.

## Schemas

All output types are frozen dataclasses with SHA-256 text hashes for audit:

- `SurvivalRecord`, `SurvivalDataset` — time-to-event data
- `PriorCandidate`, `PriorBundle` — informative prior candidates
- `RegimenRecord`, `RegimenTable` — dosing regimen records
- `ExtractedSpan` — raw NER span with label, offsets, confidence
- `ProvenanceBundle` — model version, runtime env, input hashes, timestamp

## Testing

```bash
cd bindings/nextstat-nlp
python -m pytest tests/ -v  # 59 tests, heuristic backend, ~0.06s
```

## Benchmarks

```bash
python benchmarks/bench_extraction.py --backend heuristic --n-iter 100
# On macOS (Apple Silicon), MLX/Metal backend:
#   export NEXTSTAT_GLINER2_MLX_CLI=/path/to/gliner2_mlx_cli
#   python benchmarks/bench_extraction.py --backend mlx --n-iter 50
```

## End-to-End Workflow Demo (to NextStat)

This demo fetches small public snippets, extracts regimens + survival records via
`nextstat-nlp`, then runs a tiny NextStat fit (CoxPH + 1-subject NLME synthetic):

```bash
python tools/demo_pharma_to_nextstat.py --backend onnx --out-dir /tmp/ns_nlp_demo
```

## Reproducible Workflow Matrix (3x)

Fetch internet snippets once, then re-run the pipelines 3 times offline to verify
bit-exact determinism of the extraction outputs:

```bash
python tools/run_workflow_matrix.py --backends heuristic onnx --n-repeats 3 --out-dir /tmp/ns_nlp_matrix
cat /tmp/ns_nlp_matrix/matrix_summary.json
```

If you don't have `nextstat` installed, you can still validate the extraction and
determinism by skipping the downstream consumer smoke:

```bash
python tools/run_workflow_matrix.py --backends heuristic onnx --n-repeats 3 --skip-nextstat-demo
```

## Exporting Regimens to NextStat

If you want to pass extracted regimens into `nextstat._core.nlme_foce`, you can convert
records into the `_core` `regimens=[{events:[...]}]` shape:

```python
from nextstat_nlp.regimens import extract_regimens, to_nextstat_regimens

table = extract_regimens(["1000 mg IV infusion over 2 hours, QD dosing for 14 days."], backend="onnx")
regimens_min = to_nextstat_regimens(table.records)
regimens_expanded = to_nextstat_regimens(table.records, expand_frequency=True, default_course_days=14.0)
```

## Architecture

```
text → Backend (GLiNER2/ONNX/heuristic) → EntitySpan[]
     → _parsers (deterministic regex)    → time, event, covariates
     → Pipeline (survival/priors/regimens) → validated frozen dataclass
     → ProvenanceBundle                    → JSON audit trail
```
