---
title: "NextStat API Conventions"
status: active
---

# NextStat API Conventions

Internal binding rules for all Python, CLI, and WASM API work.

---

## 1. Function Naming

| Layer | Convention | Example |
|-------|-----------|---------|
| Python | `snake_case` | `profile_scan()`, `fit_toys()` |
| CLI | `kebab-case` | `profile-scan`, `fit-toys` |
| WASM | `camelCase` | `profileScan`, `fitToys` |
| Rust | `snake_case` | `profile_scan()`, `fit_toys()` |

**Rules:**

- NO model-type prefixes (`unbinned_`, `histfactory_`). Use runtime dispatch on model type.
- NO device suffixes (`_gpu`, `_metal`). Use `device` parameter.
- Abbreviations: `nll` (ok, established), `poi` (ok), `cls` (ok). Avoid introducing new abbreviations.
- Verb-first: `fit_toys`, `profile_scan`, `upper_limit` — not `toys_fit`, `scan_profile`.

---

## 2. Parameter Ordering

Strict priority order for all Python-exposed functions:

```
1. model           — always first, if applicable
2. test_value      — poi_test / mu_test (scalar being tested)
3. scan_values     — mu_values / scan (array of test points)
4. data arrays     — y first, then x, then metadata (entity_ids, cluster_ids, etc.)
5. dimensions      — p, k, m (integer counts)
6. *               — keyword-only separator
7. config kwargs   — n_toys, seed, max_iter, tol, method, etc.
8. device="cpu"    — always last config kwarg
9. return_* flags  — return_params, return_tail_probs, return_curve, etc.
```

**Examples:**

```python
ranking(model, *, device="cpu")
hypotest(poi_test, model, *, data=None, return_tail_probs=False)
profile_scan(model, mu_values, *, data=None, device="cpu", return_params=False)
fit_toys(model, params, *, n_toys=1000, seed=42, device="cpu")
panel_fe(y, x, entity_ids, p, *, time_ids=None, cluster_ids=None)
```

---

## 3. Device Handling

Single `device: str = "cpu"` parameter. Valid values: `"cpu"`, `"cuda"`, `"metal"`.

**Runtime capability checks:**

```python
nextstat.has_cuda()   # True if compiled with --features cuda AND GPU present
nextstat.has_metal()  # True if compiled with --features metal AND Apple Silicon
```

**Error pattern:**

```
Device 'cuda' not available. Build with --features cuda.
```

**Implementation:**

- Feature-gated at Rust compile time via `#[cfg(feature = "cuda")]`
- Runtime dispatch at Python level via `match device { ... }`
- CPU is always available — no feature gate needed

---

## 4. Model-Type Dispatch

Public functions accept any applicable model type and dispatch internally.

**Python-side (in `_core.pyi`):**

```python
def ranking(
    model: Union[HistFactoryModel, UnbinnedModel],
    *,
    device: str = "cpu",
) -> List[RankingEntry]: ...
```

**Rust-side (in `lib.rs`):**

```rust
fn ranking(py: Python, model: &Bound<PyAny>, device: &str) -> PyResult<...> {
    if let Ok(hf) = model.extract::<PyRef<PyHistFactoryModel>>() {
        // HistFactory ranking
    } else if let Ok(ub) = model.extract::<PyRef<PyUnbinnedModel>>() {
        // Unbinned ranking
    } else {
        Err(PyTypeError::new_err("Expected HistFactoryModel or UnbinnedModel"))
    }
}
```

Users never need to know which model type they have — just call `ranking()`.

---

## 5. Return Types

**Every function** returning structured data MUST have a TypedDict in `_core.pyi`.

**Naming conventions:**

| Pattern | TypedDict Name | Example |
|---------|---------------|---------|
| Single result | `{Function}Result` | `HypotestResult`, `ProfileScanResult` |
| List item | `{Item}Entry` | `RankingEntry`, `ProfileScanPoint` |
| Nested dict | Own TypedDict | `WorkspaceAuditModifiers` |

**Rules:**

- No `Dict[str, Any]` in return types — always specify the exact structure.
- No `Any` in return types — use Union or specific types.
- Lists of dicts use `List[{Item}Entry]`.

---

## 6. Data Placement

| Where | What | Example |
|-------|------|---------|
| Constructor | Static data that defines the model | `LinearRegressionModel(x, y)` |
| Function | Variable data that changes between calls | `hypotest(poi_test, model)` |

**Rule:** If data changes between calls on the same model, it's a function parameter. If it's fixed for the model's lifetime, it's a constructor parameter.

**`data` override for HistFactory:**

```python
fit(model, data=[...])           # Override observed data for this call
profile_scan(model, mu_values, data=[...])
```

This is the one exception: HistFactory models carry observed data in the workspace, but users can override it per-call.

---

## 7. New Function Checklist

When adding a new Python-exposed function:

1. Define TypedDict return type in `_core.pyi`
2. Add `#[pyfunction]` in `lib.rs` with proper parameter ordering (Section 2)
3. Add `_get("name")` in `__init__.py`
4. Add to `__all__` in `__init__.py`
5. If accepts model: accept `Union[...]` for all applicable model types
6. If compute-heavy: use `py.detach()` for GIL release
7. If GPU-applicable: add `device: str = "cpu"` as last config kwarg
8. Update `docs/references/python-api.md`
9. Update `CHANGELOG.md` (user-facing features only)

---

## 8. Sampling API

Unified dispatcher: `sample(model, method="nuts"|"mams"|"laps", ...)`.

Individual `sample_nuts()`, `sample_mams()`, `sample_laps()` exist as internal aliases but are NOT in `__all__`. Users should use `sample()`.

---

## 9. Naming Exceptions

- `hypotest(poi_test, model)` — `poi_test` before `model` follows HEP convention (pyhf compatibility). This is intentional.
- `cls_curve()` — visualization function, separate from `profile_scan()`. Keep as-is.
- `from_pyhf()`, `from_histfactory_xml()` — factory functions, not inference. Keep as-is.
