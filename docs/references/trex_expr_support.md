# TREx/ROOT Expression Compatibility (NTUP)

Status: active (incremental; contract is TDD-backed)

This doc tracks what NextStat’s Rust expression engine supports for TREx-style `Selection`, `Weight`, and `Variable` expressions (NTUP mode).

Source of truth:
- Parser/evaluator: `crates/ns-root/src/expr.rs`
- TREx config importer usage: `crates/ns-translate/src/ntuple/processor.rs`

## Supported (today)

### Literals
- Floating-point literals (incl. scientific notation): `1`, `1.5`, `1.5e2`, `3.0E-1`

### Variables (branch names)
- ASCII identifiers, including dots: `pt`, `weight_mc`, `jet.pt`
- Namespace-qualified identifiers are accepted for function names: `TMath::Abs(...)` (see below)

### Operators
- Arithmetic: `+`, `-`, `*`, `/`
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Boolean: `&&`, `||`, `!`
- Ternary: `cond ? a : b` (right-associative)

### Truthiness (ROOT/TTreeFormula)
- Numeric conditions are truthy iff they are **non-zero** (including negatives and NaN).

### Indexing syntax (parsing)
- `branch[idx]` is accepted syntactically and rewritten into a required scalar “virtual” branch name `branch[idx]`.
- Reading `branch[idx]` from ROOT TTrees is supported via `RootFile::branch_data("name[idx]")`:
  - **Jagged/variable-length leaflist** branches (per-basket entry-offset table): out-of-range → `0.0`.
  - **Fixed-length array** branches (flat storage with `N = entries * len`): out-of-range → `0.0`.
  - **Scalar** branches: `name[0]` is allowed and behaves like `name`; `name[k>0]` is a read-time error.

### Dynamic indexing (runtime)
- Expressions like `branch[expr]` (e.g. `jet_pt[njet - 1]`) compile into a jagged-load instruction.
- The ntuple histogrammer automatically reads the base branch as jagged data via `RootFile::branch_data_jagged(...)` when needed.
- Index semantics follow ROOT/TTreeFormula numeric convention:
  - out-of-range, negative, or non-finite index → `0.0`.

### Functions (case-insensitive, namespace-insensitive)
The engine matches functions by:
1) stripping `...::` namespaces, then
2) lowercasing, then
3) mapping aliases.

Unary:
- `abs(x)` (aliases: `fabs(x)`, `TMath::Abs(x)`)
- `sqrt(x)`
- `log(x)` (natural log)
- `log10(x)` (aliases: `TMath::Log10(x)`)
- `exp(x)`
- `sin(x)` (aliases: `TMath::Sin(x)`)
- `cos(x)` (aliases: `TMath::Cos(x)`)

Binary:
- `pow(x, y)` (aliases: `power(x, y)`, `TMath::Power(x, y)`)
- `atan2(y, x)` (aliases: `TMath::ATan2(y, x)`)
- `min(a, b)`
- `max(a, b)`

### Evaluation modes
- `eval_row`: scalar evaluation for one event (used in tests/diagnostics)
- `eval_bulk`: columnar evaluation with a vectorized fast path (falls back to row-wise when control-flow is present)

## Limitations / known gaps

### Vector branches (real `jet_pt[0]`)
Goal: allow expressions like `jet_pt[0]` when `jet_pt` is stored as a vector branch in a ROOT TTree.

Current status:
- Implemented for:
  - uproot-style jagged leaflist branches with per-basket entry-offset tables,
  - fixed-length arrays (heuristic based on decoded length vs entries).
- Implemented for unsplit `TBranchElement` / STL `std::vector<T>` branches in the common ROOT-written layout (ROOT streamer):
  - per entry: `[bytecount+version][u32 len][len elements]` (big-endian),
  - entry boundaries come from the per-basket entry-offset table (`fEntryOffsetLen > 0`),
  - ROOT’s on-disk convention `entry_offsets[last] == 0` is treated as a sentinel for “end at fLast”.
- Best-effort fallback for non-streamer numeric layout (seen in some writers):
  - per entry: `u32 len` (big-endian) followed by `len` elements,
  - element type inferred from the branch leaf type (with fallback probes).
  This is sufficient for typical `branch[idx]` usage in selections/weights but is not a complete ROOT streamer implementation (e.g. nested vectors, `vector<bool>`, custom classes).

BMCP:
- Epic: `TREx Replacement: Expression Compatibility (TTreeFormula subset + vector branches)` (`53b3d3fc-ef1f-42b2-b067-b4df90c1044e`)
- Task: `Vector-branch indexing: materialize branch[idx] scalar columns` (`29a35e89-f3e8-4af7-aa46-9db445fcefd8`) — DONE (leaflist/jagged via offsets)

### TTreeFormula / ROOT-specific constructs
Not implemented (and we should only add them based on real config corpus):
- `Sum$`, `Max$`, `Min$` over vector branches
- `TMath::Pi()` and other constants
- String ops, regex, etc.

## How to extend safely (TDD)

1) Add a unit test to `crates/ns-root/src/expr.rs` that compiles + evaluates the new construct.

## Tooling: expression coverage report

To scan a TREx config and see which expressions compile (and which branches they reference), run:

```bash
nextstat import trex-config --config trex.txt --base-dir . --output workspace.json --expr-coverage-json expr_coverage.json
```

This report is **compile-only**: it does not validate that branches exist in a particular ROOT file.
2) Keep the vectorized bulk path deterministic (no data-dependent branching unless it falls back to row-wise).
3) Only then wire it into NTUP ingestion (which already compiles `Selection`/`Weight`/`Variable` via `CompiledExpr::compile`).
