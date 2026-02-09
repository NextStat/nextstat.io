# TREx ReadFrom=HIST Semantics (NextStat)

NextStat supports a **HIST wrapper** over an existing HistFactory export directory. This is intentionally a *filtering* layer (not a full TREx reimplementation).

## Inputs

Minimal required config keys:

- `ReadFrom: HIST`
- `HistoPath: <dir>`: a directory tree containing exactly one `combination.xml` (or pass an explicit `CombinationXml:`).

The HistFactory export dir is expected to contain:

- `combination.xml`
- `channels/*.xml`
- referenced ROOT histograms (commonly `data.root`)

## Region and sample masking

In HIST mode, many TREx `.config` files use `Region:` and nested `Sample:` blocks to express channel/sample selection.

NextStat interpretation:

- `Region:` blocks without `Variable/Binning` are treated as **channel include-lists**.
- `Sample:` blocks without `File/Path` are treated as **sample include-lists** scoped by `Regions: ...`.
- If a config requests a channel that does not exist in the imported HistFactory workspace, NextStat errors.
- If a config requests a sample that does not exist in a selected channel, NextStat errors.
- If filtering removes all samples from a selected channel, NextStat errors (explicit selection should not silently produce an empty channel).

This behavior is covered by Rust tests in `crates/ns-translate/src/trex/mod.rs` using realistic committed export dirs under `tests/baselines/trex/`.

## Known divergences (current)

- No attempt is made to infer or rebuild HistFactory objects from TREx rules; the export dir is authoritative.
- Any TREx settings that would normally affect histogram production, smoothing, or pruning are out of scope for HIST wrapper (those are preprocessing steps in the export, not the wrapper).

