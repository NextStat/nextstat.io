# HistFactory export fixtures (TREx-style)

This directory contains committed HistFactory export directories used for TREx replacement
parity/robustness testing.

Each export directory includes:
- `combination.xml`
- `HistFactorySchema.dtd`
- `channels/*.xml`
- `data.root` (histograms referenced by the XMLs)

These fixtures are intended to exercise “real export dir” path semantics beyond the smaller
unit fixtures under `tests/fixtures/pyhf_*` and `tests/fixtures/histfactory/`.

