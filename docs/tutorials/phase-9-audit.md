---
title: "Phase 9: Audit Trail Hooks (OSS Baseline)"
status: draft
---

# Phase 9: Audit Trail Hooks (OSS Baseline)

This tutorial documents the **OSS baseline** for compliance-related workflows:
local run metadata logging and reproducible run bundles.

It is intentionally limited. Enterprise audit trail features (append-only logs,
e-signatures, approvals, model registry) are out of scope for OSS and belong to
Pro modules (see `docs/internal/legal/open-core-boundaries.md`).

## What is implemented

### 1) CLI run bundles

The CLI supports a global `--bundle` flag that writes an immutable bundle
containing inputs, outputs, hashes, and a manifest. See:
`docs/tutorials/phase-3.1-frequentist.md`.

Note: bundle `meta.json` includes `created_unix_ms` by default. When a command supports
`--deterministic`, timestamps are normalized to `0` for reproducible artifacts.

### 2) Python run bundles

Python can write the same bundle layout via:

```python
import json
from pathlib import Path

from nextstat import audit

inp = Path("tests/fixtures/simple_workspace.json")
out = {"example": "result"}

audit.write_bundle(
  "tmp/run_bundle_py",
  command="example",
  args={"note": "python-run"},
  input_path=inp,
  output_value=out,
)
```

## Pro hooks (identified)

To reach regulated audit trail requirements (e.g. 21 CFR Part 11), typical needs are:

- append-only event log with tamper evidence (WORM storage)
- identity/authn, approvals, and e-signatures
- access control and retention policies
- validation packs and controlled execution environments

OSS provides the run bundle primitives needed to feed those systems, but does not
implement them.
