# HEP Suite (Minimal Seed)

This is a minimal HEP benchmark seed: **NLL parity + timing** for a tiny HistFactory workspace.

Inputs:
- `datasets/simple_workspace.json` (tiny, in-repo fixture)

Outputs:
- A single JSON result following `manifests/schema/benchmark_result_v1.schema.json`

Run:

```bash
python run.py --deterministic --out ../../out/hep_simple_nll.json
```

