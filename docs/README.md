# NextStat Docs Index

This repo uses plain Markdown docs. Start here and then jump into the track that matches what you're doing.

## Roadmap

- Project roadmap, milestones, and known limitations: `docs/ROADMAP.md`

## Start Here

- Tutorials index (end-to-end workflows): `docs/tutorials/README.md`
- Quickstarts (10 minutes to result): `docs/quickstarts/README.md`
- Adoption playbook (HEP routes A/B/C): `docs/guides/README.md`
- Python reference: `docs/references/python-api.md`
- Python packaging (wheels/extras): `docs/references/python-packaging.md`
- Arrow / Parquet I/O (histogram tables): `docs/references/arrow-parquet-io.md`
- CLI reference: `docs/references/cli.md`
- Rust reference: `docs/references/rust-api.md`
- Terminology and style guide: `docs/references/terminology.md`
- Glossary (cross-domain term definitions): `docs/references/glossary.md`
- RNTuple effort estimate (minimal/converter/full): `docs/references/rntuple-minimal-reader-estimate.md`
- RNTuple compatibility matrix (verified rows + CI gates): `docs/references/rntuple-compatibility-matrix.md`
- RNTuple rollout/migration notes (v1 scope + limits): `docs/references/rntuple-rollout-v1.md`
- RNTuple benchmark note (`nextstat-bench`, `ns-root` vs ROOT): `docs/benchmarks/rntuple-nextstat-bench-2026-02-16.md`
- RNTuple mixed-layout verification addendum (`2,000,000` entries, release perf-gate): `docs/benchmarks/rntuple-nextstat-bench-2026-02-16.md`
- RNTuple reproducible comparison harness (`make rntuple-root-vs-nsroot`): `scripts/benchmarks/run_rntuple_root_vs_nsroot.sh`

## Demos

- Physics Assistant demo (ROOT -> anomaly scan -> p-values + plots): `docs/demos/physics-assistant.md`

## Benchmarks and Trust Artifacts

- Benchmarks hub: `docs/benchmarks.md`
- Public benchmark suites (seed repo): `benchmarks/nextstat-public-benchmarks/`
- Validation report (JSON/PDF contract): `docs/references/validation-report.md`

## Tools and Server (LLM/Agent Integration)

- Tool API contract: `docs/references/tool-api.md`
- Server API (`/v1/tools/execute`, etc.): `docs/references/server-api.md`
- Plot artifacts (JSON): `docs/references/plot-artifacts.md`

## Neural Density Estimation

- Neural PDFs guide (FlowPdf, DcrSurrogate, training, ONNX): `docs/neural-density-estimation.md`
- Differentiable HistFactory (binned-likelihood workspace) layer for PyTorch: `docs/differentiable-layer.md`

## R Bindings

- R package reference (experimental): `docs/references/r-bindings.md`

## Arrow / Parquet

- Binned histogram Parquet schema (v2, with modifiers): `docs/references/binned-parquet-schema.md`
- Unbinned event-level Parquet schema (v1): `docs/references/unbinned-parquet-schema.md`

## Architecture Decisions (RFC/ADR)

- ADR-0001 RNTuple Support Policy: `docs/rfcs/rntuple-support-policy.md`

## HPC / Cluster Deployment

- HTCondor & HPC cluster guide: `docs/guides/htcondor-hpc.md`
- HTCondor examples (.sub files, DAGMan): `docs/examples/htcondor/`
- Apptainer/Singularity containers: `containers/`

## GPU Support

- GPU contract and backend matrix: `docs/gpu-contract.md`

## Personas

These are navigation pages that map NextStat concepts and docs to non-HEP workflows.

- Data Scientists: `docs/personas/data-scientists.md`
- Quants: `docs/personas/quants.md`
- Biologists / Pharmacometricians: `docs/personas/biologists.md`

## Русскоязычная документация

- Индекс (RU): `docs/ru/README.md`
- Глоссарий (RU): `docs/ru/references/glossary.md`
