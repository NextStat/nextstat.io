---
title: "WASM Playground Reference"
status: stable
last_updated: 2026-03-19
---

# WASM Playground Reference

The NextStat WASM surface is the browser playground built from `bindings/ns-wasm/`
into `playground/pkg/`.

**Status:** Stable for the documented source-build public surface.

## Stable source-build boundary

The current stable WASM surface covers:

- browser-side HistFactory loading from pyhf JSON, HS3, and simplified-likelihood JSON
- asymptotic CLs / Brazil-band evaluation
- profile scans and maximum-likelihood fits
- the documented GLM regression playground modes
- deterministic source build into `playground/pkg/`

Out of scope for the current WASM stable boundary:

- the full native CLI / Python surface
- large workspaces beyond the documented browser safety limits
- GPU execution
- heavy production workflows that should use the native engine

## Build

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.108

make playground-build-wasm
```

This writes the canonical browser bundle to:

- `playground/pkg/ns_wasm.js`
- `playground/pkg/ns_wasm_bg.wasm`

## Run locally

```bash
make playground-serve
```

Then open `http://localhost:8000/`.

## Verification

Local source-build verification:

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.108
bash scripts/playground_build_wasm.sh
```

CI coverage lives in `.github/workflows/rust-tests.yml`:

- `wasm-smoke` verifies the wasm-target build path for `ns-translate` / `ns-unbinned`
- `wasm-playground-build` verifies the public playground bundle build

## Public boundary

The WASM playground is a stable public interface for the documented browser
subset. It is not a claim that the entire native NextStat engine is available
inside the browser.
