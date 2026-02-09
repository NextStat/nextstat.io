# NextStat Playground

Static browser demo for running NextStat asymptotic CLs (q~tilde~) directly on a `pyhf`-style `workspace.json`.

## Build (WASM)

Prereqs:
- Rust toolchain (see `rust-toolchain.toml`)
- `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- `wasm-bindgen` CLI: `cargo install wasm-bindgen-cli --version 0.2.108` (version should match `Cargo.lock`)

Build the JS + WASM bundle into `playground/pkg/`:

From the repo root:

```bash
make playground-build-wasm
```

## Run locally

You must serve the files over HTTP (workers/modules don’t work from `file://`).

```bash
make playground-serve
```

Open `http://localhost:8000/`.

## Deploy (GitHub Pages)

GitHub Pages can serve the `playground/` directory, but you must ensure `playground/pkg/` is built and present in the published artifact.
Use either:
- commit `playground/pkg/` (quickest for a demo), or
- add a Pages workflow that runs `scripts/playground_build_wasm.sh` and publishes `playground/`.
