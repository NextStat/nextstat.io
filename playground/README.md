# NextStat Playground

Static browser demo for running NextStat asymptotic CLs (q~tilde~) directly on a `pyhf`-style `workspace.json`.

## Build (WASM)

Prereqs:
- Rust toolchain (see `rust-toolchain.toml`)
- `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- `wasm-bindgen` CLI (version should match the `wasm-bindgen` crate in `Cargo.lock`)

Build the JS + WASM bundle into `playground/pkg/`:

```bash
bash scripts/playground_build_wasm.sh
```

## Run locally

You must serve the files over HTTP (workers/modules don’t work from `file://`).

```bash
bash scripts/playground_serve.sh 8000
```

Open `http://localhost:8000/`.

## Deploy (GitHub Pages)

GitHub Pages can serve the `playground/` directory, but you must ensure `playground/pkg/` is built and present in the published artifact.
Use either:
- commit `playground/pkg/` (quickest for a demo), or
- add a Pages workflow that runs `scripts/playground_build_wasm.sh` and publishes `playground/`.

