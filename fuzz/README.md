# Fuzzing (ns-root decompression)

This directory contains `cargo-fuzz` harnesses for hardening `ns-root` against malformed inputs.

## Setup

```bash
cargo install cargo-fuzz
```

You also need a working clang toolchain (required by libFuzzer).

## Run

```bash
cargo fuzz run ns_root_decompress -- -max_total_time=60
```

Goal: no panics/UB. Errors must be returned as `Result::Err`.
