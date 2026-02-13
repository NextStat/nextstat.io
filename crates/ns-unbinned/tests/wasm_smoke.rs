#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::process::Command;

#[test]
#[ignore]
fn wasm32_unknown_unknown_builds_with_parquet_zstd() {
    // Compile-only smoke test for the wasm target.
    //
    // This validates the Parquet+Zstd adapter path on wasm:
    // - `parquet/zstd` is enabled via `arrow-io-zstd`
    // - `zstd` is patched to `crates/zstd-shim` in the workspace
    // - `getrandom` backend is configured for wasm_js
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("expected ns-unbinned to be at crates/ns-unbinned");

    let existing_rustflags = std::env::var("RUSTFLAGS").unwrap_or_default();
    let wasm_backend_flag = r#"--cfg getrandom_backend="wasm_js""#;
    let rustflags = if existing_rustflags.trim().is_empty() {
        wasm_backend_flag.to_string()
    } else if existing_rustflags.contains("getrandom_backend") {
        existing_rustflags
    } else {
        format!("{existing_rustflags} {wasm_backend_flag}")
    };

    let status = Command::new("cargo")
        .current_dir(workspace_root)
        .env("RUSTFLAGS", rustflags)
        .args([
            "build",
            "-p",
            "ns-unbinned",
            "--target",
            "wasm32-unknown-unknown",
            "--no-default-features",
            "--features",
            "arrow-io-zstd",
        ])
        .status()
        .expect("failed to spawn cargo");

    assert!(
        status.success(),
        "wasm smoke build failed (try `rustup target add wasm32-unknown-unknown`)"
    );
}
