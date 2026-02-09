#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;
use std::process::Command;

#[test]
#[ignore]
fn wasm32_unknown_unknown_builds_with_parquet_zstd() {
    // This is a compile-only smoke test intended for CI / guardrail runs.
    //
    // It verifies that `ns-translate` + `parquet/zstd` builds for `wasm32-unknown-unknown`,
    // relying on the workspace `zstd` shim (patched to `crates/zstd-shim`) backed by `ns-zstd`.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("expected ns-translate to be at crates/ns-translate");

    let status = Command::new("cargo")
        .current_dir(workspace_root)
        .args([
            "build",
            "-p",
            "ns-translate",
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
