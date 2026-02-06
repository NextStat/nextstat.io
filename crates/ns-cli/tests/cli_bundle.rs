use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(name)
}

fn tmp_dir_path(suffix: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("nextstat_bundle_{}_{}_{}", std::process::id(), nanos, suffix));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(64);
    for b in out {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

#[test]
fn fit_writes_repro_bundle_with_hashes() {
    let input = fixture_path("simple_workspace.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let bundle = tmp_dir_path("fit");
    let out = run(&[
        "--bundle",
        bundle.to_string_lossy().as_ref(),
        "fit",
        "--input",
        input.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(
        out.status.success(),
        "fit should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Ensure stdout stays pure JSON.
    let _v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be JSON");

    let meta_path = bundle.join("meta.json");
    let manifest_path = bundle.join("manifest.json");
    let input_copy = bundle.join("inputs/input.json");
    let spec_path = bundle.join("inputs/model_spec.json");
    let data_path = bundle.join("inputs/data.json");
    let result_path = bundle.join("outputs/result.json");

    for p in [&meta_path, &manifest_path, &input_copy, &spec_path, &data_path, &result_path] {
        assert!(p.exists(), "expected bundle file: {}", p.display());
    }

    // meta.json should reference the correct hashes.
    let meta_bytes = std::fs::read(&meta_path).unwrap();
    let meta: serde_json::Value = serde_json::from_slice(&meta_bytes).unwrap();

    let input_bytes = std::fs::read(&input).unwrap();
    let input_sha = sha256_hex(&input_bytes);
    assert_eq!(
        meta.pointer("/input/input_sha256").and_then(|v| v.as_str()),
        Some(input_sha.as_str())
    );

    let spec_bytes = std::fs::read(&spec_path).unwrap();
    let data_bytes = std::fs::read(&data_path).unwrap();
    assert_eq!(
        meta.pointer("/input/model_spec_sha256").and_then(|v| v.as_str()),
        Some(sha256_hex(&spec_bytes).as_str())
    );
    assert_eq!(
        meta.pointer("/input/data_sha256").and_then(|v| v.as_str()),
        Some(sha256_hex(&data_bytes).as_str())
    );

    // manifest.json hashes must match file contents.
    let manifest_bytes = std::fs::read(&manifest_path).unwrap();
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    let files = manifest.get("files").and_then(|v| v.as_array()).expect("manifest.files should be array");
    assert!(!files.is_empty(), "manifest.files should be non-empty");

    for f in files {
        let rel = f.get("path").and_then(|v| v.as_str()).expect("file.path should be string");
        let want = f.get("sha256").and_then(|v| v.as_str()).expect("file.sha256 should be string");
        let p = bundle.join(rel);
        let got = sha256_hex(&std::fs::read(&p).unwrap());
        assert_eq!(want, got, "sha256 mismatch for {}", rel);
    }

    let _ = std::fs::remove_dir_all(&bundle);
}

#[test]
fn bundle_errors_on_non_empty_dir() {
    let input = fixture_path("simple_workspace.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let bundle = tmp_dir_path("non_empty");
    std::fs::create_dir_all(&bundle).unwrap();
    std::fs::write(bundle.join("junk.txt"), "x").unwrap();

    let out = run(&[
        "--bundle",
        bundle.to_string_lossy().as_ref(),
        "fit",
        "--input",
        input.to_string_lossy().as_ref(),
        "--threads",
        "1",
    ]);
    assert!(!out.status.success(), "expected failure for non-empty bundle dir");

    let _ = std::fs::remove_dir_all(&bundle);
}

