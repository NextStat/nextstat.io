use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
    // crates/ns-cli -> repo root
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn tmp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("nextstat_cli_{}_{}_{}", std::process::id(), nanos, name));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

#[test]
fn trex_import_config_writes_analysis_spec_and_mapping_report() {
    let root = repo_root();
    let cfg = root.join("tests/fixtures/trex_config/minimal_tutorial.config");
    assert!(cfg.exists(), "missing fixture: {}", cfg.display());

    let out_dir = tmp_dir("trex_import_config");
    std::fs::create_dir_all(&out_dir).unwrap();
    let out = out_dir.join("analysis.yaml");
    let report = out_dir.join("mapping.json");

    let outp = run(&[
        "trex",
        "import-config",
        "--config",
        cfg.to_string_lossy().as_ref(),
        "--out",
        out.to_string_lossy().as_ref(),
        "--report",
        report.to_string_lossy().as_ref(),
        "--overwrite",
    ]);
    assert!(
        outp.status.success(),
        "trex import-config should succeed, stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&outp.stdout),
        String::from_utf8_lossy(&outp.stderr)
    );

    assert!(out.exists(), "missing analysis spec: {}", out.display());
    assert!(report.exists(), "missing mapping report: {}", report.display());

    let yaml = std::fs::read_to_string(&out).unwrap();
    assert!(yaml.contains("schema_version:"), "analysis spec missing schema_version");
    assert!(yaml.contains("trex_analysis_spec_v0"), "analysis spec wrong schema version");

    let rep_bytes = std::fs::read(&report).unwrap();
    let _rep: serde_json::Value = serde_json::from_slice(&rep_bytes)
        .unwrap_or_else(|e| panic!("invalid JSON mapping report {}: {}", report.display(), e));

    let _ = std::fs::remove_dir_all(&out_dir);
}

