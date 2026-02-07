use std::path::PathBuf;
use std::process::{Command, Output};

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

#[test]
fn config_schema_default_is_analysis_spec_v0_json() {
    let out = run(&["config", "schema"]);
    assert!(
        out.status.success(),
        "config schema should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("schema output should be valid JSON");
    assert_eq!(v["$id"], "https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json");
    assert_eq!(v["$schema"], "https://json-schema.org/draft/2020-12/schema");
}

#[test]
fn config_schema_can_emit_report_schema() {
    let out = run(&["config", "schema", "--name", "report_yields_v0"]);
    assert!(
        out.status.success(),
        "config schema --name report_yields_v0 should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("schema output should be valid JSON");
    assert_eq!(v["$schema"], "https://json-schema.org/draft/2020-12/schema");
}
