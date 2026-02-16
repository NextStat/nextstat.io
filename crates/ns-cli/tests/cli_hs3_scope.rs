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

fn tmp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    std::env::temp_dir().join(format!(
        "nextstat_hs3_scope_{}_{}_{}",
        std::process::id(),
        nanos,
        name
    ))
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

fn assert_pyhf_only_rejection(out: Output, command_label: &str) {
    assert!(!out.status.success(), "{command_label} should reject HS3 input");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("pyhf") && stderr.contains("HS3"),
        "{command_label} stderr should mention pyhf/HS3 scope, stderr={stderr}"
    );
}

#[test]
fn audit_rejects_hs3_input() {
    let hs3 = fixture_path("workspace-postFit_PTV.json");
    let out = run(&["audit", "--input", hs3.to_string_lossy().as_ref()]);
    assert_pyhf_only_rejection(out, "audit");
}

#[test]
fn report_rejects_hs3_input() {
    let hs3 = fixture_path("workspace-postFit_PTV.json");
    let xml = fixture_path("histfactory/combination.xml");
    let out_dir = tmp_path("report_hs3");

    let out = run(&[
        "report",
        "--input",
        hs3.to_string_lossy().as_ref(),
        "--histfactory-xml",
        xml.to_string_lossy().as_ref(),
        "--out-dir",
        out_dir.to_string_lossy().as_ref(),
    ]);
    assert_pyhf_only_rejection(out, "report");

    let _ = std::fs::remove_dir_all(&out_dir);
}

#[test]
fn viz_distributions_rejects_hs3_input() {
    let hs3 = fixture_path("workspace-postFit_PTV.json");
    let xml = fixture_path("histfactory/combination.xml");
    let out_json = tmp_path("viz_distributions_hs3.json");

    let out = run(&[
        "viz",
        "distributions",
        "--input",
        hs3.to_string_lossy().as_ref(),
        "--histfactory-xml",
        xml.to_string_lossy().as_ref(),
        "--output",
        out_json.to_string_lossy().as_ref(),
    ]);
    assert_pyhf_only_rejection(out, "viz distributions");

    let _ = std::fs::remove_file(&out_json);
}

#[test]
fn preprocess_smooth_rejects_hs3_input() {
    let hs3 = fixture_path("workspace-postFit_PTV.json");
    let out = run(&["preprocess", "smooth", "--input", hs3.to_string_lossy().as_ref()]);
    assert_pyhf_only_rejection(out, "preprocess smooth");
}
