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

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(name)
}

fn tmp_path(filename: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let mut p = std::env::temp_dir();
    p.push(format!("nextstat_cli_{}_{}_{}", std::process::id(), nanos, filename));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

#[test]
fn import_histfactory_writes_workspace_json_to_stdout() {
    let xml = fixture_path("histfactory/combination.xml");
    let expected = fixture_path("histfactory/workspace.json");
    assert!(xml.exists(), "missing fixture: {}", xml.display());
    assert!(expected.exists(), "missing fixture: {}", expected.display());

    let out = run(&["import", "histfactory", "--xml", xml.to_string_lossy().as_ref()]);
    assert!(
        out.status.success(),
        "import histfactory should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let got: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    let want: serde_json::Value = serde_json::from_slice(&std::fs::read(&expected).unwrap())
        .expect("expected fixture should be valid JSON");

    assert_eq!(got, want, "workspace JSON mismatch");
}

#[test]
fn import_histfactory_dir_autodiscovers_combination_xml() {
    let dir = fixture_path("histfactory");
    let expected = fixture_path("histfactory/workspace.json");
    assert!(dir.is_dir(), "missing fixture dir: {}", dir.display());
    assert!(expected.exists(), "missing fixture: {}", expected.display());

    let out = run(&["import", "histfactory", "--dir", dir.to_string_lossy().as_ref()]);
    assert!(
        out.status.success(),
        "import histfactory --dir should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let got: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout should be valid JSON");
    let want: serde_json::Value = serde_json::from_slice(&std::fs::read(&expected).unwrap())
        .expect("expected fixture should be valid JSON");

    assert_eq!(got, want, "workspace JSON mismatch");
}

#[test]
fn import_histfactory_writes_workspace_json_to_file() {
    let xml = fixture_path("histfactory/combination.xml");
    let out_path = tmp_path("import_histfactory_workspace.json");

    let out = run(&[
        "import",
        "histfactory",
        "--xml",
        xml.to_string_lossy().as_ref(),
        "--output",
        out_path.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "import histfactory should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out_path.exists(), "expected output file to exist: {}", out_path.display());

    let got: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&out_path).unwrap()).expect("output should be JSON");
    assert!(got.get("channels").is_some(), "expected channels in output");
    assert!(got.get("observations").is_some(), "expected observations in output");
    assert!(got.get("measurements").is_some(), "expected measurements in output");

    let _ = std::fs::remove_file(&out_path);
}

#[test]
fn import_histfactory_errors_on_missing_xml() {
    let missing = tmp_path("does_not_exist_combination.xml");
    let out = run(&["import", "histfactory", "--xml", missing.to_string_lossy().as_ref()]);
    assert!(!out.status.success(), "expected failure for missing xml");
}
