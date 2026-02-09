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

fn python_supports_pyhf_writexml() -> bool {
    let out = Command::new("python3").args(["-c", "import pyhf, uproot, pyhf.writexml"]).output();
    matches!(out, Ok(o) if o.status.success())
}

#[test]
fn export_histfactory_writes_expected_files_when_python_available() {
    if !python_supports_pyhf_writexml() {
        // Optional cross-check feature: skip if the environment doesn't have pyhf/uproot.
        return;
    }

    let input = fixture_path("histfactory/workspace.json");
    assert!(input.exists(), "missing fixture: {}", input.display());

    let out_dir = tmp_dir("export_histfactory");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    let out = run(&[
        "export",
        "histfactory",
        "--input",
        input.to_string_lossy().as_ref(),
        "--out-dir",
        out_dir.to_string_lossy().as_ref(),
        "--python",
        "python3",
    ]);
    assert!(
        out.status.success(),
        "export histfactory should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(out_dir.join("combination.xml").is_file());
    assert!(out_dir.join("data.root").is_file());
    assert!(out_dir.join("HistFactorySchema.dtd").is_file());
    assert!(out_dir.join("channels").is_dir());

    // Smoke: we can re-import what we just exported.
    let out2 = run(&["import", "histfactory", "--dir", out_dir.to_string_lossy().as_ref()]);
    assert!(
        out2.status.success(),
        "re-import should succeed, stderr={}",
        String::from_utf8_lossy(&out2.stderr)
    );
    let got: serde_json::Value =
        serde_json::from_slice(&out2.stdout).expect("re-import stdout should be valid JSON");
    assert!(got.get("channels").is_some());
    assert!(got.get("observations").is_some());
    assert!(got.get("measurements").is_some());

    let _ = std::fs::remove_dir_all(&out_dir);
}
