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

fn fixture_path(rel: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(rel)
}

#[test]
fn import_trex_config_emits_analysis_yaml_and_coverage_json_and_expr_coverage_json() {
    let root = repo_root();
    let config = root.join("docs/examples/trex_config_ntup_minimal.txt");
    assert!(config.exists(), "missing config: {}", config.display());

    let ws_path = tmp_path("trex_ws.json");
    let yaml_path = tmp_path("trex_analysis.yaml");
    let cov_path = tmp_path("trex_coverage.json");
    let expr_cov_path = tmp_path("trex_expr_coverage.json");

    let out = run(&[
        "import",
        "trex-config",
        "--config",
        config.to_string_lossy().as_ref(),
        "--base-dir",
        root.to_string_lossy().as_ref(),
        "--output",
        ws_path.to_string_lossy().as_ref(),
        "--analysis-yaml",
        yaml_path.to_string_lossy().as_ref(),
        "--coverage-json",
        cov_path.to_string_lossy().as_ref(),
        "--expr-coverage-json",
        expr_cov_path.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "import trex-config should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(ws_path.exists(), "missing workspace output: {}", ws_path.display());
    assert!(yaml_path.exists(), "missing analysis yaml: {}", yaml_path.display());
    assert!(cov_path.exists(), "missing coverage json: {}", cov_path.display());
    assert!(expr_cov_path.exists(), "missing expr coverage json: {}", expr_cov_path.display());

    let spec: serde_yaml_ng::Value =
        serde_yaml_ng::from_slice(&std::fs::read(&yaml_path).unwrap()).unwrap();
    assert_eq!(spec.get("schema_version").and_then(|v| v.as_str()), Some("trex_analysis_spec_v0"));
    assert_eq!(
        spec.get("inputs").and_then(|v| v.get("mode")).and_then(|v| v.as_str()),
        Some("trex_config_txt")
    );
    let cfg_path = spec
        .get("inputs")
        .and_then(|v| v.get("trex_config_txt"))
        .and_then(|v| v.get("config_path"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(
        cfg_path.ends_with("docs/examples/trex_config_ntup_minimal.txt"),
        "expected config_path to reference fixture, got={cfg_path:?}"
    );
    let base_dir = spec
        .get("inputs")
        .and_then(|v| v.get("trex_config_txt"))
        .and_then(|v| v.get("base_dir"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(
        base_dir.ends_with("nextstat.io"),
        "expected base_dir to be repo root, got={base_dir:?}"
    );

    let cov: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&cov_path).unwrap()).unwrap();
    assert_eq!(cov.get("schema_version").and_then(|v| v.as_str()), Some("trex_config_coverage_v0"));
    let unknown = cov.get("unknown").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert!(unknown.is_empty(), "expected no unknown keys in minimal config, got={unknown:?}");

    let rep: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&expr_cov_path).unwrap()).unwrap();
    assert_eq!(rep.get("schema_version").and_then(|v| v.as_str()), Some("trex_expr_coverage_v0"));
    assert_eq!(rep.get("n_err").and_then(|v| v.as_u64()), Some(0));
    let items = rep.get("items").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    assert!(!items.is_empty(), "expected some expressions in report");

    let _ = std::fs::remove_file(&ws_path);
    let _ = std::fs::remove_file(&yaml_path);
    let _ = std::fs::remove_file(&cov_path);
    let _ = std::fs::remove_file(&expr_cov_path);
}

#[test]
fn import_trex_config_hist_mode_matches_histfactory_fixture() {
    let root = repo_root();
    let config = root.join("docs/examples/trex_config_hist_minimal.txt");
    let expected = fixture_path("histfactory/workspace.json");
    assert!(config.exists(), "missing config: {}", config.display());
    assert!(expected.exists(), "missing fixture: {}", expected.display());

    let ws_path = tmp_path("trex_hist_ws.json");

    let out = run(&[
        "import",
        "trex-config",
        "--config",
        config.to_string_lossy().as_ref(),
        "--base-dir",
        root.to_string_lossy().as_ref(),
        "--output",
        ws_path.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "import trex-config (HIST) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let got: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&ws_path).unwrap()).expect("output should be JSON");
    let want: serde_json::Value = serde_json::from_slice(&std::fs::read(&expected).unwrap())
        .expect("expected fixture should be valid JSON");
    assert_eq!(got, want, "workspace JSON mismatch (HIST mode)");

    let _ = std::fs::remove_file(&ws_path);
}
