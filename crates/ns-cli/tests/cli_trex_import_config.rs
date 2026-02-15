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

fn fixture_path(rel: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(rel)
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

#[test]
fn trex_import_config_hist_runs_nextstat_run_full_report_e2e() {
    let root = repo_root();
    let cfg = root.join("docs/examples/trex_config_hist_minimal.txt");
    let histfactory_xml = fixture_path("histfactory/combination.xml");
    assert!(cfg.exists(), "missing fixture: {}", cfg.display());
    assert!(histfactory_xml.exists(), "missing HistFactory XML: {}", histfactory_xml.display());

    let out_dir = tmp_dir("trex_import_config_hist_e2e");
    std::fs::create_dir_all(&out_dir).unwrap();

    let analysis_path = out_dir.join("analysis.yaml");
    let mapping_path = out_dir.join("mapping.json");
    let run_ws_path = out_dir.join("workspace.json");
    let report_dir = out_dir.join("report");

    let outp = run(&[
        "trex",
        "import-config",
        "--config",
        cfg.to_string_lossy().as_ref(),
        "--out",
        analysis_path.to_string_lossy().as_ref(),
        "--report",
        mapping_path.to_string_lossy().as_ref(),
        "--workspace-out",
        run_ws_path.to_string_lossy().as_ref(),
        "--overwrite",
    ]);
    assert!(
        outp.status.success(),
        "trex import-config should succeed, stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&outp.stdout),
        String::from_utf8_lossy(&outp.stderr)
    );

    assert!(analysis_path.exists(), "missing analysis spec: {}", analysis_path.display());
    assert!(mapping_path.exists(), "missing mapping report: {}", mapping_path.display());

    let mut spec: serde_json::Value =
        serde_yaml_ng::from_slice(&std::fs::read(&analysis_path).unwrap()).unwrap();
    spec["inputs"]["trex_config_yaml"]["base_dir"] =
        serde_json::Value::String(root.to_string_lossy().into_owned());
    spec["execution"]["report"]["enabled"] = serde_json::Value::Bool(true);
    spec["execution"]["report"]["out_dir"] =
        serde_json::Value::String(report_dir.to_string_lossy().into_owned());
    spec["execution"]["report"]["overwrite"] = serde_json::Value::Bool(true);
    spec["execution"]["report"]["histfactory_xml"] =
        serde_json::Value::String(histfactory_xml.to_string_lossy().into_owned());
    spec["execution"]["report"]["skip_uncertainty"] = serde_json::Value::Bool(false);
    std::fs::write(&analysis_path, serde_yaml_ng::to_string(&spec).unwrap()).unwrap();

    let runp = run(&["run", "--config", analysis_path.to_string_lossy().as_ref()]);
    assert!(
        runp.status.success(),
        "nextstat run should succeed, stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&runp.stdout),
        String::from_utf8_lossy(&runp.stderr)
    );

    assert!(run_ws_path.exists(), "missing workspace.json: {}", run_ws_path.display());
    assert!(report_dir.join("fit.json").exists(), "missing report fit.json");
    assert!(report_dir.join("distributions.json").exists(), "missing report distributions.json");
    assert!(report_dir.join("yields.json").exists(), "missing report yields.json");
    assert!(report_dir.join("pulls.json").exists(), "missing report pulls.json");
    assert!(report_dir.join("corr.json").exists(), "missing report corr.json");
    assert!(report_dir.join("uncertainty.json").exists(), "missing report uncertainty.json");

    let _ = std::fs::remove_dir_all(&out_dir);
}
