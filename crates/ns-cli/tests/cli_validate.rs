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
fn validate_accepts_analysis_spec_v0() {
    let root = repo_root();
    let export_dir = root.join("tests/fixtures/histfactory");
    let spec_dir = tmp_dir("validate_spec");
    std::fs::create_dir_all(&spec_dir).unwrap();

    let spec_path = spec_dir.join("analysis.yaml");
    let yaml = format!(
        r#"schema_version: trex_analysis_spec_v0
analysis: {{ name: "fixture", description: "fixture", tags: ["test"] }}
inputs:
  mode: histfactory_xml
  histfactory:
    export_dir: "{export_dir}"
    combination_xml: null
    measurement: NominalMeasurement
execution:
  determinism: {{ threads: 1 }}
  import: {{ enabled: true, output_json: "{ws_out}" }}
  fit: {{ enabled: false, output_json: "{fit_out}" }}
  profile_scan: {{ enabled: false, start: 0.0, stop: 5.0, points: 21, output_json: "{scan_out}" }}
  report:
    enabled: false
    out_dir: "{report_dir}"
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render: {{ enabled: false, pdf: null, svg_dir: null, python: null }}
    skip_uncertainty: true
    uncertainty_grouping: prefix_1
gates:
  baseline_compare: {{ enabled: false, baseline_dir: tmp/baselines, require_same_host: true, max_slowdown: 1.3 }}
"#,
        export_dir = export_dir.display(),
        ws_out = spec_dir.join("workspace.json").display(),
        fit_out = spec_dir.join("fit.json").display(),
        scan_out = spec_dir.join("scan.json").display(),
        report_dir = spec_dir.join("report").display(),
    );
    std::fs::write(&spec_path, yaml).unwrap();

    let out = run(&["validate", "--config", spec_path.to_string_lossy().as_ref()]);
    assert!(
        out.status.success(),
        "validate should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("analysis_spec_v0"));

    let _ = std::fs::remove_dir_all(&spec_dir);
}

#[test]
fn validate_accepts_analysis_spec_v0_trex_config_yaml() {
    let root = repo_root();
    let spec_dir = tmp_dir("validate_spec_trex_yaml");
    std::fs::create_dir_all(&spec_dir).unwrap();

    let spec_path = spec_dir.join("analysis.yaml");
    let yaml = format!(
        r#"schema_version: trex_analysis_spec_v0
analysis: {{ name: "fixture", description: "fixture", tags: ["test"] }}
inputs:
  mode: trex_config_yaml
  trex_config_yaml:
    base_dir: "{base_dir}"
    read_from: NTUP
    tree_name: events
    measurement: meas
    poi: mu
    regions:
      - name: SR
        variable: mbb
        binning_edges: [0, 1]
    samples:
      - name: signal
        kind: mc
        file: tests/fixtures/simple_tree.root
execution:
  determinism: {{ threads: 1 }}
  import: {{ enabled: true, output_json: "{ws_out}" }}
  fit: {{ enabled: false, output_json: "{fit_out}" }}
  profile_scan: {{ enabled: false, start: 0.0, stop: 5.0, points: 21, output_json: "{scan_out}" }}
  report:
    enabled: false
    out_dir: "{report_dir}"
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render: {{ enabled: false, pdf: null, svg_dir: null, python: null }}
    skip_uncertainty: true
    uncertainty_grouping: prefix_1
gates:
  baseline_compare: {{ enabled: false, baseline_dir: tmp/baselines, require_same_host: true, max_slowdown: 1.3 }}
"#,
        base_dir = root.display(),
        ws_out = spec_dir.join("workspace.json").display(),
        fit_out = spec_dir.join("fit.json").display(),
        scan_out = spec_dir.join("scan.json").display(),
        report_dir = spec_dir.join("report").display(),
    );
    std::fs::write(&spec_path, yaml).unwrap();

    let out = run(&["validate", "--config", spec_path.to_string_lossy().as_ref()]);
    assert!(
        out.status.success(),
        "validate should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let _ = std::fs::remove_dir_all(&spec_dir);
}

#[test]
fn validate_accepts_legacy_run_config() {
    let root = repo_root();
    let xml = root.join("tests/fixtures/histfactory/combination.xml");
    assert!(xml.exists());

    let run_dir = tmp_dir("validate_run_config");
    std::fs::create_dir_all(&run_dir).unwrap();
    let cfg_path = run_dir.join("run.yaml");

    let yaml = format!(
        r#"histfactory_xml: "{xml}"
out_dir: "{out_dir}"
threads: 1
deterministic: true
overwrite: true
include_covariance: false
skip_uncertainty: true
uncertainty_grouping: prefix_1
render: false
"#,
        xml = xml.display(),
        out_dir = run_dir.join("out").display(),
    );
    std::fs::write(&cfg_path, yaml).unwrap();

    let out = run(&["validate", "--config", cfg_path.to_string_lossy().as_ref()]);
    assert!(
        out.status.success(),
        "validate should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("run_config_legacy"));

    let _ = std::fs::remove_dir_all(&run_dir);
}
