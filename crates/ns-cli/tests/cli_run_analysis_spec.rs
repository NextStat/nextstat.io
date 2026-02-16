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

fn run_with_env(args: &[&str], envs: &[(&str, &str)]) -> Output {
    let mut cmd = Command::new(bin_path());
    cmd.args(args);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.output().unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

fn python_for_render() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NEXTSTAT_TEST_PYTHON")
        && !p.trim().is_empty()
    {
        return Some(PathBuf::from(p));
    }
    let venv = repo_root().join(".venv/bin/python");
    if venv.exists() {
        return Some(venv);
    }
    None
}

fn python_has_render_deps(python: &PathBuf) -> bool {
    let mut cmd = Command::new(python);
    cmd.arg("-c").arg(
        "import importlib.util; \
         assert importlib.util.find_spec('matplotlib') is not None; \
         assert importlib.util.find_spec('PIL') is not None",
    );
    cmd.status().map(|s| s.success()).unwrap_or(false)
}

#[test]
fn run_accepts_analysis_spec_v0_trex_config_yaml_fit() {
    let root = repo_root();
    let spec_dir = tmp_dir("run_spec_trex_yaml");
    std::fs::create_dir_all(&spec_dir).unwrap();

    let workspace_json = spec_dir.join("workspace.json");
    let fit_json = spec_dir.join("fit.json");
    let spec_path = spec_dir.join("analysis.yaml");

    let yaml = format!(
        r#"$schema: https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json
schema_version: trex_analysis_spec_v0

analysis:
  name: "fixture"
  description: "fixture"
  tags: ["test"]

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
        binning_edges: [0, 50, 100, 150, 200, 300]
        selection: "njet >= 4"
    samples:
      - name: signal
        kind: mc
        file: tests/fixtures/simple_tree.root
        weight: weight_mc
        regions: ["SR"]
        norm_factors: ["mu"]
        stat_error: true
    systematics:
      - name: jes
        type: weight
        samples: ["signal"]
        regions: ["SR"]
        weight_up: weight_jes_up
        weight_down: weight_jes_down

execution:
  determinism:
    threads: 1

  import:
    enabled: true
    output_json: "{workspace_json}"

  fit:
    enabled: true
    output_json: "{fit_json}"

  profile_scan:
    enabled: false
    start: 0.0
    stop: 5.0
    points: 21
    output_json: "{scan_json}"

  report:
    enabled: false
    out_dir: "{report_dir}"
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render:
      enabled: false
      pdf: null
      svg_dir: null
      python: null
    skip_uncertainty: true
    uncertainty_grouping: prefix_1

gates:
  baseline_compare:
    enabled: false
    baseline_dir: tmp/baselines
    require_same_host: true
    max_slowdown: 1.3
"#,
        base_dir = root.display(),
        workspace_json = workspace_json.display(),
        fit_json = fit_json.display(),
        scan_json = spec_dir.join("scan.json").display(),
        report_dir = spec_dir.join("report").display(),
    );
    std::fs::write(&spec_path, yaml).unwrap();

    let out = run(&["run", "--config", spec_path.to_string_lossy().as_ref()]);
    assert!(
        out.status.success(),
        "run should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(workspace_json.exists(), "missing workspace.json: {}", workspace_json.display());
    assert!(fit_json.exists(), "missing fit.json: {}", fit_json.display());

    let _ = std::fs::remove_dir_all(&spec_dir);
}

#[test]
fn run_accepts_analysis_spec_v0_trex_config_yaml_hist_mode_import_only() {
    let root = repo_root();
    let spec_dir = tmp_dir("run_spec_trex_yaml_hist");
    std::fs::create_dir_all(&spec_dir).unwrap();

    let workspace_json = spec_dir.join("workspace.json");
    let spec_path = spec_dir.join("analysis.yaml");

    let yaml = format!(
        r#"$schema: https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json
schema_version: trex_analysis_spec_v0

analysis:
  name: "fixture"
  description: "fixture"
  tags: ["test"]

inputs:
  mode: trex_config_yaml
  trex_config_yaml:
    base_dir: "{base_dir}"
    read_from: HIST
    histo_path: tests/fixtures/histfactory

execution:
  determinism:
    threads: 1

  import:
    enabled: true
    output_json: "{workspace_json}"

  fit:
    enabled: false
    output_json: "{fit_json}"

  profile_scan:
    enabled: false
    start: 0.0
    stop: 5.0
    points: 21
    output_json: "{scan_json}"

  report:
    enabled: false
    out_dir: "{report_dir}"
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render:
      enabled: false
      pdf: null
      svg_dir: null
      python: null
    skip_uncertainty: true
    uncertainty_grouping: prefix_1

gates:
  baseline_compare:
    enabled: false
    baseline_dir: tmp/baselines
    require_same_host: true
    max_slowdown: 1.3
"#,
        base_dir = root.display(),
        workspace_json = workspace_json.display(),
        fit_json = spec_dir.join("fit.json").display(),
        scan_json = spec_dir.join("scan.json").display(),
        report_dir = spec_dir.join("report").display(),
    );
    std::fs::write(&spec_path, yaml).unwrap();

    let out = run(&["run", "--config", spec_path.to_string_lossy().as_ref()]);
    assert!(
        out.status.success(),
        "run should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        workspace_json.exists(),
        "expected workspace.json to exist: {}",
        workspace_json.display()
    );
}

#[test]
fn run_accepts_analysis_spec_v0_histfactory_import_only_with_bundle() {
    let root = repo_root();
    let export_dir = root.join("tests/fixtures/histfactory");
    let spec_dir = tmp_dir("run_spec_histfactory_bundle");
    let bundle_dir = tmp_dir("run_spec_bundle_dir");
    std::fs::create_dir_all(&spec_dir).unwrap();

    // Ensure clean.
    let _ = std::fs::remove_dir_all(&bundle_dir);

    let workspace_json = spec_dir.join("workspace.json");
    let spec_path = spec_dir.join("analysis.yaml");

    let yaml = format!(
        r#"$schema: https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json
schema_version: trex_analysis_spec_v0

analysis:
  name: "fixture"
  description: "fixture"
  tags: ["test"]

inputs:
  mode: histfactory_xml
  histfactory:
    export_dir: "{export_dir}"
    combination_xml: null
    measurement: NominalMeasurement

execution:
  determinism:
    threads: 1

  import:
    enabled: true
    output_json: "{workspace_json}"

  fit:
    enabled: false
    output_json: "{fit_json}"

  profile_scan:
    enabled: false
    start: 0.0
    stop: 5.0
    points: 21
    output_json: "{scan_json}"

  report:
    enabled: false
    out_dir: "{report_dir}"
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render:
      enabled: false
      pdf: null
      svg_dir: null
      python: null
    skip_uncertainty: true
    uncertainty_grouping: prefix_1

gates:
  baseline_compare:
    enabled: false
    baseline_dir: tmp/baselines
    require_same_host: true
    max_slowdown: 1.3
"#,
        export_dir = export_dir.display(),
        workspace_json = workspace_json.display(),
        fit_json = spec_dir.join("fit.json").display(),
        scan_json = spec_dir.join("scan.json").display(),
        report_dir = spec_dir.join("report").display(),
    );
    std::fs::write(&spec_path, yaml).unwrap();

    let out = run(&[
        "--bundle",
        bundle_dir.to_string_lossy().as_ref(),
        "run",
        "--config",
        spec_path.to_string_lossy().as_ref(),
    ]);
    assert!(
        out.status.success(),
        "run should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(workspace_json.exists(), "missing workspace.json: {}", workspace_json.display());

    // Bundle presence (contract).
    assert!(bundle_dir.join("meta.json").exists());
    assert!(bundle_dir.join("provenance.json").exists());
    assert!(bundle_dir.join("manifest.json").exists());
    assert!(bundle_dir.join("inputs/analysis.yaml").exists());
    assert!(bundle_dir.join("inputs/histfactory/combination.xml").exists());
    assert!(bundle_dir.join("outputs/workspace.json").exists());

    let _ = std::fs::remove_dir_all(&spec_dir);
    let _ = std::fs::remove_dir_all(&bundle_dir);
}

#[test]
fn run_accepts_analysis_spec_v0_histfactory_report_only() {
    let root = repo_root();
    let export_dir = root.join("tests/fixtures/histfactory");
    let spec_dir = tmp_dir("run_spec_histfactory");
    std::fs::create_dir_all(&spec_dir).unwrap();

    let workspace_json = spec_dir.join("workspace.json");
    let report_dir = spec_dir.join("report");
    let spec_path = spec_dir.join("analysis.yaml");

    let yaml = format!(
        r#"$schema: https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json
schema_version: trex_analysis_spec_v0

analysis:
  name: "fixture"
  description: "fixture"
  tags: ["test"]

inputs:
  mode: histfactory_xml
  histfactory:
    export_dir: "{export_dir}"
    combination_xml: null
    measurement: NominalMeasurement

execution:
  determinism:
    threads: 1

  import:
    enabled: true
    output_json: "{workspace_json}"

  fit:
    enabled: false
    output_json: "{fit_json}"

  profile_scan:
    enabled: false
    start: 0.0
    stop: 5.0
    points: 21
    output_json: "{scan_json}"

  report:
    enabled: true
    out_dir: "{report_dir}"
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render:
      enabled: false
      pdf: null
      svg_dir: null
      python: null
    skip_uncertainty: true
    uncertainty_grouping: prefix_1

gates:
  baseline_compare:
    enabled: false
    baseline_dir: tmp/baselines
    require_same_host: true
    max_slowdown: 1.3
"#,
        export_dir = export_dir.display(),
        workspace_json = workspace_json.display(),
        report_dir = report_dir.display(),
        fit_json = spec_dir.join("fit.json").display(),
        scan_json = spec_dir.join("scan.json").display(),
    );
    std::fs::write(&spec_path, yaml).unwrap();

    let out = run(&["run", "--config", spec_path.to_string_lossy().as_ref()]);
    assert!(
        out.status.success(),
        "run should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(workspace_json.exists(), "missing workspace.json: {}", workspace_json.display());
    assert!(report_dir.join("fit.json").exists(), "missing report fit.json");
    assert!(report_dir.join("distributions.json").exists(), "missing distributions.json");
    assert!(report_dir.join("yields.json").exists(), "missing yields.json");
    assert!(report_dir.join("pulls.json").exists(), "missing pulls.json");
    assert!(report_dir.join("corr.json").exists(), "missing corr.json");

    let _ = std::fs::remove_dir_all(&spec_dir);
}

#[test]
fn run_analysis_spec_v0_histfactory_report_render_e2e() {
    let Some(python) = python_for_render() else {
        eprintln!("skip e2e render test: .venv/bin/python (or NEXTSTAT_TEST_PYTHON) not available");
        return;
    };
    if !python_has_render_deps(&python) {
        eprintln!("skip e2e render test: python lacks matplotlib/Pillow");
        return;
    }

    let root = repo_root();
    let export_dir = root.join("tests/fixtures/histfactory");
    let spec_dir = tmp_dir("run_spec_histfactory_render_e2e");
    std::fs::create_dir_all(&spec_dir).unwrap();

    let workspace_json = spec_dir.join("workspace.json");
    let report_dir = spec_dir.join("report");
    let svg_dir = report_dir.join("svg");
    let pdf_path = report_dir.join("report.pdf");
    let spec_path = spec_dir.join("analysis.yaml");

    let yaml = format!(
        r#"$schema: https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json
schema_version: trex_analysis_spec_v0

analysis:
  name: "fixture-render-e2e"
  description: "fixture-render-e2e"
  tags: ["test", "render", "e2e"]

inputs:
  mode: histfactory_xml
  histfactory:
    export_dir: "{export_dir}"
    combination_xml: null
    measurement: NominalMeasurement

execution:
  determinism:
    threads: 1

  import:
    enabled: true
    output_json: "{workspace_json}"

  fit:
    enabled: false
    output_json: "{fit_json}"

  profile_scan:
    enabled: false
    start: 0.0
    stop: 5.0
    points: 21
    output_json: "{scan_json}"

  report:
    enabled: true
    out_dir: "{report_dir}"
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render:
      enabled: true
      pdf: "{pdf_path}"
      svg_dir: "{svg_dir}"
      python: "{python}"
      label_status: Preliminary
      sqrt_s_tev: 13.6
      show_mc_band: true
      show_stat_band: true
      band_hatch: "////"
      palette: tableau10
    skip_uncertainty: true
    uncertainty_grouping: prefix_1

gates:
  baseline_compare:
    enabled: false
    baseline_dir: tmp/baselines
    require_same_host: true
    max_slowdown: 1.3
"#,
        export_dir = export_dir.display(),
        workspace_json = workspace_json.display(),
        report_dir = report_dir.display(),
        pdf_path = pdf_path.display(),
        svg_dir = svg_dir.display(),
        python = python.display(),
        fit_json = spec_dir.join("fit.json").display(),
        scan_json = spec_dir.join("scan.json").display(),
    );
    std::fs::write(&spec_path, yaml).unwrap();

    let out = run_with_env(
        &["run", "--config", spec_path.to_string_lossy().as_ref()],
        &[("NEXTSTAT_FORCE_PYTHONPATH", "1")],
    );
    assert!(
        out.status.success(),
        "run(render e2e) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(workspace_json.exists(), "missing workspace.json");
    assert!(report_dir.join("fit.json").exists(), "missing report fit.json");
    assert!(report_dir.join("distributions.json").exists(), "missing distributions.json");
    assert!(pdf_path.exists(), "missing report.pdf");
    assert!(pdf_path.metadata().map(|m| m.len()).unwrap_or(0) > 0, "empty report.pdf");

    let dist_svg = svg_dir.join("distributions__SR.svg");
    let dist_png = svg_dir.join("distributions__SR.png");
    assert!(dist_svg.exists(), "missing rendered svg");
    assert!(dist_png.exists(), "missing rendered png");
    assert!(dist_png.metadata().map(|m| m.len()).unwrap_or(0) > 0, "empty rendered png");

    let svg_text = std::fs::read_to_string(&dist_svg).expect("read svg");
    assert!(svg_text.contains("NEXTSTAT"), "svg should include NEXTSTAT header");
    assert!(svg_text.contains("Preliminary"), "svg should include label_status");
    assert!(svg_text.contains("13.6"), "svg should include sqrt(s) value");

    let _ = std::fs::remove_dir_all(&spec_dir);
}

#[test]
fn run_analysis_spec_v0_histfactory_report_render_blinded_e2e() {
    let Some(python) = python_for_render() else {
        eprintln!(
            "skip blinded e2e render test: .venv/bin/python (or NEXTSTAT_TEST_PYTHON) not available"
        );
        return;
    };
    if !python_has_render_deps(&python) {
        eprintln!("skip blinded e2e render test: python lacks matplotlib/Pillow");
        return;
    }

    let root = repo_root();
    let export_dir = root.join("tests/fixtures/histfactory");
    let spec_dir = tmp_dir("run_spec_histfactory_render_blinded_e2e");
    std::fs::create_dir_all(&spec_dir).unwrap();

    let workspace_json = spec_dir.join("workspace.json");
    let report_dir = spec_dir.join("report");
    let svg_dir = report_dir.join("svg");
    let pdf_path = report_dir.join("report.pdf");
    let spec_path = spec_dir.join("analysis.yaml");

    let yaml = format!(
        r#"$schema: https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json
schema_version: trex_analysis_spec_v0

analysis:
  name: "fixture-render-blinded-e2e"
  description: "fixture-render-blinded-e2e"
  tags: ["test", "render", "e2e", "blinded"]

inputs:
  mode: histfactory_xml
  histfactory:
    export_dir: "{export_dir}"
    combination_xml: null
    measurement: NominalMeasurement

execution:
  determinism:
    threads: 1

  import:
    enabled: true
    output_json: "{workspace_json}"

  fit:
    enabled: false
    output_json: "{fit_json}"

  profile_scan:
    enabled: false
    start: 0.0
    stop: 5.0
    points: 21
    output_json: "{scan_json}"

  report:
    enabled: true
    out_dir: "{report_dir}"
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render:
      enabled: true
      pdf: "{pdf_path}"
      svg_dir: "{svg_dir}"
      python: "{python}"
      label_status: Internal
      sqrt_s_tev: 13.0
      show_mc_band: true
      show_stat_band: true
      band_hatch: "////"
      palette: hep2026
    skip_uncertainty: true
    uncertainty_grouping: prefix_1
    blind_regions: [SR]

gates:
  baseline_compare:
    enabled: false
    baseline_dir: tmp/baselines
    require_same_host: true
    max_slowdown: 1.3
"#,
        export_dir = export_dir.display(),
        workspace_json = workspace_json.display(),
        report_dir = report_dir.display(),
        pdf_path = pdf_path.display(),
        svg_dir = svg_dir.display(),
        python = python.display(),
        fit_json = spec_dir.join("fit.json").display(),
        scan_json = spec_dir.join("scan.json").display(),
    );
    std::fs::write(&spec_path, yaml).unwrap();

    let out = run_with_env(
        &["run", "--config", spec_path.to_string_lossy().as_ref()],
        &[("NEXTSTAT_FORCE_PYTHONPATH", "1")],
    );
    assert!(
        out.status.success(),
        "run(render blinded e2e) should succeed, stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(workspace_json.exists(), "missing workspace.json");
    assert!(report_dir.join("distributions.json").exists(), "missing distributions.json");
    assert!(pdf_path.exists(), "missing report.pdf");

    let dist_svg = svg_dir.join("distributions__SR.svg");
    let dist_png = svg_dir.join("distributions__SR.png");
    assert!(dist_svg.exists(), "missing rendered svg");
    assert!(dist_png.exists(), "missing rendered png");

    // Artifact-level blinding contract.
    let dist_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(report_dir.join("distributions.json")).unwrap())
            .expect("parse distributions.json");
    let channels = dist_json.get("channels").and_then(|v| v.as_array()).expect("channels array");
    let sr = channels
        .iter()
        .find(|c| c.get("channel_name").and_then(|v| v.as_str()) == Some("SR"))
        .expect("SR channel");
    assert_eq!(sr.get("data_is_blinded").and_then(|v| v.as_bool()), Some(true));
    for key in ["data_y", "ratio_y"] {
        let xs = sr.get(key).and_then(|v| v.as_array()).expect("numeric array");
        assert!(
            xs.iter().all(|v| v.as_f64().unwrap_or(f64::NAN) == 0.0),
            "blinded channel should have zeroed {}",
            key
        );
    }

    // Render-level blinding contract.
    let svg_text = std::fs::read_to_string(&dist_svg).expect("read svg");
    assert!(svg_text.contains("BLINDED"), "svg should mark blinded channel");
    assert!(
        !svg_text.contains("Data / MC"),
        "ratio panel label should be hidden for blinded channel"
    );

    let _ = std::fs::remove_dir_all(&spec_dir);
}

#[test]
fn run_analysis_spec_v0_workspace_json_requires_report_histfactory_xml() {
    let root = repo_root();
    let ws = root.join("tests/fixtures/histfactory/workspace.json");
    assert!(ws.exists());

    let spec_dir = tmp_dir("run_spec_workspace_json");
    std::fs::create_dir_all(&spec_dir).unwrap();
    let report_dir = spec_dir.join("report");
    let spec_path = spec_dir.join("analysis.yaml");

    let yaml = format!(
        r#"$schema: https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json
schema_version: trex_analysis_spec_v0

analysis:
  name: "fixture"
  description: "fixture"
  tags: ["test"]

inputs:
  mode: workspace_json
  workspace_json:
    path: "{ws}"

execution:
  determinism:
    threads: 1

  import:
    enabled: false
    output_json: "{unused}"

  fit:
    enabled: false
    output_json: "{unused_fit}"

  profile_scan:
    enabled: false
    start: 0.0
    stop: 5.0
    points: 21
    output_json: "{unused_scan}"

  report:
    enabled: true
    out_dir: "{report_dir}"
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render:
      enabled: false
      pdf: null
      svg_dir: null
      python: null
    skip_uncertainty: true
    uncertainty_grouping: prefix_1

gates:
  baseline_compare:
    enabled: false
    baseline_dir: tmp/baselines
    require_same_host: true
    max_slowdown: 1.3
"#,
        ws = ws.display(),
        report_dir = report_dir.display(),
        unused = spec_dir.join("unused.json").display(),
        unused_fit = spec_dir.join("unused_fit.json").display(),
        unused_scan = spec_dir.join("unused_scan.json").display(),
    );
    std::fs::write(&spec_path, yaml).unwrap();

    let out = run(&["run", "--config", spec_path.to_string_lossy().as_ref()]);
    assert!(
        !out.status.success(),
        "run should fail when report.histfactory_xml is null in workspace_json mode"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("execution.report.histfactory_xml"),
        "stderr should mention report.histfactory_xml, got={stderr}"
    );

    let _ = std::fs::remove_dir_all(&spec_dir);
}

#[test]
fn run_analysis_spec_v0_baseline_gate_requires_manifest() {
    let root = repo_root();
    let ws = root.join("tests/fixtures/histfactory/workspace.json");
    assert!(ws.exists());

    let spec_dir = tmp_dir("run_spec_baseline_gate_missing_manifest");
    std::fs::create_dir_all(&spec_dir).unwrap();
    let missing_baseline_dir = spec_dir.join("missing_baselines");
    let spec_path = spec_dir.join("analysis.yaml");

    let yaml = format!(
        r#"$schema: https://nextstat.io/schemas/trex/analysis_spec_v0.schema.json
schema_version: trex_analysis_spec_v0

analysis:
  name: "fixture"
  description: "fixture"
  tags: ["test"]

inputs:
  mode: workspace_json
  workspace_json:
    path: "{ws}"

execution:
  determinism:
    threads: 1

  import:
    enabled: false
    output_json: "{unused}"

  fit:
    enabled: false
    output_json: "{unused_fit}"

  profile_scan:
    enabled: false
    start: 0.0
    stop: 5.0
    points: 21
    output_json: "{unused_scan}"

  report:
    enabled: false
    out_dir: "{report_dir}"
    overwrite: true
    include_covariance: false
    histfactory_xml: null
    render:
      enabled: false
      pdf: null
      svg_dir: null
      python: null
    skip_uncertainty: true
    uncertainty_grouping: prefix_1

gates:
  baseline_compare:
    enabled: true
    baseline_dir: "{baseline_dir}"
    require_same_host: true
    max_slowdown: 1.3
"#,
        ws = ws.display(),
        unused = spec_dir.join("unused.json").display(),
        unused_fit = spec_dir.join("unused_fit.json").display(),
        unused_scan = spec_dir.join("unused_scan.json").display(),
        report_dir = spec_dir.join("report").display(),
        baseline_dir = missing_baseline_dir.display(),
    );
    std::fs::write(&spec_path, yaml).unwrap();

    let out = run(&["run", "--config", spec_path.to_string_lossy().as_ref()]);
    assert!(
        !out.status.success(),
        "run should fail when baseline_compare gate is enabled without manifest"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("baseline_compare gate is enabled")
            && stderr.contains("latest_trex_analysis_spec_manifest.json"),
        "stderr should explain missing baseline manifest, got={stderr}"
    );

    let _ = std::fs::remove_dir_all(&spec_dir);
}
