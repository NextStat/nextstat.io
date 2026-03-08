use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_TMP_ID: AtomicU64 = AtomicU64::new(0);

fn bin_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_nextstat"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn fixture_path(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures").join(name)
}

fn doc_spec_path(name: &str) -> PathBuf {
    repo_root().join("docs/specs").join(name)
}

fn tmp_file_path(suffix: &str, ext: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let seq = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "nextstat_m15_profiles_{}_{}_{}_{}.{}",
        std::process::id(),
        nanos,
        seq,
        suffix,
        ext
    ));
    p
}

fn run(args: &[&str]) -> Output {
    Command::new(bin_path())
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {:?} {:?}: {}", bin_path(), args, e))
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).expect("json output should exist"))
        .expect("json output should parse")
}

fn build_assessment_table(config: &Path) -> PathBuf {
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out = tmp_file_path("assessment_table", "json");
    let result = run(&[
        "m15",
        "assessment-table",
        "--config",
        config.to_string_lossy().as_ref(),
        "--validation-report",
        validation_report.to_string_lossy().as_ref(),
        "--pharma-validation",
        pharma_validation.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 assessment-table should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    out
}

fn render_assessment_table_markdown(config: &Path) -> PathBuf {
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out = tmp_file_path("assessment_table_markdown", "md");
    let result = run(&[
        "m15",
        "assessment-table",
        "--config",
        config.to_string_lossy().as_ref(),
        "--validation-report",
        validation_report.to_string_lossy().as_ref(),
        "--pharma-validation",
        pharma_validation.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--format",
        "markdown",
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 assessment-table markdown should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    out
}

fn build_map(config: &Path, assessment_table: &Path) -> PathBuf {
    let out = tmp_file_path("map", "json");
    let result = run(&[
        "m15",
        "map",
        "--config",
        config.to_string_lossy().as_ref(),
        "--assessment-table",
        assessment_table.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 map should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    out
}

fn render_map_markdown(config: &Path, assessment_table: &Path) -> PathBuf {
    let out = tmp_file_path("map_markdown", "md");
    let result = run(&[
        "m15",
        "map",
        "--config",
        config.to_string_lossy().as_ref(),
        "--assessment-table",
        assessment_table.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--format",
        "markdown",
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 map markdown should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    out
}

fn build_mar(map: &Path, assessment_table: &Path) -> PathBuf {
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out = tmp_file_path("mar", "json");
    let result = run(&[
        "m15",
        "mar",
        "--map",
        map.to_string_lossy().as_ref(),
        "--assessment-table",
        assessment_table.to_string_lossy().as_ref(),
        "--validation-report",
        validation_report.to_string_lossy().as_ref(),
        "--pharma-validation",
        pharma_validation.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 mar should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    out
}

fn render_mar_markdown(map: &Path, assessment_table: &Path) -> PathBuf {
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out = tmp_file_path("mar_markdown", "md");
    let result = run(&[
        "m15",
        "mar",
        "--map",
        map.to_string_lossy().as_ref(),
        "--assessment-table",
        assessment_table.to_string_lossy().as_ref(),
        "--validation-report",
        validation_report.to_string_lossy().as_ref(),
        "--pharma-validation",
        pharma_validation.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--format",
        "markdown",
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 mar markdown should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    out
}

fn build_bundle(config: &Path, assessment_table: &Path, map: &Path, mar: &Path) -> PathBuf {
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out = tmp_file_path("bundle", "json");
    let result = run(&[
        "m15",
        "bundle",
        "--config",
        config.to_string_lossy().as_ref(),
        "--assessment-table",
        assessment_table.to_string_lossy().as_ref(),
        "--map",
        map.to_string_lossy().as_ref(),
        "--mar",
        mar.to_string_lossy().as_ref(),
        "--validation-report",
        validation_report.to_string_lossy().as_ref(),
        "--pharma-validation",
        pharma_validation.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 bundle should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    out
}

fn build_profile_diff(config: &Path) -> PathBuf {
    let out = tmp_file_path("profile_diff", "json");
    let result = run(&[
        "m15",
        "profile-diff",
        "--config",
        config.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 profile-diff should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    out
}

#[test]
fn m15_generated_json_examples_match_docs_specs() {
    let config = doc_spec_path("m15_config_v1.example.json");
    let assessment_table = build_assessment_table(&config);
    let map = build_map(&config, &assessment_table);
    let mar = build_mar(&map, &assessment_table);
    let bundle = build_bundle(&config, &assessment_table, &map, &mar);

    let expected_assessment = read_json(&doc_spec_path("m15_assessment_table_v1.example.json"));
    let expected_map = read_json(&doc_spec_path("m15_map_v1.example.json"));
    let expected_mar = read_json(&doc_spec_path("m15_mar_v1.example.json"));
    let expected_profile_diff =
        read_json(&doc_spec_path("m15_profile_diff_report_v1.example.json"));
    let expected_bundle = read_json(&doc_spec_path("m15_bundle_manifest_v1.example.json"));
    let profile_diff = build_profile_diff(&config);

    assert_eq!(read_json(&assessment_table), expected_assessment);
    assert_eq!(read_json(&map), expected_map);
    assert_eq!(read_json(&mar), expected_mar);
    assert_eq!(read_json(&profile_diff), expected_profile_diff);
    assert_eq!(read_json(&bundle), expected_bundle);

    let _ = std::fs::remove_file(&assessment_table);
    let _ = std::fs::remove_file(&map);
    let _ = std::fs::remove_file(&mar);
    let _ = std::fs::remove_file(&profile_diff);
    let _ = std::fs::remove_file(&bundle);
}

#[test]
fn m15_jurisdiction_profile_matrix_round_trips_across_all_artifacts() {
    let base_config_path = doc_spec_path("m15_config_v1.example.json");
    let base_config: Value = serde_json::from_slice(
        &std::fs::read(&base_config_path).expect("config example should exist"),
    )
    .expect("config example should parse");

    for (profile, label, assessment_heading, map_heading, mar_heading) in [
        (
            "ich_core",
            "ICH Core",
            "ICH Core Assessment Framing",
            "ICH Core Planning Framing",
            "ICH Core Results Framing",
        ),
        (
            "ema_step5_2026",
            "EMA Step 5 (2026)",
            "EMA Step 5 Assessment Framing",
            "EMA Step 5 Planning Framing",
            "EMA Step 5 Results Framing",
        ),
        (
            "fda_draft_2024",
            "FDA Draft Guidance (2024)",
            "FDA Draft Guidance Assessment Framing",
            "FDA Draft Guidance Planning Framing",
            "FDA Draft Guidance Results Framing",
        ),
    ] {
        let mut config = base_config.clone();
        config["jurisdiction_profile"] = Value::String(profile.to_string());
        let config_path = tmp_file_path(profile, "json");
        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&config).expect("profile config should serialize"),
        )
        .expect("profile config should write");

        let assessment_table = build_assessment_table(&config_path);
        let map = build_map(&config_path, &assessment_table);
        let mar = build_mar(&map, &assessment_table);
        let bundle = build_bundle(&config_path, &assessment_table, &map, &mar);

        for path in [&assessment_table, &map, &mar, &bundle] {
            let artifact = read_json(path);
            assert_eq!(
                artifact.get("jurisdiction_profile").and_then(|v| v.as_str()),
                Some(profile),
                "artifact {} should preserve canonical profile id {}",
                path.display(),
                profile
            );
        }
        let assessment_artifact = read_json(&assessment_table);
        let map_artifact = read_json(&map);
        let mar_artifact = read_json(&mar);
        assert_eq!(
            assessment_artifact
                .pointer("/profile_requirements/profile_label")
                .and_then(|v| v.as_str()),
            Some(label)
        );
        assert_eq!(
            assessment_artifact
                .pointer("/profile_requirements/framing_heading")
                .and_then(|v| v.as_str()),
            Some(assessment_heading)
        );
        assert_eq!(
            map_artifact.pointer("/profile_requirements/framing_heading").and_then(|v| v.as_str()),
            Some(map_heading)
        );
        assert_eq!(
            mar_artifact.pointer("/profile_requirements/framing_heading").and_then(|v| v.as_str()),
            Some(mar_heading)
        );

        let _ = std::fs::remove_file(&config_path);
        let _ = std::fs::remove_file(&assessment_table);
        let _ = std::fs::remove_file(&map);
        let _ = std::fs::remove_file(&mar);
        let _ = std::fs::remove_file(&bundle);
    }
}

#[test]
fn m15_profile_specific_markdown_includes_required_sections_and_wording() {
    let base_config_path = doc_spec_path("m15_config_v1.example.json");
    let base_config: Value = serde_json::from_slice(
        &std::fs::read(&base_config_path).expect("config example should exist"),
    )
    .expect("config example should parse");

    for (profile, label, assessment_heading, map_heading, mar_heading) in [
        (
            "ich_core",
            "ICH Core",
            "## ICH Core Assessment Framing",
            "## ICH Core Planning Framing",
            "## ICH Core Results Framing",
        ),
        (
            "ema_step5_2026",
            "EMA Step 5 (2026)",
            "## EMA Step 5 Assessment Framing",
            "## EMA Step 5 Planning Framing",
            "## EMA Step 5 Results Framing",
        ),
        (
            "fda_draft_2024",
            "FDA Draft Guidance (2024)",
            "## FDA Draft Guidance Assessment Framing",
            "## FDA Draft Guidance Planning Framing",
            "## FDA Draft Guidance Results Framing",
        ),
    ] {
        let mut config = base_config.clone();
        config["jurisdiction_profile"] = Value::String(profile.to_string());
        let config_path = tmp_file_path(profile, "json");
        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&config).expect("profile config should serialize"),
        )
        .expect("profile config should write");

        let assessment_table = build_assessment_table(&config_path);
        let map = build_map(&config_path, &assessment_table);
        let assessment_md = std::fs::read_to_string(render_assessment_table_markdown(&config_path))
            .expect("assessment markdown should exist");
        let map_md = std::fs::read_to_string(render_map_markdown(&config_path, &assessment_table))
            .expect("map markdown should exist");
        let mar_md = std::fs::read_to_string(render_mar_markdown(&map, &assessment_table))
            .expect("mar markdown should exist");

        for markdown in [&assessment_md, &map_md, &mar_md] {
            assert!(markdown.contains("- Profile label: "));
            assert!(markdown.contains(label));
            assert!(markdown.contains("## Mandatory Sections"));
        }
        assert!(assessment_md.contains("## Questions of Interest"));
        assert!(assessment_md.contains(assessment_heading));
        assert!(map_md.contains(map_heading));
        assert!(mar_md.contains(mar_heading));

        let _ = std::fs::remove_file(&config_path);
        let _ = std::fs::remove_file(&assessment_table);
        let _ = std::fs::remove_file(&map);
    }
}
