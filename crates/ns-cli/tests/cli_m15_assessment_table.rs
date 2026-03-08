use std::path::PathBuf;
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
        "nextstat_m15_assessment_table_{}_{}_{}_{}.{}",
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

#[test]
fn m15_assessment_table_writes_deterministic_json() {
    let config = doc_spec_path("m15_config_v1.example.json");
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out1 = tmp_file_path("one", "json");
    let out2 = tmp_file_path("two", "json");

    for out in [&out1, &out2] {
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
    }

    let bytes1 = std::fs::read(&out1).expect("first JSON output should exist");
    let bytes2 = std::fs::read(&out2).expect("second JSON output should exist");
    assert_eq!(bytes1, bytes2, "deterministic outputs must be byte-identical");

    let v: serde_json::Value = serde_json::from_slice(&bytes1).expect("output should be JSON");
    assert_eq!(v.get("schema_version").and_then(|x| x.as_str()), Some("m15_assessment_table_v1"));
    assert_eq!(v.get("deterministic").and_then(|x| x.as_bool()), Some(true));
    assert!(v.get("generated_at").map(|x| x.is_null()).unwrap_or(false));
    assert_eq!(
        v.pointer("/profile_requirements/profile_label").and_then(|x| x.as_str()),
        Some("ICH Core")
    );
    assert_eq!(
        v.pointer("/profile_requirements/framing_heading").and_then(|x| x.as_str()),
        Some("ICH Core Assessment Framing")
    );
    assert_eq!(v.pointer("/summary/n_entries").and_then(|x| x.as_u64()), Some(2));
    assert_eq!(v.pointer("/summary/unresolved_items").and_then(|x| x.as_u64()), Some(0));
    assert_eq!(
        v.pointer("/entries/0/evidence_refs/0/path").and_then(|x| x.as_str()),
        Some("validation_report.json")
    );
    assert_eq!(
        v.pointer("/entries/0/evidence_refs/1/path").and_then(|x| x.as_str()),
        Some("pharma_validation.json")
    );

    let _ = std::fs::remove_file(&out1);
    let _ = std::fs::remove_file(&out2);
}

#[test]
fn m15_assessment_table_can_render_markdown() {
    let config = doc_spec_path("m15_config_v1.example.json");
    let validation_report = doc_spec_path("validation_report_v1.example.json");
    let pharma_validation = fixture_path("pharma_validation_ok.json");
    let out = tmp_file_path("markdown", "md");

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

    let md = std::fs::read_to_string(&out).expect("markdown output should exist");
    assert!(md.contains("# ICH M15 Assessment Table"));
    assert!(md.contains("QOI-001"));
    assert!(md.contains("Recommended reporting level: full"));

    let _ = std::fs::remove_file(&out);
}
