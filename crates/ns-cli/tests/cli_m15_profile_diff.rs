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

fn doc_spec_path(name: &str) -> PathBuf {
    repo_root().join("docs/specs").join(name)
}

fn tmp_file_path(suffix: &str, ext: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let seq = NEXT_TMP_ID.fetch_add(1, Ordering::Relaxed);
    let mut p = std::env::temp_dir();
    p.push(format!(
        "nextstat_m15_profile_diff_{}_{}_{}_{}.{}",
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
fn m15_profile_diff_writes_deterministic_json() {
    let config = doc_spec_path("m15_config_v1.example.json");
    let out1 = tmp_file_path("one", "json");
    let out2 = tmp_file_path("two", "json");

    for out in [&out1, &out2] {
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
    }

    let bytes1 = std::fs::read(&out1).expect("first JSON output should exist");
    let bytes2 = std::fs::read(&out2).expect("second JSON output should exist");
    assert_eq!(bytes1, bytes2, "deterministic outputs must be byte-identical");

    let v: serde_json::Value = serde_json::from_slice(&bytes1).expect("output should be JSON");
    assert_eq!(
        v.get("schema_version").and_then(|x| x.as_str()),
        Some("m15_profile_diff_report_v1")
    );
    assert_eq!(v.get("deterministic").and_then(|x| x.as_bool()), Some(true));
    assert!(v.get("generated_at").map(|x| x.is_null()).unwrap_or(false));
    assert_eq!(v.get("selected_profile").and_then(|x| x.as_str()), Some("ich_core"));
    assert_eq!(
        v.pointer("/documents/0/document_kind").and_then(|x| x.as_str()),
        Some("assessment_table")
    );
    assert_eq!(
        v.pointer("/documents/0/section_presence/0/section_name").and_then(|x| x.as_str()),
        Some("Questions of Interest")
    );
    assert_eq!(
        v.pointer("/documents/0/profile_views/1/profile_label").and_then(|x| x.as_str()),
        Some("EMA Step 5 (2026)")
    );
    assert_eq!(
        v.pointer("/documents/0/profile_views/2/profile_only_sections/0").and_then(|x| x.as_str()),
        Some("FDA Draft Guidance Assessment Framing")
    );

    let _ = std::fs::remove_file(&out1);
    let _ = std::fs::remove_file(&out2);
}

#[test]
fn m15_profile_diff_can_render_markdown() {
    let config = doc_spec_path("m15_config_v1.example.json");
    let out = tmp_file_path("markdown", "md");

    let result = run(&[
        "m15",
        "profile-diff",
        "--config",
        config.to_string_lossy().as_ref(),
        "--output",
        out.to_string_lossy().as_ref(),
        "--format",
        "markdown",
        "--deterministic",
    ]);
    assert!(
        result.status.success(),
        "m15 profile-diff markdown should succeed, stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    let md = std::fs::read_to_string(&out).expect("markdown output should exist");
    assert!(md.contains("# ICH M15 Profile Diff Report"));
    assert!(md.contains("## assessment_table"));
    assert!(md.contains("### ICH Core (ich_core)"));
    assert!(md.contains("Profile-only mandatory sections"));
    assert!(md.contains("FDA Draft Guidance Assessment Framing"));

    let _ = std::fs::remove_file(&out);
}
