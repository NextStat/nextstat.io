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

fn read_bytes(path: &PathBuf) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {}", path.display(), e))
}

#[test]
fn report_deterministic_outputs_are_byte_identical() {
    let ws = fixture_path("histfactory/workspace.json");
    let xml = fixture_path("histfactory/combination.xml");
    assert!(ws.exists(), "missing fixture: {}", ws.display());
    assert!(xml.exists(), "missing fixture: {}", xml.display());

    let out_dir_a = tmp_dir("report_det_a");
    let out_dir_b = tmp_dir("report_det_b");

    for out_dir in [&out_dir_a, &out_dir_b] {
        let out = run(&[
            "report",
            "--input",
            ws.to_string_lossy().as_ref(),
            "--histfactory-xml",
            xml.to_string_lossy().as_ref(),
            "--out-dir",
            out_dir.to_string_lossy().as_ref(),
            "--threads",
            "1",
            "--deterministic",
            "--skip-uncertainty",
        ]);
        assert!(
            out.status.success(),
            "report should succeed, stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    for name in ["fit.json", "distributions.json", "yields.json", "pulls.json", "corr.json"] {
        let a = out_dir_a.join(name);
        let b = out_dir_b.join(name);
        assert!(a.exists(), "missing {}: {}", name, a.display());
        assert!(b.exists(), "missing {}: {}", name, b.display());
        assert_eq!(read_bytes(&a), read_bytes(&b), "deterministic output differs for {}", name);
    }

    let _ = std::fs::remove_dir_all(&out_dir_a);
    let _ = std::fs::remove_dir_all(&out_dir_b);
}
