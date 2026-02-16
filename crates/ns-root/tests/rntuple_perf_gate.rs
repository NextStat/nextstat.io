use ns_root::RootFile;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures").join(name)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|s| s.parse::<f64>().ok()).unwrap_or(default)
}

fn env_cases_var(var_name: &str, default: &[&str]) -> Vec<String> {
    let Some(raw) = std::env::var(var_name).ok() else {
        return default.iter().map(|s| (*s).to_string()).collect();
    };
    let parsed = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if parsed.is_empty() { default.iter().map(|s| (*s).to_string()).collect() } else { parsed }
}

fn case_to_path(case: &str) -> PathBuf {
    if case.contains('/') || Path::new(case).is_absolute() {
        PathBuf::from(case)
    } else {
        fixture_path(case)
    }
}

fn run_perf_gate(cases: Vec<String>, iters: usize, max_avg_ms: f64, min_suites_per_sec: f64) {
    let mut files = Vec::with_capacity(cases.len());
    let mut total_entries = 0usize;
    for case in &cases {
        let path = case_to_path(case);
        assert!(path.exists(), "missing fixture: {}", path.display());
        let f = RootFile::open(&path).unwrap_or_else(|e| panic!("failed to open {case}: {e}"));
        let warmup = f
            .read_rntuple_decoded_columns_all_clusters_f64("Events")
            .unwrap_or_else(|e| panic!("warmup decode should succeed for {case}: {e}"));
        let entries = warmup
            .iter()
            .map(|cg| usize::try_from(cg.entry_span).expect("entry_span should fit usize"))
            .sum::<usize>();
        assert!(entries > 0, "fixture {} decoded zero entries", case);
        total_entries += entries;
        files.push((case.to_string(), f, entries));
    }

    let mut samples_ms = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let mut iter_entries = 0usize;
        for (case, f, expected_entries) in &files {
            let decoded = f
                .read_rntuple_decoded_columns_all_clusters_f64("Events")
                .unwrap_or_else(|e| panic!("benchmark decode should succeed for {}: {}", case, e));
            let entries = decoded
                .iter()
                .map(|cg| usize::try_from(cg.entry_span).expect("entry_span should fit usize"))
                .sum::<usize>();
            assert_eq!(
                entries, *expected_entries,
                "decoded entry count drifted during benchmark for {}",
                case
            );
            iter_entries += entries;
        }
        let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(iter_entries);
        samples_ms.push(dt_ms);
    }

    samples_ms.sort_by(|a, b| a.total_cmp(b));
    let avg_ms = samples_ms.iter().sum::<f64>() / (iters as f64);
    let p95_idx = ((iters.saturating_sub(1)) * 95) / 100;
    let p95_ms = samples_ms[p95_idx];
    let suites_per_sec = 1000.0 / avg_ms;
    let entries_per_sec = (total_entries as f64) / (avg_ms / 1000.0);

    eprintln!(
        "rntuple_perf_gate suite_size={} iters={} total_entries={} avg_ms={:.3} p95_ms={:.3} suites_per_sec={:.2} entries_per_sec={:.1}",
        files.len(),
        iters,
        total_entries,
        avg_ms,
        p95_ms,
        suites_per_sec,
        entries_per_sec
    );

    assert!(
        avg_ms <= max_avg_ms,
        "RNTuple decode perf gate failed: avg_ms={:.3} > max_avg_ms={:.3}",
        avg_ms,
        max_avg_ms
    );
    assert!(
        suites_per_sec >= min_suites_per_sec,
        "RNTuple decode perf gate failed: suites_per_sec={:.2} < min_suites_per_sec={:.2}",
        suites_per_sec,
        min_suites_per_sec
    );
}

#[test]
#[ignore]
fn rntuple_decode_perf_gate_baseline() {
    // Example:
    // NS_ROOT_RNTUPLE_PERF_ITERS=20 NS_ROOT_RNTUPLE_PERF_MAX_AVG_MS=120 NS_ROOT_RNTUPLE_PERF_MIN_SUITES_PER_SEC=8 cargo test -p ns-root --test rntuple_perf_gate rntuple_decode_perf_gate_baseline -- --ignored --nocapture
    let default_cases = [
        "rntuple_simple.root",
        "rntuple_complex.root",
        "rntuple_multicluster.root",
        "rntuple_schema_evolution.root",
        "rntuple_pair_scalar_variable.root",
        "rntuple_pair_variable_scalar.root",
        "rntuple_pair_variable_variable.root",
    ];
    let cases = env_cases_var("NS_ROOT_RNTUPLE_PERF_CASES", &default_cases);
    let iters = env_usize("NS_ROOT_RNTUPLE_PERF_ITERS", 10);
    let max_avg_ms = env_f64("NS_ROOT_RNTUPLE_PERF_MAX_AVG_MS", 120.0);
    let min_suites_per_sec = env_f64("NS_ROOT_RNTUPLE_PERF_MIN_SUITES_PER_SEC", 8.0);
    run_perf_gate(cases, iters, max_avg_ms, min_suites_per_sec);
}

#[test]
#[ignore]
fn rntuple_decode_perf_gate_large_mixed_optional() {
    // Example:
    // NS_ROOT_RNTUPLE_PERF_LARGE_MIXED_CASES=/abs/path/to/rntuple_bench_large.root cargo test -p ns-root --test rntuple_perf_gate rntuple_decode_perf_gate_large_mixed_optional --release -- --ignored --nocapture
    let default_cases = ["rntuple_bench_large.root"];
    let cases = env_cases_var("NS_ROOT_RNTUPLE_PERF_LARGE_MIXED_CASES", &default_cases);
    let iters = env_usize("NS_ROOT_RNTUPLE_PERF_LARGE_MIXED_ITERS", 5);
    let max_avg_ms = env_f64("NS_ROOT_RNTUPLE_PERF_LARGE_MIXED_MAX_AVG_MS", 200.0);
    let min_suites_per_sec = env_f64("NS_ROOT_RNTUPLE_PERF_LARGE_MIXED_MIN_SUITES_PER_SEC", 5.0);
    run_perf_gate(cases, iters, max_avg_ms, min_suites_per_sec);
}
