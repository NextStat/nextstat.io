use ns_root::{CacheConfig, RootFile};
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

fn env_string(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn mbps(bytes: usize, secs: f64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0) / secs
}

#[test]
fn cache_reuse_branch_reader_avoids_additional_misses() {
    let path = fixture_path("simple_tree_zstd.root");
    if !path.exists() {
        eprintln!("Fixture not found: {:?}", path);
        return;
    }

    let mut f = RootFile::open(&path).expect("failed to open ROOT file");
    f.set_cache_config(CacheConfig { max_bytes: 64 * 1024 * 1024, enabled: true });
    let tree = f.get_tree("events").expect("failed to get tree 'events'");

    f.basket_cache().clear();
    let s0 = f.basket_cache().stats();

    let first = f.branch_data(&tree, "pt").expect("first branch read failed");
    assert_eq!(first.len(), tree.entries as usize);
    let s1 = f.basket_cache().stats();
    assert!(
        s1.misses > s0.misses,
        "expected cache misses on first read, got before={} after={}",
        s0.misses,
        s1.misses
    );

    let second = f.branch_data(&tree, "pt").expect("second branch read failed");
    assert_eq!(second.len(), first.len());
    let s2 = f.basket_cache().stats();

    assert_eq!(
        s2.misses, s1.misses,
        "second read should not add misses (cache should serve baskets)"
    );
    assert!(
        s2.hits > s1.hits,
        "second read should add cache hits, got before={} after={}",
        s1.hits,
        s2.hits
    );
}

#[test]
fn single_reader_reuses_loaded_baskets_without_extra_cache_lookups() {
    let path = fixture_path("simple_tree_zstd.root");
    if !path.exists() {
        eprintln!("Fixture not found: {:?}", path);
        return;
    }

    let mut f = RootFile::open(&path).expect("failed to open ROOT file");
    f.set_cache_config(CacheConfig { max_bytes: 64 * 1024 * 1024, enabled: true });
    let tree = f.get_tree("events").expect("failed to get tree 'events'");
    let reader = f.branch_reader(&tree, "njet").expect("failed to create branch reader");

    f.basket_cache().clear();
    let s0 = f.basket_cache().stats();

    let first = reader.as_f64().expect("first typed read failed");
    assert_eq!(first.len(), tree.entries as usize);
    let s1 = f.basket_cache().stats();
    assert!(
        s1.misses > s0.misses,
        "first read should fill cache (before={} after={})",
        s0.misses,
        s1.misses
    );

    let second = reader.as_i32().expect("second typed read failed");
    assert_eq!(second.len(), first.len());
    let s2 = f.basket_cache().stats();

    assert_eq!(
        s2.misses, s1.misses,
        "second read via same BranchReader should not add cache misses"
    );
    assert_eq!(
        s2.hits, s1.hits,
        "second read via same BranchReader should reuse already loaded basket refs"
    );

    let third = reader.as_f32().expect("third typed read failed");
    assert_eq!(third.len(), first.len());
    let s3 = f.basket_cache().stats();

    assert_eq!(
        s3.misses, s2.misses,
        "third read via same BranchReader should not add cache misses"
    );
    assert_eq!(
        s3.hits, s2.hits,
        "third read via same BranchReader should reuse already loaded basket refs"
    );
}

#[test]
#[ignore]
fn bench_branch_reader_cache_hot_vs_cold() {
    // Example:
    // NS_ROOT_BRANCH_ITERS=200 NS_ROOT_BRANCH=pt cargo test -p ns-root --test branch_reader_cache_bench -- --ignored --nocapture
    let path = fixture_path("simple_tree_zstd.root");
    if !path.exists() {
        eprintln!("Fixture not found: {:?}", path);
        return;
    }

    let iters = env_usize("NS_ROOT_BRANCH_ITERS", 200);
    let min_speedup = env_f64("NS_ROOT_BRANCH_MIN_SPEEDUP", 1.20);
    let branch = std::env::var("NS_ROOT_BRANCH").unwrap_or_else(|_| "pt".to_string());

    let mut f = RootFile::open(&path).expect("failed to open ROOT file");
    f.set_cache_config(CacheConfig { max_bytes: 64 * 1024 * 1024, enabled: true });
    let tree = f.get_tree("events").expect("failed to get tree 'events'");

    // Cold run: clear cache for each read.
    let mut cold_best = 0.0f64;
    let mut cold_total = 0.0f64;
    for _ in 0..iters {
        f.basket_cache().clear();
        let t0 = Instant::now();
        let out = f.branch_data(&tree, &branch).expect("cold branch read failed");
        let dt = t0.elapsed().as_secs_f64();
        let tput = mbps(out.len() * core::mem::size_of::<f64>(), dt);
        std::hint::black_box(out.len());
        cold_total += tput;
        if tput > cold_best {
            cold_best = tput;
        }
    }

    // Warm run: prefill cache once, then measure repeated hot reads.
    f.basket_cache().clear();
    let _warmup = f.branch_data(&tree, &branch).expect("warmup branch read failed");
    let s_warm_start = f.basket_cache().stats();

    let mut warm_best = 0.0f64;
    let mut warm_total = 0.0f64;
    for _ in 0..iters {
        let t0 = Instant::now();
        let out = f.branch_data(&tree, &branch).expect("warm branch read failed");
        let dt = t0.elapsed().as_secs_f64();
        let tput = mbps(out.len() * core::mem::size_of::<f64>(), dt);
        std::hint::black_box(out.len());
        warm_total += tput;
        if tput > warm_best {
            warm_best = tput;
        }
    }

    let s_warm_end = f.basket_cache().stats();
    eprintln!(
        "branch_reader_cache_bench branch={} iters={} cold(best/avg)={:.1}/{:.1} MiB/s warm(best/avg)={:.1}/{:.1} MiB/s speedup(avg)={:.2}x hits_delta={} misses_delta={} hit_rate={:.3}",
        branch,
        iters,
        cold_best,
        cold_total / (iters as f64),
        warm_best,
        warm_total / (iters as f64),
        (warm_total / cold_total).max(0.0),
        s_warm_end.hits.saturating_sub(s_warm_start.hits),
        s_warm_end.misses.saturating_sub(s_warm_start.misses),
        s_warm_end.hit_rate()
    );

    assert!(
        s_warm_end.hits > s_warm_start.hits,
        "expected warm loop to add cache hits (start={} end={})",
        s_warm_start.hits,
        s_warm_end.hits
    );
    assert!(
        warm_total >= cold_total * min_speedup,
        "expected warm avg speedup >= {:.2}x, got {:.2}x",
        min_speedup,
        warm_total / cold_total
    );
}

#[test]
#[ignore]
fn bench_branch_reader_cache_hot_vs_cold_external_root() {
    // Example:
    // NS_ROOT_BENCH_FILE=/path/to/large.root \
    // NS_ROOT_BENCH_TREE=Events \
    // NS_ROOT_BENCH_BRANCH=run \
    // NS_ROOT_BRANCH_ITERS=40 \
    // NS_ROOT_BRANCH_MIN_SPEEDUP=1.02 \
    // cargo test -p ns-root --test branch_reader_cache_bench -- --ignored --nocapture
    let file = match std::env::var("NS_ROOT_BENCH_FILE") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("NS_ROOT_BENCH_FILE is not set; skipping external ROOT benchmark");
            return;
        }
    };
    let path = PathBuf::from(file);
    if !path.exists() {
        eprintln!("External ROOT file not found: {:?}", path);
        return;
    }

    let tree_name = env_string("NS_ROOT_BENCH_TREE", "Events");
    let branch = env_string("NS_ROOT_BENCH_BRANCH", "run");
    let iters = env_usize("NS_ROOT_BRANCH_ITERS", 40);
    let min_speedup = env_f64("NS_ROOT_BRANCH_MIN_SPEEDUP", 1.02);

    let mut f = RootFile::open(&path).expect("failed to open external ROOT file");
    f.set_cache_config(CacheConfig { max_bytes: 1024 * 1024 * 1024, enabled: true });
    let tree = f
        .get_tree(&tree_name)
        .unwrap_or_else(|e| panic!("failed to get tree '{tree_name}' from {:?}: {e}", path));
    if tree.find_branch(&branch).is_none() {
        let names = tree.branch_names();
        let preview = names.iter().take(40).copied().collect::<Vec<_>>().join(", ");
        panic!(
            "branch '{}' not found in tree '{}'. first {} branches: {}",
            branch,
            tree_name,
            names.len().min(40),
            preview
        );
    }

    let mut cold_best = 0.0f64;
    let mut cold_total = 0.0f64;
    let mut warm_best = 0.0f64;
    let mut warm_total = 0.0f64;

    for _ in 0..iters {
        f.basket_cache().clear();
        let t0 = Instant::now();
        let out = f.branch_data(&tree, &branch).expect("cold external branch read failed");
        let dt = t0.elapsed().as_secs_f64();
        let tput = mbps(out.len() * core::mem::size_of::<f64>(), dt);
        std::hint::black_box(out.len());
        cold_total += tput;
        cold_best = cold_best.max(tput);
    }

    f.basket_cache().clear();
    let _warmup = f.branch_data(&tree, &branch).expect("warmup external branch read failed");
    let s_warm_start = f.basket_cache().stats();

    for _ in 0..iters {
        let t0 = Instant::now();
        let out = f.branch_data(&tree, &branch).expect("warm external branch read failed");
        let dt = t0.elapsed().as_secs_f64();
        let tput = mbps(out.len() * core::mem::size_of::<f64>(), dt);
        std::hint::black_box(out.len());
        warm_total += tput;
        warm_best = warm_best.max(tput);
    }

    let s_warm_end = f.basket_cache().stats();
    let speedup = if cold_total > 0.0 { warm_total / cold_total } else { 0.0 };
    eprintln!(
        "branch_reader_external_bench file={:?} tree={} branch={} iters={} cold(best/avg)={:.1}/{:.1} MiB/s warm(best/avg)={:.1}/{:.1} MiB/s speedup(avg)={:.2}x hits_delta={} misses_delta={} hit_rate={:.3}",
        path,
        tree_name,
        branch,
        iters,
        cold_best,
        cold_total / (iters as f64),
        warm_best,
        warm_total / (iters as f64),
        speedup,
        s_warm_end.hits.saturating_sub(s_warm_start.hits),
        s_warm_end.misses.saturating_sub(s_warm_start.misses),
        s_warm_end.hit_rate()
    );

    assert!(
        s_warm_end.hits > s_warm_start.hits,
        "expected warm external loop to add cache hits (start={} end={})",
        s_warm_start.hits,
        s_warm_end.hits
    );
    assert!(
        speedup >= min_speedup,
        "expected warm external avg speedup >= {:.2}x, got {:.2}x",
        min_speedup,
        speedup
    );
}
