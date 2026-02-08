use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ns_root::{KeyInfo, RootFile};

const BUF_SIZE: usize = 64 * 1024;

fn main() -> Result<(), io::Error> {
    let cfg = Config::parse()?;

    let file = RootFile::open(&cfg.root_path)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("failed to open root file: {e}")))?;

    let bytes = file.file_data();
    let _is_large = file.is_large();

    if cfg.bench_scan {
        let stats = bench_scan_root_for_zs_blocks(bytes, cfg.verbose, cfg.fail_fast)?;
        println!(
            "Bench(scan): zs_blocks_decompressed={}, candidates_skipped={}, failures={}, in_bytes={}, out_bytes={}, seconds={:.6}, blocks_per_s={:.3}, out_mib_per_s={:.3}",
            stats.zs_blocks_decompressed,
            stats.candidates_skipped,
            stats.failures,
            stats.total_in_bytes,
            stats.total_out_bytes,
            stats.seconds,
            stats.blocks_per_s,
            stats.out_mib_per_s,
        );
        if stats.failures == 0 {
            return Ok(());
        }

        return Err(io::Error::new(io::ErrorKind::Other, "root_accuracy_gate bench failed"));
    }

#[derive(Copy, Clone, Debug, Default)]
struct BenchCompareStats {
    branches_checked: usize,
    baskets_checked: usize,
    baskets_uncompressed_skipped: usize,
    baskets_non_zs_skipped: usize,
    failures: usize,
    total_out_bytes: u64,
    new_seconds: f64,
    old_seconds: f64,
    new_out_mib_per_s: f64,
    old_out_mib_per_s: f64,
    speedup_x: f64,
}

#[derive(Copy, Clone, Debug)]
struct BasketJob {
    start: usize,
    end: usize,
    expected_len: usize,
}

fn basket_is_zs_only(src: &[u8], expected_len: usize) -> Result<bool, String> {
    let mut offset = 0usize;
    let mut total_uncompressed = 0usize;

    while total_uncompressed < expected_len {
        if offset + 9 > src.len() {
            return Err(format!(
                "truncated ROOT compression block header at payload offset {} (need 9 bytes)",
                offset
            ));
        }

        let tag = &src[offset..offset + 2];
        let c_size = read_le24(&src[offset + 3..offset + 6]);
        let u_size = read_le24(&src[offset + 6..offset + 9]);
        offset += 9;

        let end = offset + c_size;
        if end > src.len() {
            return Err(format!(
                "compressed sub-block claims {} bytes but only {} remain",
                c_size,
                src.len() - offset
            ));
        }

        if tag != b"ZS" {
            return Ok(false);
        }

        total_uncompressed += u_size;
        offset = end;
    }

    if total_uncompressed != expected_len {
        return Err(format!(
            "total uncompressed bytes {} != expected {}",
            total_uncompressed, expected_len
        ));
    }

    Ok(true)
}

fn decompress_root_zs_only_legacy(src: &[u8], expected_len: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(expected_len);

    let mut offset = 0usize;
    let mut total_uncompressed = 0usize;

    while total_uncompressed < expected_len {
        if offset + 9 > src.len() {
            return Err(format!(
                "truncated ROOT compression block header at payload offset {} (need 9 bytes)",
                offset
            ));
        }

        let tag = &src[offset..offset + 2];
        let c_size = read_le24(&src[offset + 3..offset + 6]);
        let u_size = read_le24(&src[offset + 6..offset + 9]);
        offset += 9;

        let end = offset + c_size;
        if end > src.len() {
            return Err(format!(
                "compressed sub-block claims {} bytes but only {} remain",
                c_size,
                src.len() - offset
            ));
        }

        if tag != b"ZS" {
            return Err(format!(
                "non-ZS sub-block encountered in legacy ZS-only decompressor: {:?}",
                std::str::from_utf8(tag).unwrap_or("??")
            ));
        }

        let compressed = &src[offset..end];

        let mut ruz_source = compressed;
        let mut ruz_dec = ruzstd::decoding::StreamingDecoder::new(&mut ruz_source)
            .map_err(|e| format!("ruzstd init failed: {e}"))?;
        let mut block_out = Vec::with_capacity(u_size);
        ruz_dec
            .read_to_end(&mut block_out)
            .map_err(|e| format!("ruzstd read failed: {e}"))?;

        if block_out.len() != u_size {
            return Err(format!("decoded size mismatch: got {} expected {}", block_out.len(), u_size));
        }

        out.extend_from_slice(&block_out);

        total_uncompressed += u_size;
        offset = end;
    }

    if out.len() != expected_len {
        return Err(format!(
            "total decompressed length {} != expected {}",
            out.len(),
            expected_len
        ));
    }

    Ok(out)
}

fn bench_compare_tree_baskets(
    bytes: &[u8],
    root_path: &Path,
    tree_name: &str,
    branches: &[ns_root::BranchInfo],
    max_baskets_per_branch: usize,
    fail_fast: bool,
) -> Result<BenchCompareStats, io::Error> {
    let mut stats = BenchCompareStats::default();
    let mut jobs: Vec<BasketJob> = Vec::new();

    for branch in branches {
        stats.branches_checked += 1;
        let n = branch.n_baskets.min(max_baskets_per_branch);
        for i in 0..n {
            let seek = branch.basket_seek[i];
            let (obj_len, key_len, n_bytes) = read_tkey_header_fast(bytes, seek as usize)?;

            let key_end = seek as usize + n_bytes;
            let obj_start = seek as usize + key_len;
            if key_end > bytes.len() || obj_start > key_end {
                stats.failures += 1;
                eprintln!(
                    "FAIL(bench-compare) root={} tree={} branch={} basket={} seek={} : key slice out of bounds",
                    root_path.display(),
                    tree_name,
                    branch.name,
                    i,
                    seek
                );
                if fail_fast {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "root_accuracy_gate bench failed",
                    ));
                }
                continue;
            }

            let compressed_len = n_bytes - key_len;
            if obj_len == compressed_len {
                stats.baskets_uncompressed_skipped += 1;
                continue;
            }

            let compressed_data = &bytes[obj_start..key_end];
            match basket_is_zs_only(compressed_data, obj_len) {
                Ok(true) => {
                    jobs.push(BasketJob {
                        start: obj_start,
                        end: key_end,
                        expected_len: obj_len,
                    });
                    stats.total_out_bytes += obj_len as u64;
                }
                Ok(false) => {
                    stats.baskets_non_zs_skipped += 1;
                }
                Err(e) => {
                    stats.failures += 1;
                    eprintln!(
                        "FAIL(bench-compare) root={} tree={} branch={} basket={} seek={} : {}",
                        root_path.display(),
                        tree_name,
                        branch.name,
                        i,
                        seek,
                        e
                    );
                    if fail_fast {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "root_accuracy_gate bench failed",
                        ));
                    }
                }
            }
        }
    }

    if let Some(first) = jobs.first() {
        let slice = &bytes[first.start..first.end];
        let new_out = ns_root::decompress::decompress(slice, first.expected_len).map_err(|e| {
            io::Error::new(io::ErrorKind::Other, format!("bench-compare new decompress failed: {e}"))
        })?;
        let old_out = decompress_root_zs_only_legacy(slice, first.expected_len)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("bench-compare legacy failed: {e}")))?;
        if new_out != old_out {
            return Err(io::Error::new(
                io::ErrorKind::Other,
                "bench-compare mismatch between new and legacy outputs",
            ));
        }
    }

    let start_new = Instant::now();
    for job in &jobs {
        let slice = &bytes[job.start..job.end];
        match ns_root::decompress::decompress(slice, job.expected_len) {
            Ok(out) => {
                if out.len() != job.expected_len {
                    stats.failures += 1;
                    if fail_fast {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "root_accuracy_gate bench failed",
                        ));
                    }
                }
            }
            Err(e) => {
                stats.failures += 1;
                if fail_fast {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("root_accuracy_gate bench failed: {e}"),
                    ));
                }
            }
        }
    }
    stats.new_seconds = start_new.elapsed().as_secs_f64();

    let start_old = Instant::now();
    for job in &jobs {
        let slice = &bytes[job.start..job.end];
        match decompress_root_zs_only_legacy(slice, job.expected_len) {
            Ok(out) => {
                if out.len() != job.expected_len {
                    stats.failures += 1;
                    if fail_fast {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "root_accuracy_gate bench failed",
                        ));
                    }
                }
            }
            Err(e) => {
                stats.failures += 1;
                if fail_fast {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("root_accuracy_gate bench failed: {e}"),
                    ));
                }
            }
        }
    }
    stats.old_seconds = start_old.elapsed().as_secs_f64();

    stats.baskets_checked = jobs.len();

    if stats.new_seconds > 0.0 {
        stats.new_out_mib_per_s = (stats.total_out_bytes as f64 / (1024.0 * 1024.0)) / stats.new_seconds;
    }
    if stats.old_seconds > 0.0 {
        stats.old_out_mib_per_s = (stats.total_out_bytes as f64 / (1024.0 * 1024.0)) / stats.old_seconds;
    }
    if stats.new_seconds > 0.0 {
        stats.speedup_x = stats.old_seconds / stats.new_seconds;
    }

    Ok(stats)
}

    if cfg.bench_compare {
        let tree_name = match cfg.tree_name {
            Some(t) => t,
            None => select_first_tree_name(&file)?,
        };

        let tree = file.get_tree(&tree_name).map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("failed to read tree '{tree_name}': {e} (try --bench-scan)"),
            )
        })?;

        let stats = bench_compare_tree_baskets(
            bytes,
            &cfg.root_path,
            &tree.name,
            &tree.branches,
            cfg.max_baskets_per_branch.unwrap_or(usize::MAX),
            cfg.fail_fast,
        )?;

        println!(
            "Bench(compare): branches_checked={}, baskets_checked={}, baskets_uncompressed_skipped={}, baskets_non_zs_skipped={}, failures={}, out_bytes={}, new_seconds={:.6}, new_out_mib_per_s={:.3}, old_seconds={:.6}, old_out_mib_per_s={:.3}, speedup_x={:.3}",
            stats.branches_checked,
            stats.baskets_checked,
            stats.baskets_uncompressed_skipped,
            stats.baskets_non_zs_skipped,
            stats.failures,
            stats.total_out_bytes,
            stats.new_seconds,
            stats.new_out_mib_per_s,
            stats.old_seconds,
            stats.old_out_mib_per_s,
            stats.speedup_x,
        );

        if stats.failures == 0 {
            return Ok(());
        }
        return Err(io::Error::new(io::ErrorKind::Other, "root_accuracy_gate bench failed"));
    }

    if cfg.bench_tree {
        let tree_name = match cfg.tree_name {
            Some(t) => t,
            None => select_first_tree_name(&file)?,
        };

        let tree = file.get_tree(&tree_name).map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("failed to read tree '{tree_name}': {e} (try --bench-scan)"),
            )
        })?;

        let stats = bench_tree_baskets(
            bytes,
            &cfg.root_path,
            &tree.name,
            &tree.branches,
            cfg.max_baskets_per_branch.unwrap_or(usize::MAX),
            cfg.fail_fast,
        )?;

        println!(
            "Bench(tree): branches_checked={}, baskets_checked={}, baskets_uncompressed_skipped={}, failures={}, in_bytes={}, out_bytes={}, seconds={:.6}, baskets_per_s={:.3}, out_mib_per_s={:.3}",
            stats.branches_checked,
            stats.baskets_checked,
            stats.baskets_uncompressed_skipped,
            stats.failures,
            stats.total_in_bytes,
            stats.total_out_bytes,
            stats.seconds,
            stats.baskets_per_s,
            stats.out_mib_per_s,
        );

        if stats.failures == 0 {
            return Ok(());
        }
        return Err(io::Error::new(io::ErrorKind::Other, "root_accuracy_gate bench failed"));
    }

    if cfg.scan {
        let stats = scan_root_for_zs_blocks(bytes, cfg.verbose, cfg.fail_fast)?;
        println!(
            "Summary(scan): zs_blocks_checked={}, candidates_skipped={}, failures={}",
            stats.zs_blocks_checked, stats.candidates_skipped, stats.failures
        );
        if stats.failures == 0 {
            return Ok(());
        }
        return Err(io::Error::new(io::ErrorKind::Other, "root_accuracy_gate failed"));
    }

    let tree_name = match cfg.tree_name {
        Some(t) => t,
        None => select_first_tree_name(&file)?,
    };

    let tree = file
        .get_tree(&tree_name)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("failed to read tree '{tree_name}': {e} (try --scan)")))?;

    let max_baskets_per_branch = cfg.max_baskets_per_branch.unwrap_or(usize::MAX);

    let mut branches_checked = 0usize;
    let mut baskets_checked = 0usize;
    let mut baskets_skipped_uncompressed = 0usize;
    let mut blocks_checked = 0usize;
    let mut blocks_skipped_non_zs = 0usize;
    let mut failures = 0usize;

    for branch in &tree.branches {
        branches_checked += 1;
        let n = branch.n_baskets.min(max_baskets_per_branch);
        for i in 0..n {
            let seek = branch.basket_seek[i];

            let (obj_len, key_len, n_bytes) = read_tkey_header_fast(bytes, seek as usize)?;

            let key_end = seek as usize + n_bytes;
            let obj_start = seek as usize + key_len;
            if key_end > bytes.len() || obj_start > key_end {
                failures += 1;
                eprintln!(
                    "FAIL branch='{}' basket={} seek={} : key slice out of bounds",
                    branch.name, i, seek
                );
                if cfg.fail_fast {
                    return Err(io::Error::new(io::ErrorKind::Other, "root_accuracy_gate failed"));
                }
                continue;
            }

            let compressed_data = &bytes[obj_start..key_end];
            let compressed_len = n_bytes - key_len;

            if obj_len == compressed_len {
                baskets_skipped_uncompressed += 1;
                continue;
            }

            let ctx = Ctx {
                root_path: &cfg.root_path,
                tree_name: &tree.name,
                branch_name: &branch.name,
                basket_index: i,
                basket_seek: seek,
            };

            match verify_root_blocks_zs_only(compressed_data, obj_len, &ctx, cfg.verbose) {
                Ok(stats) => {
                    baskets_checked += 1;
                    blocks_checked += stats.blocks_checked;
                    blocks_skipped_non_zs += stats.blocks_skipped_non_zs;
                }
                Err(e) => {
                    failures += 1;
                    eprintln!(
                        "FAIL branch='{}' basket={} seek={} : {}",
                        branch.name, i, seek, e
                    );
                    if cfg.fail_fast {
                        return Err(io::Error::new(io::ErrorKind::Other, "root_accuracy_gate failed"));
                    }
                }
            }
        }
    }

    println!(
        "Summary: branches_checked={}, baskets_checked={}, baskets_uncompressed_skipped={}, zstd_blocks_checked={}, non_zs_blocks_skipped={}, failures={}",
        branches_checked,
        baskets_checked,
        baskets_skipped_uncompressed,
        blocks_checked,
        blocks_skipped_non_zs,
        failures
    );

    if failures == 0 {
        Ok(())
    } else {
        Err(io::Error::new(io::ErrorKind::Other, "root_accuracy_gate failed"))
    }
}

#[derive(Clone, Debug)]
struct Config {
    root_path: PathBuf,
    tree_name: Option<String>,
    max_baskets_per_branch: Option<usize>,
    fail_fast: bool,
    verbose: bool,
    scan: bool,
    bench_scan: bool,
    bench_compare: bool,
    bench_tree: bool,
}

impl Config {
    fn parse() -> Result<Self, io::Error> {
        let mut args = std::env::args_os();
        let _exe = args.next();

        let mut root_path: Option<PathBuf> = None;
        let mut tree_name: Option<String> = None;
        let mut max_baskets_per_branch: Option<usize> = Some(32);
        let mut fail_fast = false;
        let mut verbose = false;
        let mut scan = false;
        let mut bench_scan = false;
        let mut bench_compare = false;
        let mut bench_tree = false;

        while let Some(arg) = args.next() {
            match arg.to_string_lossy().as_ref() {
                "--root" => {
                    let Some(v) = args.next() else {
                        return Err(io::Error::new(io::ErrorKind::InvalidInput, "--root requires a path"));
                    };
                    root_path = Some(PathBuf::from(v));
                }
                "--tree" => {
                    let Some(v) = args.next() else {
                        return Err(io::Error::new(io::ErrorKind::InvalidInput, "--tree requires a value"));
                    };
                    tree_name = Some(v.to_string_lossy().to_string());
                }
                "--max-baskets-per-branch" => {
                    let Some(v) = args.next() else {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "--max-baskets-per-branch requires a value",
                        ));
                    };
                    let s = v.to_string_lossy();
                    let n: usize = s.parse().map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!("invalid --max-baskets-per-branch: {s}"),
                        )
                    })?;
                    max_baskets_per_branch = if n == 0 { None } else { Some(n) };
                }
                "--fail-fast" => fail_fast = true,
                "--scan" => scan = true,
                "--bench-scan" => bench_scan = true,
                "--bench-compare" => bench_compare = true,
                "--bench-tree" => bench_tree = true,
                "--verbose" | "-v" => verbose = true,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                s if s.starts_with('-') => {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("unknown flag: {s}")));
                }
                _ => {
                    if root_path.is_none() {
                        root_path = Some(PathBuf::from(arg));
                    } else {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "unexpected positional argument (did you mean --tree <name>?)",
                        ));
                    }
                }
            }
        }

        let root_path = match root_path {
            Some(p) => p,
            None => {
                print_help();
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "missing --root <file.root>"));
            }
        };

        Ok(Self {
            root_path,
            tree_name,
            max_baskets_per_branch,
            fail_fast,
            verbose,
            scan,
            bench_scan,
            bench_compare,
            bench_tree,
        })
    }
}

fn print_help() {
    eprintln!(
        "Usage:\n  root_accuracy_gate --root <file.root> [--tree <tree>] [--max-baskets-per-branch N] [--scan] [--bench-scan] [--bench-compare] [--bench-tree] [--fail-fast] [--verbose]\n\nBehavior:\n  Default mode (tree verify):\n    - Iterates all branches in the tree (default: first TTree in top-level keys)\n    - Iterates up to N baskets per branch (default: 32; N=0 means all)\n    - For each ROOT compression sub-block tagged 'ZS': decompress with ruzstd and reference libzstd (zstd crate)\n    - Compare output byte-for-byte\n\n  Scan mode (--scan):\n    - Scans raw file bytes for plausible ROOT 'ZS' sub-block headers and validates by successful libzstd decompression\n    - Compares ruzstd vs libzstd for every validated block\n\n  Bench mode (--bench-scan):\n    - Scans raw file bytes for plausible ROOT 'ZS' sub-block headers\n    - Decompresses using ns_root production decompressor (no reference, no byte-compare)\n    - Prints throughput stats\n\n  Bench mode (--bench-compare):\n    - Iterates tree branches/baskets and benchmarks the same baskets with:\n        (1) ns_root production decompressor (bulk)\n        (2) legacy ruzstd StreamingDecoder+read_to_end (ZSTD only)\n    - Validates first basket output byte-for-byte then reports MiB/s and speedup\n\n  Bench mode (--bench-tree):\n    - Iterates tree branches/baskets and decompresses each basket payload using ns_root production decompressor\n    - Prints throughput stats\n\nExit status:\n  - 0 if all checked blocks match (verify modes) / no failures (bench modes)\n  - non-zero on any mismatch / decompression failure\n"
    );
}

#[derive(Copy, Clone, Debug, Default)]
struct ScanStats {
    zs_blocks_checked: usize,
    candidates_skipped: usize,
    failures: usize,
}

#[derive(Copy, Clone, Debug, Default)]
struct BenchScanStats {
    zs_blocks_decompressed: usize,
    candidates_skipped: usize,
    failures: usize,
    total_in_bytes: u64,
    total_out_bytes: u64,
    seconds: f64,
    blocks_per_s: f64,
    out_mib_per_s: f64,
}

fn scan_root_for_zs_blocks(
    bytes: &[u8],
    verbose: bool,
    fail_fast: bool,
) -> Result<ScanStats, io::Error> {
    const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

    let mut stats = ScanStats::default();
    let mut pos = 0usize;
    while pos + 9 <= bytes.len() {
        if &bytes[pos..pos + 2] != b"ZS" {
            pos += 1;
            continue;
        }

        let c_size = read_le24(&bytes[pos + 3..pos + 6]);
        let u_size = read_le24(&bytes[pos + 6..pos + 9]);
        if c_size == 0 || u_size == 0 {
            stats.candidates_skipped += 1;
            pos += 2;
            continue;
        }

        let payload = pos + 9;
        let end = payload.saturating_add(c_size);
        if end > bytes.len() {
            stats.candidates_skipped += 1;
            pos += 2;
            continue;
        }

        if payload + 4 > bytes.len() || bytes[payload..payload + 4] != ZSTD_MAGIC {
            stats.candidates_skipped += 1;
            pos += 2;
            continue;
        }

        let compressed = &bytes[payload..end];

        let ctx = ScanCtx {
            file_offset: pos,
            block_c_size: c_size,
            block_u_size: u_size,
        };

        match compare_zstd_block_scan(compressed, &ctx, verbose) {
            Ok(()) => {
                stats.zs_blocks_checked += 1;
                pos = end;
            }
            Err(e) => {
                stats.failures += 1;
                eprintln!("FAIL(scan) file_off={} : {}", pos, e);
                if fail_fast {
                    return Err(io::Error::new(io::ErrorKind::Other, "root_accuracy_gate failed"));
                }
                pos += 2;
            }
        }
    }

    Ok(stats)
}

fn bench_scan_root_for_zs_blocks(
    bytes: &[u8],
    verbose: bool,
    fail_fast: bool,
) -> Result<BenchScanStats, io::Error> {
    const ZSTD_MAGIC: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];

    let start = Instant::now();
    let mut stats = BenchScanStats::default();

    let mut pos = 0usize;
    while pos + 9 <= bytes.len() {
        if &bytes[pos..pos + 2] != b"ZS" {
            pos += 1;
            continue;
        }

        let c_size = read_le24(&bytes[pos + 3..pos + 6]);
        let u_size = read_le24(&bytes[pos + 6..pos + 9]);
        if c_size == 0 || u_size == 0 {
            stats.candidates_skipped += 1;
            pos += 2;
            continue;
        }

        let payload = pos + 9;
        let end = payload.saturating_add(c_size);
        if end > bytes.len() {
            stats.candidates_skipped += 1;
            pos += 2;
            continue;
        }

        if payload + 4 > bytes.len() || bytes[payload..payload + 4] != ZSTD_MAGIC {
            stats.candidates_skipped += 1;
            pos += 2;
            continue;
        }

        let root_block = &bytes[pos..end];
        match ns_root::decompress::decompress(root_block, u_size) {
            Ok(out) => {
                if out.len() != u_size {
                    stats.failures += 1;
                    if verbose {
                        eprintln!(
                            "FAIL(bench-scan) file_off={} : output size mismatch got {} expected {}",
                            pos,
                            out.len(),
                            u_size
                        );
                    }
                    if fail_fast {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "root_accuracy_gate bench failed",
                        ));
                    }
                    pos += 2;
                    continue;
                }

                stats.zs_blocks_decompressed += 1;
                stats.total_in_bytes += c_size as u64;
                stats.total_out_bytes += u_size as u64;
                pos = end;
            }
            Err(e) => {
                if fail_fast {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        format!("root_accuracy_gate bench failed: {e}"),
                    ));
                }
                stats.candidates_skipped += 1;
                if verbose {
                    eprintln!("SKIP(bench-scan) file_off={} : {}", pos, e);
                }
                pos += 2;
            }
        }
    }

    let seconds = start.elapsed().as_secs_f64();
    stats.seconds = seconds;
    if seconds > 0.0 {
        stats.blocks_per_s = stats.zs_blocks_decompressed as f64 / seconds;
        stats.out_mib_per_s = (stats.total_out_bytes as f64 / (1024.0 * 1024.0)) / seconds;
    }

    Ok(stats)
}

#[derive(Copy, Clone)]
struct ScanCtx {
    file_offset: usize,
    block_c_size: usize,
    block_u_size: usize,
}

fn compare_zstd_block_scan(compressed: &[u8], ctx: &ScanCtx, verbose: bool) -> Result<(), String> {
    use std::io::Cursor;

    let mut ref_dec = zstd::stream::Decoder::new(Cursor::new(compressed))
        .map_err(|e| format!("libzstd (zstd crate) init failed: {e}"))?;

    let mut ruz_source = compressed;
    let mut ruz_dec = ruzstd::decoding::StreamingDecoder::new(&mut ruz_source)
        .map_err(|e| format!("ruzstd init failed: {e}"))?;

    let bytes = stream_compare(&mut ruz_dec, &mut ref_dec).map_err(|e| {
        format!(
            "{} (file_off={} c_size={} u_size={})",
            e, ctx.file_offset, ctx.block_c_size, ctx.block_u_size
        )
    })?;

    if bytes != ctx.block_u_size as u64 {
        return Err(format!(
            "decoded size mismatch: got {} expected {} (file_off={} c_size={})",
            bytes, ctx.block_u_size, ctx.file_offset, ctx.block_c_size
        ));
    }

    if verbose {
        println!(
            "OK(scan) file_off={} out={} bytes",
            ctx.file_offset, bytes
        );
    }
    Ok(())
}

fn select_first_tree_name(file: &RootFile) -> Result<String, io::Error> {
    let keys: Vec<KeyInfo> = file
        .list_keys()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("failed to list keys: {e}")))?;

    for k in keys {
        if k.class_name == "TTree" {
            return Ok(k.name);
        }
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "no TTree found in top-level keys; pass --tree <name>",
    ))
}

#[derive(Copy, Clone, Debug, Default)]
struct BenchTreeStats {
    branches_checked: usize,
    baskets_checked: usize,
    baskets_uncompressed_skipped: usize,
    failures: usize,
    total_in_bytes: u64,
    total_out_bytes: u64,
    seconds: f64,
    baskets_per_s: f64,
    out_mib_per_s: f64,
}

fn bench_tree_baskets(
    bytes: &[u8],
    root_path: &Path,
    tree_name: &str,
    branches: &[ns_root::BranchInfo],
    max_baskets_per_branch: usize,
    fail_fast: bool,
) -> Result<BenchTreeStats, io::Error> {
    let start = Instant::now();
    let mut stats = BenchTreeStats::default();

    for branch in branches {
        stats.branches_checked += 1;
        let n = branch.n_baskets.min(max_baskets_per_branch);
        for i in 0..n {
            let seek = branch.basket_seek[i];
            let (obj_len, key_len, n_bytes) = read_tkey_header_fast(bytes, seek as usize)?;

            let key_end = seek as usize + n_bytes;
            let obj_start = seek as usize + key_len;
            if key_end > bytes.len() || obj_start > key_end {
                stats.failures += 1;
                eprintln!(
                    "FAIL(bench-tree) root={} tree={} branch={} basket={} seek={} : key slice out of bounds",
                    root_path.display(),
                    tree_name,
                    branch.name,
                    i,
                    seek
                );
                if fail_fast {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "root_accuracy_gate bench failed",
                    ));
                }
                continue;
            }

            let compressed_data = &bytes[obj_start..key_end];
            let compressed_len = n_bytes - key_len;
            if obj_len == compressed_len {
                stats.baskets_uncompressed_skipped += 1;
                continue;
            }

            match ns_root::decompress::decompress(compressed_data, obj_len) {
                Ok(out) => {
                    if out.len() != obj_len {
                        stats.failures += 1;
                        eprintln!(
                            "FAIL(bench-tree) root={} tree={} branch={} basket={} seek={} : output size mismatch got {} expected {}",
                            root_path.display(),
                            tree_name,
                            branch.name,
                            i,
                            seek,
                            out.len(),
                            obj_len
                        );
                        if fail_fast {
                            return Err(io::Error::new(
                                io::ErrorKind::Other,
                                "root_accuracy_gate bench failed",
                            ));
                        }
                        continue;
                    }

                    stats.baskets_checked += 1;
                    stats.total_in_bytes += compressed_data.len() as u64;
                    stats.total_out_bytes += obj_len as u64;
                }
                Err(e) => {
                    stats.failures += 1;
                    eprintln!(
                        "FAIL(bench-tree) root={} tree={} branch={} basket={} seek={} : {}",
                        root_path.display(),
                        tree_name,
                        branch.name,
                        i,
                        seek,
                        e
                    );
                    if fail_fast {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "root_accuracy_gate bench failed",
                        ));
                    }
                }
            }
        }
    }

    let seconds = start.elapsed().as_secs_f64();
    stats.seconds = seconds;
    if seconds > 0.0 {
        stats.baskets_per_s = stats.baskets_checked as f64 / seconds;
        stats.out_mib_per_s = (stats.total_out_bytes as f64 / (1024.0 * 1024.0)) / seconds;
    }

    Ok(stats)
}

#[derive(Copy, Clone)]
struct Ctx<'a> {
    root_path: &'a Path,
    tree_name: &'a str,
    branch_name: &'a str,
    basket_index: usize,
    basket_seek: u64,
}

#[derive(Copy, Clone, Debug, Default)]
struct VerifyStats {
    blocks_checked: usize,
    blocks_skipped_non_zs: usize,
}

fn read_tkey_header_fast(file_data: &[u8], pos: usize) -> Result<(usize, usize, usize), io::Error> {
    use ns_root::rbuffer::RBuffer;

    let mut r = RBuffer::new(file_data);
    r.set_pos(pos);

    let n_bytes = r
        .read_u32()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("read TKey n_bytes: {e}")))?
        as usize;
    let _version = r
        .read_u16()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("read TKey version: {e}")))?;
    let obj_len = r
        .read_u32()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("read TKey obj_len: {e}")))?
        as usize;
    let _datime = r
        .read_u32()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("read TKey datime: {e}")))?;
    let key_len = r
        .read_u16()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("read TKey key_len: {e}")))?
        as usize;

    Ok((obj_len, key_len, n_bytes))
}

fn verify_root_blocks_zs_only(
    src: &[u8],
    expected_len: usize,
    ctx: &Ctx<'_>,
    verbose: bool,
) -> Result<VerifyStats, String> {
    let mut stats = VerifyStats::default();

    let mut offset = 0usize;
    let mut total_uncompressed = 0usize;

    while total_uncompressed < expected_len {
        if offset + 9 > src.len() {
            return Err(format!(
                "truncated ROOT compression block header at payload offset {} (need 9 bytes)",
                offset
            ));
        }

        let tag = &src[offset..offset + 2];
        let c_size = read_le24(&src[offset + 3..offset + 6]);
        let u_size = read_le24(&src[offset + 6..offset + 9]);
        offset += 9;

        let end = offset + c_size;
        if end > src.len() {
            return Err(format!(
                "compressed sub-block claims {} bytes but only {} remain",
                c_size,
                src.len() - offset
            ));
        }

        let compressed = &src[offset..end];

        if tag == b"ZS" {
            stats.blocks_checked += 1;
            compare_zstd_block(compressed, u_size, ctx, total_uncompressed, verbose)?;
        } else {
            stats.blocks_skipped_non_zs += 1;
        }

        total_uncompressed += u_size;
        offset = end;
    }

    if total_uncompressed != expected_len {
        return Err(format!(
            "total uncompressed bytes {} != expected {}",
            total_uncompressed, expected_len
        ));
    }

    Ok(stats)
}

fn compare_zstd_block(
    compressed: &[u8],
    expected_u_size: usize,
    ctx: &Ctx<'_>,
    basket_uncompressed_offset: usize,
    verbose: bool,
) -> Result<(), String> {
    use std::io::Cursor;

    let mut ref_dec = zstd::stream::Decoder::new(Cursor::new(compressed))
        .map_err(|e| format!("libzstd (zstd crate) init failed: {e}"))?;

    let mut ruz_source = compressed;
    let mut ruz_dec = ruzstd::decoding::StreamingDecoder::new(&mut ruz_source)
        .map_err(|e| format!("ruzstd init failed: {e}"))?;

    let bytes = stream_compare(&mut ruz_dec, &mut ref_dec).map_err(|e| {
        format!(
            "{} (tree='{}' branch='{}' basket={} seek={} basket_off={})",
            e,
            ctx.tree_name,
            ctx.branch_name,
            ctx.basket_index,
            ctx.basket_seek,
            basket_uncompressed_offset,
        )
    })?;

    if bytes != expected_u_size as u64 {
        return Err(format!(
            "decoded size mismatch: got {} expected {} (tree='{}' branch='{}' basket={} seek={} basket_off={})",
            bytes,
            expected_u_size,
            ctx.tree_name,
            ctx.branch_name,
            ctx.basket_index,
            ctx.basket_seek,
            basket_uncompressed_offset,
        ));
    }

    if verbose {
        println!(
            "OK root={} tree={} branch={} basket={} seek={} block_out={} bytes",
            ctx.root_path.display(),
            ctx.tree_name,
            ctx.branch_name,
            ctx.basket_index,
            ctx.basket_seek,
            bytes
        );
    }

    Ok(())
}

fn stream_compare(a: &mut dyn Read, b: &mut dyn Read) -> Result<u64, String> {
    let mut a_buf = vec![0u8; BUF_SIZE];
    let mut b_buf = vec![0u8; BUF_SIZE];

    let mut a_pos = 0usize;
    let mut a_len = 0usize;
    let mut b_pos = 0usize;
    let mut b_len = 0usize;

    let mut total: u64 = 0;

    loop {
        if a_pos == a_len {
            a_len = a.read(&mut a_buf).map_err(|e| format!("read ruzstd: {e}"))?;
            a_pos = 0;
        }
        if b_pos == b_len {
            b_len = b.read(&mut b_buf).map_err(|e| format!("read libzstd: {e}"))?;
            b_pos = 0;
        }

        if a_len == 0 && b_len == 0 {
            return Ok(total);
        }
        if a_len == 0 && b_len != 0 {
            return Err(format!(
                "output length mismatch at offset {total}: ruzstd ended early, reference has more data"
            ));
        }
        if b_len == 0 && a_len != 0 {
            return Err(format!(
                "output length mismatch at offset {total}: reference ended early, ruzstd has more data"
            ));
        }

        let a_avail = a_len - a_pos;
        let b_avail = b_len - b_pos;
        let n = a_avail.min(b_avail);

        let a_slice = &a_buf[a_pos..a_pos + n];
        let b_slice = &b_buf[b_pos..b_pos + n];

        if a_slice != b_slice {
            let mut i = 0usize;
            while i < n {
                if a_slice[i] != b_slice[i] {
                    break;
                }
                i += 1;
            }
            let off = total + i as u64;

            let a_tail = hex_preview(&a_slice[i..], 32);
            let b_tail = hex_preview(&b_slice[i..], 32);

            return Err(format!(
                "byte mismatch at offset {off}: ruzstd=0x{:02X} ref=0x{:02X}; ruzstd_tail={a_tail} ref_tail={b_tail}",
                a_slice[i],
                b_slice[i]
            ));
        }

        a_pos += n;
        b_pos += n;
        total += n as u64;
    }
}

fn hex_preview(bytes: &[u8], max: usize) -> String {
    let mut out = String::new();
    let n = bytes.len().min(max);
    for (i, &b) in bytes[..n].iter().enumerate() {
        if i != 0 {
            out.push(' ');
        }
        out.push_str(&format!("{:02X}", b));
    }
    out
}

fn read_le24(b: &[u8]) -> usize {
    b[0] as usize | ((b[1] as usize) << 8) | ((b[2] as usize) << 16)
}
