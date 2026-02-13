//! Quick benchmark: expression engine throughput.
//! Run:
//!   cargo run -p ns-root --release --example bench_expr

use ns_root::{CompiledExpr, JaggedCol};

fn rand_f64(state: &mut u64) -> f64 {
    // xorshift64*
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    let x = (*state).wrapping_mul(2685821657736338717u64);
    // Map to [0, 1)
    (x >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    let rounds = 5;
    let iters = 15;

    // --- Vectorized arithmetic (min/max heavy) ---
    let n = 2_000_000usize;
    let mut state = 0x0123_4567_89ab_cdefu64;
    let mut a = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];
    for i in 0..n {
        a[i] = rand_f64(&mut state) * 4.0 - 2.0;
        b[i] = rand_f64(&mut state) * 4.0 - 2.0;
    }

    let expr = CompiledExpr::compile("min(max(a * 1.0001 + b * 0.9999, -1), 1)").unwrap();
    let cols: Vec<&[f64]> = expr
        .required_branches
        .iter()
        .map(|name| match name.as_str() {
            "a" => a.as_slice(),
            "b" => b.as_slice(),
            other => panic!("unexpected branch: {other}"),
        })
        .collect();

    // Warmup
    for _ in 0..3 {
        let out = expr.eval_bulk(&cols);
        std::hint::black_box(out);
    }

    println!("--- eval_bulk (vectorized) ---");
    let mut m_elems = Vec::new();
    for r in 0..rounds {
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let out = expr.eval_bulk(&cols);
            std::hint::black_box(&out);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let elems = (n as f64 * iters as f64) / elapsed / 1e6;
        m_elems.push(elems);
        println!("  Round {}: {:.1} Melem/s", r + 1, elems);
    }
    println!("Median: {:.1} Melem/s", median(m_elems));

    // --- Row-wise DynLoad (short-circuit control flow) ---
    let n = 500_000usize;
    let mut njet = vec![0.0f64; n];
    let mut offsets = Vec::with_capacity(n + 1);
    offsets.push(0usize);
    let mut flat = Vec::new();
    for x in &mut njet {
        // 0..=4 jets
        let nj = (rand_f64(&mut state) * 5.0) as usize;
        *x = nj as f64;
        for _ in 0..nj {
            flat.push(rand_f64(&mut state) * 200.0);
        }
        offsets.push(flat.len());
    }
    let jet_pt = JaggedCol { flat, offsets };

    let expr = CompiledExpr::compile("njet > 0 && jet_pt[njet - 1] > 25").unwrap();
    let cols: Vec<&[f64]> = expr
        .required_branches
        .iter()
        .map(|name| match name.as_str() {
            "njet" => njet.as_slice(),
            other => panic!("unexpected branch: {other}"),
        })
        .collect();
    let jagged_refs: Vec<&JaggedCol> = expr
        .required_jagged_branches
        .iter()
        .map(|name| match name.as_str() {
            "jet_pt" => &jet_pt,
            other => panic!("unexpected jagged branch: {other}"),
        })
        .collect();

    for _ in 0..3 {
        let out = expr.eval_bulk_with_jagged(&cols, &jagged_refs);
        std::hint::black_box(out);
    }

    println!("\n--- eval_bulk_with_jagged (row-wise DynLoad) ---");
    let mut m_rows = Vec::new();
    for r in 0..rounds {
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let out = expr.eval_bulk_with_jagged(&cols, &jagged_refs);
            std::hint::black_box(&out);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let rows = (n as f64 * iters as f64) / elapsed / 1e6;
        m_rows.push(rows);
        println!("  Round {}: {:.1} Mrow/s", r + 1, rows);
    }
    println!("Median: {:.1} Mrow/s", median(m_rows));
}
