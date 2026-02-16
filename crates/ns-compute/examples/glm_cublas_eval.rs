#![cfg(feature = "cuda")]

use ns_compute::cuda_glm_cublas::CudaGlmCublasEvaluator;

struct Rng64(u64);
impl Rng64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f64(&mut self) -> f64 {
        let v = self.next_u64() >> 11;
        (v as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

fn cpu_grad_nll(beta: &[f64], x_col: &[f64], y: &[f64], n: usize, p: usize) -> (Vec<f64>, f64) {
    let mut grad = vec![0.0f64; p];
    let mut nll = 0.0f64;

    for i in 0..n {
        let mut eta = 0.0;
        for j in 0..p {
            eta += x_col[j * n + i] * beta[j];
        }
        let prob = if eta >= 0.0 {
            let e = (-eta).exp();
            1.0 / (1.0 + e)
        } else {
            let e = eta.exp();
            e / (1.0 + e)
        };
        let diff = prob - y[i];
        for j in 0..p {
            grad[j] += diff * x_col[j * n + i];
        }
        nll += eta.max(0.0) + (1.0 + (-eta.abs()).exp()).ln() - y[i] * eta;
    }

    for j in 0..p {
        grad[j] += beta[j];
        nll += 0.5 * beta[j] * beta[j];
    }

    (grad, nll)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(5000usize);
    let p = std::env::var("P").ok().and_then(|s| s.parse().ok()).unwrap_or(20usize);
    let chains = std::env::var("CHAINS").ok().and_then(|s| s.parse().ok()).unwrap_or(1024usize);
    let iters = std::env::var("ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20usize);

    let mut rng = Rng64::new(42);
    let mut x_col = vec![0.0f64; n * p];
    for j in 0..p {
        for i in 0..n {
            x_col[j * n + i] = rng.next_f64() * 2.0 - 1.0;
        }
    }
    let mut y = vec![0.0f64; n];
    for yi in &mut y {
        *yi = if rng.next_f64() > 0.5 { 1.0 } else { 0.0 };
    }
    let mut beta = vec![0.0f64; chains * p];
    for b in &mut beta {
        *b = (rng.next_f64() * 2.0 - 1.0) * 0.1;
    }

    let mut eval = CudaGlmCublasEvaluator::new(&x_col, &y, n, p, chains)?;

    // Warmup
    let _ = eval.evaluate_host(&beta)?;

    let t0 = std::time::Instant::now();
    let mut last = (Vec::new(), Vec::new());
    for _ in 0..iters {
        last = eval.evaluate_host(&beta)?;
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let per_eval_ms = elapsed * 1e3 / iters as f64;

    // Parity check on chain 0 vs CPU.
    let (cpu_grad0, cpu_nll0) = cpu_grad_nll(&beta[0..p], &x_col, &y, n, p);
    let grad0 = &last.0[0..p];
    let nll0 = last.1[0];
    let mut max_abs = 0.0f64;
    for j in 0..p {
        max_abs = max_abs.max((grad0[j] - cpu_grad0[j]).abs());
    }
    let nll_abs = (nll0 - cpu_nll0).abs();

    println!(
        "cuBLAS GLM eval: n={}, p={}, chains={}, iters={}, total_s={:.4}, per_eval_ms={:.3}, evals_per_s={:.1}",
        n,
        p,
        chains,
        iters,
        elapsed,
        per_eval_ms,
        iters as f64 / elapsed
    );
    println!("parity(chain0): max_abs_grad_err={:.3e}, abs_nll_err={:.3e}", max_abs, nll_abs);

    Ok(())
}
