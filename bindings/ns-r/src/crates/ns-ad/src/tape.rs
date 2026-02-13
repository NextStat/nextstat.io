//! Tape-based reverse-mode automatic differentiation.
//!
//! Records a computation graph (forward pass), then computes **all** gradients
//! in a single backward sweep.  Cost: one forward + one backward pass regardless
//! of the number of inputs — O(1) gradient evaluations vs O(N) for forward-mode.
//!
//! # Example
//! ```
//! use ns_ad::tape::Tape;
//!
//! let mut tape = Tape::new();
//! let x = tape.var(3.0);
//! let y = tape.var(5.0);
//! let z = tape.mul(x, y);       // z = x * y = 15
//! let w = tape.add(z, x);       // w = z + x = 18
//! tape.backward(w);
//! assert_eq!(tape.adjoint(x), 6.0);  // dw/dx = y + 1 = 6
//! assert_eq!(tape.adjoint(y), 3.0);  // dw/dy = x = 3
//! ```

/// Handle to a node on the tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Var(pub(crate) usize);

/// Operation recorded on the tape.
#[derive(Debug, Clone, Copy)]
enum Op {
    /// Input variable (leaf).
    Input,
    /// Constant (adjoint never propagated).
    Const,
    // Binary ops
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    // Unary ops
    Neg(usize),
    Ln(usize),
    Exp(usize),
    Powf(usize, f64),
    Powi(usize, i32),
    /// Max(a, b): gradient flows to the winner only.
    Max(usize, usize),
}

/// Node on the tape: value + operation that produced it.
#[derive(Debug, Clone)]
struct Node {
    val: f64,
    op: Op,
}

/// Reverse-mode AD tape.
///
/// Build a computation graph by calling methods (var, add, mul, ln, …),
/// then call [`backward`](Tape::backward) and read gradients with [`adjoint`](Tape::adjoint).
#[derive(Debug)]
pub struct Tape {
    nodes: Vec<Node>,
    adjoints: Vec<f64>,
}

impl Tape {
    /// Create an empty tape.
    pub fn new() -> Self {
        Self { nodes: Vec::new(), adjoints: Vec::new() }
    }

    /// Create a tape pre-allocated for `capacity` nodes.
    pub fn with_capacity(capacity: usize) -> Self {
        Self { nodes: Vec::with_capacity(capacity), adjoints: Vec::with_capacity(capacity) }
    }

    /// Number of nodes on the tape.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the tape is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear the tape for reuse (avoids reallocation).
    #[inline]
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.adjoints.clear();
    }

    // --- Leaf constructors ---

    /// Record an input variable.
    #[inline]
    pub fn var(&mut self, val: f64) -> Var {
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Input });
        Var(idx)
    }

    /// Record a constant (gradient never flows through it).
    #[inline]
    pub fn constant(&mut self, val: f64) -> Var {
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Const });
        Var(idx)
    }

    // --- Value access ---

    /// Get the primal value of a node.
    #[inline]
    pub fn val(&self, v: Var) -> f64 {
        self.nodes[v.0].val
    }

    // --- Binary operations ---

    /// `a + b`
    #[inline]
    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let val = self.nodes[a.0].val + self.nodes[b.0].val;
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Add(a.0, b.0) });
        Var(idx)
    }

    /// `a - b`
    #[inline]
    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        let val = self.nodes[a.0].val - self.nodes[b.0].val;
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Sub(a.0, b.0) });
        Var(idx)
    }

    /// `a * b`
    #[inline]
    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let val = self.nodes[a.0].val * self.nodes[b.0].val;
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Mul(a.0, b.0) });
        Var(idx)
    }

    /// `a / b`
    #[inline]
    pub fn div(&mut self, a: Var, b: Var) -> Var {
        let val = self.nodes[a.0].val / self.nodes[b.0].val;
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Div(a.0, b.0) });
        Var(idx)
    }

    /// `max(a, b)` — gradient flows to the winner.
    #[inline]
    pub fn max(&mut self, a: Var, b: Var) -> Var {
        let va = self.nodes[a.0].val;
        let vb = self.nodes[b.0].val;
        let val = if va >= vb { va } else { vb };
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Max(a.0, b.0) });
        Var(idx)
    }

    // --- Unary operations ---

    /// `-a`
    #[inline]
    pub fn neg(&mut self, a: Var) -> Var {
        let val = -self.nodes[a.0].val;
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Neg(a.0) });
        Var(idx)
    }

    /// `ln(a)`
    #[inline]
    pub fn ln(&mut self, a: Var) -> Var {
        let val = self.nodes[a.0].val.ln();
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Ln(a.0) });
        Var(idx)
    }

    /// `exp(a)`
    #[inline]
    pub fn exp(&mut self, a: Var) -> Var {
        let val = self.nodes[a.0].val.exp();
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Exp(a.0) });
        Var(idx)
    }

    /// `a^n` (float exponent)
    pub fn powf(&mut self, a: Var, n: f64) -> Var {
        let val = self.nodes[a.0].val.powf(n);
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Powf(a.0, n) });
        Var(idx)
    }

    /// `a^n` (integer exponent)
    pub fn powi(&mut self, a: Var, n: i32) -> Var {
        let val = self.nodes[a.0].val.powi(n);
        let idx = self.nodes.len();
        self.nodes.push(Node { val, op: Op::Powi(a.0, n) });
        Var(idx)
    }

    // --- Convenience: scalar helpers ---

    /// `a + scalar`
    #[inline]
    pub fn add_f64(&mut self, a: Var, s: f64) -> Var {
        let c = self.constant(s);
        self.add(a, c)
    }

    /// `scalar - a`
    #[inline]
    pub fn f64_sub(&mut self, s: f64, a: Var) -> Var {
        let c = self.constant(s);
        self.sub(c, a)
    }

    /// `a * scalar`
    #[inline]
    pub fn mul_f64(&mut self, a: Var, s: f64) -> Var {
        let c = self.constant(s);
        self.mul(a, c)
    }

    /// `a / scalar`
    #[inline]
    pub fn div_f64(&mut self, a: Var, s: f64) -> Var {
        let c = self.constant(s);
        self.div(a, c)
    }

    /// `max(a, scalar)`
    #[inline]
    pub fn max_f64(&mut self, a: Var, s: f64) -> Var {
        let c = self.constant(s);
        self.max(a, c)
    }

    // --- Backward pass ---

    /// Run reverse-mode AD from output node `out`.
    ///
    /// After calling this, use [`adjoint`](Tape::adjoint) to read ∂out/∂x
    /// for any input `x`.
    pub fn backward(&mut self, out: Var) {
        let n = self.nodes.len();
        self.adjoints.resize(n, 0.0);
        self.adjoints.fill(0.0);
        self.adjoints[out.0] = 1.0;

        for i in (0..n).rev() {
            let adj = self.adjoints[i];
            if adj == 0.0 {
                continue; // skip zero-adjoint nodes
            }

            match self.nodes[i].op {
                Op::Input | Op::Const => {}
                Op::Add(a, b) => {
                    self.adjoints[a] += adj;
                    self.adjoints[b] += adj;
                }
                Op::Sub(a, b) => {
                    self.adjoints[a] += adj;
                    self.adjoints[b] -= adj;
                }
                Op::Mul(a, b) => {
                    let va = self.nodes[a].val;
                    let vb = self.nodes[b].val;
                    self.adjoints[a] += adj * vb;
                    self.adjoints[b] += adj * va;
                }
                Op::Div(a, b) => {
                    let va = self.nodes[a].val;
                    let vb = self.nodes[b].val;
                    self.adjoints[a] += adj / vb;
                    self.adjoints[b] -= adj * va / (vb * vb);
                }
                Op::Neg(a) => {
                    self.adjoints[a] -= adj;
                }
                Op::Ln(a) => {
                    self.adjoints[a] += adj / self.nodes[a].val;
                }
                Op::Exp(a) => {
                    // d/da exp(a) = exp(a) = self.nodes[i].val
                    self.adjoints[a] += adj * self.nodes[i].val;
                }
                Op::Powf(a, n) => {
                    // d/da a^n = n * a^(n-1)
                    self.adjoints[a] += adj * n * self.nodes[a].val.powf(n - 1.0);
                }
                Op::Powi(a, n) => {
                    self.adjoints[a] += adj * (n as f64) * self.nodes[a].val.powi(n - 1);
                }
                Op::Max(a, b) => {
                    // Gradient flows to the winner
                    if self.nodes[a].val >= self.nodes[b].val {
                        self.adjoints[a] += adj;
                    } else {
                        self.adjoints[b] += adj;
                    }
                }
            }
        }
    }

    /// Read ∂output/∂v after calling [`backward`](Tape::backward).
    #[inline]
    pub fn adjoint(&self, v: Var) -> f64 {
        self.adjoints.get(v.0).copied().unwrap_or(0.0)
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_add() {
        // f = x + y, df/dx = 1, df/dy = 1
        let mut t = Tape::new();
        let x = t.var(3.0);
        let y = t.var(5.0);
        let z = t.add(x, y);
        assert_eq!(t.val(z), 8.0);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 1.0, epsilon = 1e-12);
        assert_relative_eq!(t.adjoint(y), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sub() {
        // f = x - y, df/dx = 1, df/dy = -1
        let mut t = Tape::new();
        let x = t.var(3.0);
        let y = t.var(5.0);
        let z = t.sub(x, y);
        assert_eq!(t.val(z), -2.0);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 1.0, epsilon = 1e-12);
        assert_relative_eq!(t.adjoint(y), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_mul() {
        // f = x * y, df/dx = y, df/dy = x
        let mut t = Tape::new();
        let x = t.var(3.0);
        let y = t.var(5.0);
        let z = t.mul(x, y);
        assert_eq!(t.val(z), 15.0);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 5.0, epsilon = 1e-12);
        assert_relative_eq!(t.adjoint(y), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_div() {
        // f = x / y, df/dx = 1/y, df/dy = -x/y^2
        let mut t = Tape::new();
        let x = t.var(6.0);
        let y = t.var(3.0);
        let z = t.div(x, y);
        assert_eq!(t.val(z), 2.0);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 1.0 / 3.0, epsilon = 1e-12);
        assert_relative_eq!(t.adjoint(y), -6.0 / 9.0, epsilon = 1e-12);
    }

    #[test]
    fn test_neg() {
        let mut t = Tape::new();
        let x = t.var(3.0);
        let z = t.neg(x);
        assert_eq!(t.val(z), -3.0);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), -1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_ln() {
        // f = ln(x), df/dx = 1/x
        let mut t = Tape::new();
        let x = t.var(2.0);
        let z = t.ln(x);
        assert_relative_eq!(t.val(z), 2.0_f64.ln(), epsilon = 1e-12);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_exp() {
        // f = exp(x), df/dx = exp(x)
        let mut t = Tape::new();
        let x = t.var(1.0);
        let z = t.exp(x);
        assert_relative_eq!(t.val(z), 1.0_f64.exp(), epsilon = 1e-12);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 1.0_f64.exp(), epsilon = 1e-12);
    }

    #[test]
    fn test_powf() {
        // f = x^3, df/dx = 3*x^2
        let mut t = Tape::new();
        let x = t.var(2.0);
        let z = t.powf(x, 3.0);
        assert_relative_eq!(t.val(z), 8.0, epsilon = 1e-12);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 12.0, epsilon = 1e-12);
    }

    #[test]
    fn test_powi() {
        // f = x^2, df/dx = 2x
        let mut t = Tape::new();
        let x = t.var(5.0);
        let z = t.powi(x, 2);
        assert_relative_eq!(t.val(z), 25.0, epsilon = 1e-12);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 10.0, epsilon = 1e-12);
    }

    #[test]
    fn test_max() {
        // max(3, 5) = 5, gradient flows to b
        let mut t = Tape::new();
        let a = t.var(3.0);
        let b = t.var(5.0);
        let z = t.max(a, b);
        assert_eq!(t.val(z), 5.0);

        t.backward(z);
        assert_relative_eq!(t.adjoint(a), 0.0, epsilon = 1e-12);
        assert_relative_eq!(t.adjoint(b), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_x_squared() {
        // f = x * x, df/dx = 2x
        let mut t = Tape::new();
        let x = t.var(3.0);
        let z = t.mul(x, x);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_chain_rule() {
        // f = ln(x^2) = 2*ln(x), df/dx = 2/x
        let mut t = Tape::new();
        let x = t.var(3.0);
        let x2 = t.mul(x, x);
        let z = t.ln(x2);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 2.0 / 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_multivariate() {
        // f(x, y) = x^2 * y + y^3
        // df/dx = 2*x*y
        // df/dy = x^2 + 3*y^2
        let mut t = Tape::new();
        let x = t.var(2.0);
        let y = t.var(3.0);

        let x2 = t.mul(x, x);
        let x2y = t.mul(x2, y);
        let y3 = t.powi(y, 3);
        let f = t.add(x2y, y3);

        assert_relative_eq!(t.val(f), 4.0 * 3.0 + 27.0, epsilon = 1e-12);

        t.backward(f);
        assert_relative_eq!(t.adjoint(x), 2.0 * 2.0 * 3.0, epsilon = 1e-12);
        assert_relative_eq!(t.adjoint(y), 4.0 + 3.0 * 9.0, epsilon = 1e-12);
    }

    #[test]
    fn test_poisson_nll_single_bin() {
        // NLL(lam) = lam - n*ln(lam), d/dlam = 1 - n/lam
        let n = 10.0;
        let mut t = Tape::new();
        let lam = t.var(12.0);

        let n_const = t.constant(n);
        let ln_lam = t.ln(lam);
        let n_ln_lam = t.mul(n_const, ln_lam);
        let nll = t.sub(lam, n_ln_lam);

        t.backward(nll);
        let expected_grad = 1.0 - n / 12.0;
        assert_relative_eq!(t.adjoint(lam), expected_grad, epsilon = 1e-12);
    }

    #[test]
    fn test_gaussian_constraint() {
        // f(x) = 0.5 * ((x - mu) / sigma)^2
        // f'(x) = (x - mu) / sigma^2
        let mu = 0.0;
        let sigma = 1.0;

        let mut t = Tape::new();
        let x = t.var(0.5);
        let mu_c = t.constant(mu);
        let diff = t.sub(x, mu_c);
        let sigma_c = t.constant(sigma);
        let pull = t.div(diff, sigma_c);
        let pull2 = t.mul(pull, pull);
        let half = t.constant(0.5);
        let f = t.mul(half, pull2);

        t.backward(f);
        let expected = (0.5 - mu) / (sigma * sigma);
        assert_relative_eq!(t.adjoint(x), expected, epsilon = 1e-12);
    }

    #[test]
    fn test_matches_forward_mode_dual() {
        use crate::dual::Dual;

        // f(x, y, z) = exp(-x) * (y^2 + ln(z)) + z/y
        let xv = 1.5;
        let yv = 2.0;
        let zv = 3.0;

        // --- Reverse mode (all gradients in one pass) ---
        let mut t = Tape::new();
        let x = t.var(xv);
        let y = t.var(yv);
        let z = t.var(zv);

        let neg_x = t.neg(x);
        let exp_neg_x = t.exp(neg_x);
        let y2 = t.mul(y, y);
        let ln_z = t.ln(z);
        let sum_inner = t.add(y2, ln_z);
        let term1 = t.mul(exp_neg_x, sum_inner);
        let z_over_y = t.div(z, y);
        let f_rev = t.add(term1, z_over_y);

        t.backward(f_rev);
        let rev_dx = t.adjoint(x);
        let rev_dy = t.adjoint(y);
        let rev_dz = t.adjoint(z);

        // --- Forward mode (one pass per variable) ---
        let f_dual = |xd: Dual, yd: Dual, zd: Dual| -> Dual {
            let exp_neg_x = (-xd).exp();
            let inner = yd * yd + zd.ln();
            exp_neg_x * inner + zd / yd
        };

        let fwd_dx = f_dual(Dual::var(xv), Dual::constant(yv), Dual::constant(zv)).dot;
        let fwd_dy = f_dual(Dual::constant(xv), Dual::var(yv), Dual::constant(zv)).dot;
        let fwd_dz = f_dual(Dual::constant(xv), Dual::constant(yv), Dual::var(zv)).dot;

        // Values must match
        assert_relative_eq!(
            t.val(f_rev),
            f_dual(Dual::var(xv), Dual::constant(yv), Dual::constant(zv)).val,
            epsilon = 1e-12
        );

        // Gradients must match
        assert_relative_eq!(rev_dx, fwd_dx, epsilon = 1e-12);
        assert_relative_eq!(rev_dy, fwd_dy, epsilon = 1e-12);
        assert_relative_eq!(rev_dz, fwd_dz, epsilon = 1e-12);
    }

    #[test]
    fn test_constant_gradient_through() {
        // f = 42 * x => df/dx = 42
        // The constant itself accumulates adjoint = x = 3 (correct, but not an input).
        let mut t = Tape::new();
        let c = t.constant(42.0);
        let x = t.var(3.0);
        let z = t.mul(c, x);

        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 42.0, epsilon = 1e-12);
        // Constant adjoint = df/dc = x = 3.0 (mathematically correct, just not useful)
        assert_relative_eq!(t.adjoint(c), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_tape_reuse() {
        let mut t = Tape::new();
        let x = t.var(2.0);
        let z = t.mul(x, x);
        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 4.0, epsilon = 1e-12);

        // Clear and reuse
        t.clear();
        let x = t.var(5.0);
        let z = t.mul(x, x);
        t.backward(z);
        assert_relative_eq!(t.adjoint(x), 10.0, epsilon = 1e-12);
    }

    #[test]
    fn test_scalar_helpers() {
        let mut t = Tape::new();
        let x = t.var(3.0);

        // 2*x + 1 => d/dx = 2
        let two_x = t.mul_f64(x, 2.0);
        let y = t.add_f64(two_x, 1.0);

        t.backward(y);
        assert_relative_eq!(t.adjoint(x), 2.0, epsilon = 1e-12);
    }
}
