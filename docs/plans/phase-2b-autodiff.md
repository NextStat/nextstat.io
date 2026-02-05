# Phase II-B: Autodiff & Optimizers (P0)

> See `docs/plans/standards.md` for canonical definitions (`twice_nll`), tolerances, precision policy, and determinism requirements.

**Goal:** accelerate and stabilize fitting by introducing analytic gradients (and selected second-order information), removing reliance on finite differences, and preparing the core for future GPU gradient paths.

**Duration:** Weeks 13-20 (may run in parallel with Phase II-A; should land before Phase II-C to unblock GPU gradient support).

**Architecture:** AD is a separate layer that plugs into `Model::gradient()` and the optimizers in `ns-inference`, without breaking clean-architecture boundaries.

**Tech Stack:** Rust, nalgebra (linear algebra), in-house AD (forward + reverse), optional JAX as a golden reference.

---

## Why Phase II-B is mandatory

- **Speed:** finite-difference gradients cost `O(N)` NLL evaluations per iteration, which becomes prohibitive at 50-200 nuisance parameters.
- **Stability:** finite differences are sensitive to step size, noise, and reduction order (Rayon/GPU).
- **Uncertainty quality:** reliable curvature/Hessian information matters for pulls, constraints behavior, and ranking.

---

## Contents

- [Sprint 2B.1: Forward-mode AD (minimal base)](#sprint-2b1-forward-mode-ad-minimal-base-weeks-13-14)
- [Sprint 2B.2: Reverse-mode AD (main path)](#sprint-2b2-reverse-mode-ad-main-path-weeks-15-17)
- [Sprint 2B.3: Hessian / HVP / uncertainties](#sprint-2b3-hessian--hvp--uncertainties-weeks-18-19)
- [Sprint 2B.4: Gradient-based optimizers](#sprint-2b4-gradient-based-optimizers-weeks-19-20)

---

## Sprint 2B.1: Forward-mode AD (minimal base) (Weeks 13-14)

### Epic 2B.1.1: Dual numbers

#### Task 2B.1.1.1: Dual scalar (1D) + tests

**Priority:** P0  
**Effort:** 4-6 hours  
**Dependencies:** Phase 1 complete

**Files:**
- Modify: `Cargo.toml` (workspace member + deps)
- Create: `crates/ns-ad/Cargo.toml`
- Create: `crates/ns-ad/src/lib.rs`
- Create: `crates/ns-ad/src/dual.rs`
- Test: `crates/ns-ad/src/dual_tests.rs`

**Acceptance Criteria:**
- [ ] `d/dx (x*x) = 2x`
- [ ] `d/dx ln(x) = 1/x`
- [ ] `d/dx exp(x) = exp(x)`

**Step 0: Wire crate into workspace**

```toml
# Cargo.toml
[workspace]
members = [
  # ...
  "crates/ns-ad",
]
```

```toml
# crates/ns-ad/Cargo.toml
[package]
name = "ns-ad"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "Automatic differentiation primitives for NextStat"

[dependencies]
ns-core = { path = "../ns-core" }

[dev-dependencies]
approx.workspace = true
```

**Step 1: Write failing tests**

```rust
// crates/ns-ad/src/dual_tests.rs
#[cfg(test)]
mod tests {
    use crate::dual::Dual;
    use approx::assert_relative_eq;

    #[test]
    fn test_square_derivative() {
        let x = Dual::var(3.0);
        let y = x * x;
        assert_relative_eq!(y.val, 9.0, epsilon = 1e-12);
        assert_relative_eq!(y.d1, 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_ln_derivative() {
        let x = Dual::var(2.0);
        let y = x.ln();
        assert_relative_eq!(y.d1, 0.5, epsilon = 1e-12);
    }
}
```

**Step 2: Implement Dual**

```rust
// crates/ns-ad/src/dual.rs
use ns_core::types::Float;

#[derive(Debug, Clone, Copy)]
pub struct Dual {
    pub val: Float,
    pub d1: Float,
}

impl Dual {
    #[inline]
    pub fn constant(val: Float) -> Self {
        Self { val, d1: 0.0 }
    }

    #[inline]
    pub fn var(val: Float) -> Self {
        Self { val, d1: 1.0 }
    }

    #[inline]
    pub fn ln(self) -> Self {
        Self { val: self.val.ln(), d1: self.d1 / self.val }
    }

    #[inline]
    pub fn exp(self) -> Self {
        let e = self.val.exp();
        Self { val: e, d1: self.d1 * e }
    }
}

impl std::ops::Add for Dual {
    type Output = Dual;
    fn add(self, rhs: Dual) -> Dual {
        Dual { val: self.val + rhs.val, d1: self.d1 + rhs.d1 }
    }
}

impl std::ops::Mul for Dual {
    type Output = Dual;
    fn mul(self, rhs: Dual) -> Dual {
        Dual {
            val: self.val * rhs.val,
            d1: self.d1 * rhs.val + self.val * rhs.d1,
        }
    }
}
```

**Step 3: Wire crate**

```rust
// crates/ns-ad/src/lib.rs
#![warn(missing_docs)]

pub mod dual;
#[cfg(test)]
mod dual_tests;
```

---

## Sprint 2B.2: Reverse-mode AD (main path) (Weeks 15-17)

### Epic 2B.2.1: Tape-based reverse AD

#### Task 2B.2.1.1: Tape + Var + backward()

**Priority:** P0  
**Effort:** 12-18 hours  
**Dependencies:** Sprint 2B.1

**Files:**
- Create: `crates/ns-ad/src/tape.rs`
- Test: `crates/ns-ad/src/tape_tests.rs`

**Acceptance Criteria:**
- [ ] Gradients for compositions (`ln`, `exp`, `+`, `*`) match finite differences (rtol 1e-7)
- [ ] No allocations in the hot loop (preallocated buffers / arena)

**Implementation sketch (minimal API):**

```rust
// crates/ns-ad/src/tape.rs
use ns_core::types::Float;

#[derive(Debug, Clone, Copy)]
pub struct Var(pub usize);

#[derive(Debug, Clone, Copy)]
enum Op {
    Input,
    Add(usize, usize),
    Mul(usize, usize),
    Ln(usize),
    Exp(usize),
}

pub struct Tape {
    vals: Vec<Float>,
    adj: Vec<Float>,
    ops: Vec<Op>,
}

impl Tape {
    pub fn new() -> Self {
        Self { vals: Vec::new(), adj: Vec::new(), ops: Vec::new() }
    }

    pub fn input(&mut self, v: Float) -> Var {
        let idx = self.vals.len();
        self.vals.push(v);
        self.adj.push(0.0);
        self.ops.push(Op::Input);
        Var(idx)
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        let idx = self.vals.len();
        self.vals.push(self.vals[a.0] + self.vals[b.0]);
        self.adj.push(0.0);
        self.ops.push(Op::Add(a.0, b.0));
        Var(idx)
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        let idx = self.vals.len();
        self.vals.push(self.vals[a.0] * self.vals[b.0]);
        self.adj.push(0.0);
        self.ops.push(Op::Mul(a.0, b.0));
        Var(idx)
    }

    pub fn ln(&mut self, a: Var) -> Var {
        let idx = self.vals.len();
        self.vals.push(self.vals[a.0].ln());
        self.adj.push(0.0);
        self.ops.push(Op::Ln(a.0));
        Var(idx)
    }

    pub fn exp(&mut self, a: Var) -> Var {
        let idx = self.vals.len();
        self.vals.push(self.vals[a.0].exp());
        self.adj.push(0.0);
        self.ops.push(Op::Exp(a.0));
        Var(idx)
    }

    pub fn backward(&mut self, out: Var) {
        self.adj.fill(0.0);
        self.adj[out.0] = 1.0;

        for i in (0..self.ops.len()).rev() {
            let seed = self.adj[i];
            match self.ops[i] {
                Op::Input => {}
                Op::Add(a, b) => {
                    self.adj[a] += seed;
                    self.adj[b] += seed;
                }
                Op::Mul(a, b) => {
                    let va = self.vals[a];
                    let vb = self.vals[b];
                    self.adj[a] += seed * vb;
                    self.adj[b] += seed * va;
                }
                Op::Ln(a) => {
                    self.adj[a] += seed / self.vals[a];
                }
                Op::Exp(a) => {
                    self.adj[a] += seed * self.vals[i];
                }
            }
        }
    }

    pub fn grad(&self, x: Var) -> Float {
        self.adj[x.0]
    }
}
```

> At this point this is a "minimal" reverse-mode AD sketch. Full AD support for HistFactory will require:
> - support for `powf`, `max`, piecewise functions (and a differentiability policy),
> - efficient tape reuse for batch evaluation,
> - explicit `deterministic=true` options (see standards).

---

## Sprint 2B.3: Hessian / HVP / uncertainties (Weeks 18-19)

### Epic 2B.3.1: Second-order information

#### Task 2B.3.1.1: Hessian-vector products (HVP)

**Priority:** P0  
**Effort:** 8-12 hours  
**Dependencies:** Sprint 2B.2

**Acceptance Criteria:**
- [ ] HVP matches finite-difference gradients (rtol 1e-6)
- [ ] Curvature-based uncertainties are more stable than the Phase 1 numeric Hessian

**Note:** a full Hessian is `O(N^2)` and can be expensive. For uncertainties and profile workflows, it is often sufficient to use:
- diagonal or block-diagonal approximations,
- HVP + CG (solve `H x = b` without explicitly forming `H`).

---

## Sprint 2B.4: Gradient-based optimizers (Weeks 19-20)

### Epic 2B.4.1: Optimizer upgrade path

#### Task 2B.4.1.1: L-BFGS (+ projection) using `Model::gradient`

**Priority:** P0  
**Effort:** 8-12 hours  
**Dependencies:** Sprint 2B.2

**Files:**
- Modify: `crates/ns-core/src/model.rs` (use `gradient()` when available)
- Modify: `crates/ns-inference/src/mle.rs`

**Acceptance Criteria:**
- [ ] When `Model::gradient` is available, `mle_fit` uses a gradient-based solver
- [ ] When gradients are not available, fall back to the Phase 1 minimizer
- [ ] Best-fit parity is preserved (see `tests/python/test_pyhf_validation.py`)

---

## Completion criteria

### Exit Criteria

Phase 2B is complete when:

1. [ ] `Model::gradient` is implemented for `HistFactoryModel` (CPU) and matches finite differences
2. [ ] Gradients match JAX (optional golden reference) on simple/complex fixtures
3. [ ] MLE fits use gradients and speed up for `N > 50` parameters
4. [ ] Uncertainties and ranking use an HVP/Hessian policy (not full `O(N^2)` where avoidable)
