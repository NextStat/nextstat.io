# Фаза II-B: Autodiff & Optimizers (P0)

> См. `docs/plans/standards.md` для каноничных определений `twice_nll`, допусков, precision policy и детерминизма.

**Goal:** Ускорить и стабилизировать fitting за счёт аналитических градиентов/Гессиана, убрать зависимость от численных производных и подготовить базу для GPU.

**Duration:** Недели 13-20 (может идти параллельно с Phase II-A; перед Phase II-C для GPU градиентов)

**Architecture:** AD как отдельный слой, который подключается к `Model::gradient()` и оптимизаторам в `ns-inference`, не ломая “чистую архитектуру”.

**Tech Stack:** Rust, nalgebra (линейная алгебра), собственный AD (forward + reverse), optional JAX for golden reference.

---

## Почему Phase II-B обязателен

- **Скорость:** численный градиент = O(N) вызовов NLL на итерацию, что убивает fits при 50–200 NP.
- **Стабильность:** численные производные чувствительны к step-size, шуму и редукциям (Rayon/GPU).
- **Качество uncertainties:** корректный Hessian/curvature критичен для pulls/constraints/ranking.

---

## Содержание

- [Sprint 2B.1: Forward-mode AD (минимальный базис)](#sprint-2b1-forward-mode-ad-минимальный-базис-недели-13-14)
- [Sprint 2B.2: Reverse-mode AD (основной путь)](#sprint-2b2-reverse-mode-ad-основной-путь-недели-15-17)
- [Sprint 2B.3: Hessian / HVP / uncertainties](#sprint-2b3-hessian--hvp--uncertainties-недели-18-19)
- [Sprint 2B.4: Gradient-based optimizers](#sprint-2b4-gradient-based-optimizers-недели-19-20)

---

## Sprint 2B.1: Forward-mode AD (минимальный базис) (Недели 13-14)

### Epic 2B.1.1: Dual numbers

#### Task 2B.1.1.1: Dual scalar (1D) + тесты

**Priority:** P0  
**Effort:** 4-6 часов  
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

## Sprint 2B.2: Reverse-mode AD (основной путь) (Недели 15-17)

### Epic 2B.2.1: Tape-based reverse AD

#### Task 2B.2.1.1: Tape + Var + backward()

**Priority:** P0  
**Effort:** 12-18 часов  
**Dependencies:** Sprint 2B.1

**Files:**
- Create: `crates/ns-ad/src/tape.rs`
- Test: `crates/ns-ad/src/tape_tests.rs`

**Acceptance Criteria:**
- [ ] Градиенты для композиции (`ln`, `exp`, `+`, `*`) совпадают с finite-diff (rtol 1e-7)
- [ ] Нет аллокаций в hot-loop (предварительные буферы/arena)

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

> На этом этапе это “минимальный” reverse-AD. Полный AD для HistFactory потребует:
> - support для `powf`, `max`, piecewise функций (и политики дифференцируемости),
> - эффективного хранения и переиспользования tape на batch-evaluation,
> - опций `deterministic=true` (см. standards).

---

## Sprint 2B.3: Hessian / HVP / uncertainties (Недели 18-19)

### Epic 2B.3.1: Second-order information

#### Task 2B.3.1.1: Hessian-vector products (HVP)

**Priority:** P0  
**Effort:** 8-12 часов  
**Dependencies:** Sprint 2B.2

**Acceptance Criteria:**
- [ ] HVP совпадает с finite-diff градиента (rtol 1e-6)
- [ ] Uncertainties по curvature стабильнее численного Hessian из Phase 1

**Note:** Полный Hessian O(N²) дорог. Для uncertainties и профилей часто достаточно:
- diagonal / block-diagonal approximations,
- HVP + CG (solve `H x = b` без явного `H`).

---

## Sprint 2B.4: Gradient-based optimizers (Недели 19-20)

### Epic 2B.4.1: Optimizer upgrade path

#### Task 2B.4.1.1: L-BFGS (+ projection) using `Model::gradient`

**Priority:** P0  
**Effort:** 8-12 часов  
**Dependencies:** Sprint 2B.2

**Files:**
- Modify: `crates/ns-core/src/model.rs` (использовать `gradient()` если доступен)
- Modify: `crates/ns-inference/src/mle.rs`

**Acceptance Criteria:**
- [ ] При наличии `Model::gradient`, `mle_fit` использует градиентный solver
- [ ] При отсутствии градиента — fallback на Phase 1 minimizer
- [ ] Parity по bestfit сохраняется (см. `tests/python/test_pyhf_validation.py`)

---

## Критерии завершения

### Exit Criteria

Phase 2B завершена когда:

1. [ ] `Model::gradient` реализован для HistFactoryModel (CPU) и проходит сравнение с finite-diff
2. [ ] Градиенты совпадают с JAX (optional golden reference) на simple/complex fixtures
3. [ ] MLE fit использует градиенты и ускоряется на N>50 параметрах
4. [ ] Uncertainties/ranking используют HVP/Hessian policy, а не полный O(N²) там, где не нужно
