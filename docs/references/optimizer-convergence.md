---
title: "Optimizer Convergence & Best-NLL Philosophy"
status: stable
---

# Optimizer Convergence & Best-NLL Philosophy

## TL;DR

NextStat uses L-BFGS-B and, by default, targets the **best NLL minimum** ("best-NLL").
Differences vs pyhf in best-fit parameters on large models (>100 params) are **expected and
documented behavior**, not a bug.

---

## 1. Position: Best-NLL by default

### Principle

NextStat does **not** intentionally constrain the optimizer just to match a specific external tool.
If L-BFGS-B finds a deeper minimum than pyhf's SLSQP, that is a correct result.

### Rationale

1. **Objective parity is validated**: NextStat and pyhf compute the same NLL at the same parameter
   point (cross-eval), typically agreeing at ~1e-9 to 1e-13. The underlying likelihood model is the same.

2. **Differences come from the optimizer, not the model**: on large models, pyhf's SLSQP can stop
   with a relatively large gradient norm (e.g. ||grad|| ~ 1.4 to 4.6) before reaching the true minimum.
   L-BFGS-B, with a quasi-Newton Hessian approximation, often continues descending.

3. **Cross-validation**: pyhf(init=NS_hat) can reach the same NLL as NextStat, meaning the minimum is
   accessible to both. NextStat(init=pyhf_hat) may still go lower, indicating better robustness to init.

4. **Multi-start analysis**: on large models, SLSQP multi-start (e.g. 20 restarts) may converge in
   <25% of runs; the best outcome can still be worse than NextStat's best-NLL.

### Typical mismatch scale

| Model | Parameters | ΔNLL (NS - pyhf) | Reason |
|--------|-----------|-------------------|---------|
| simple_workspace | 2 | 0.0 | Both converge |
| complex_workspace | 9 | 0.0 | Both converge |
| tchannel | 184 | -0.01 to -0.08 | pyhf SLSQP premature stop |
| tHu | ~200 | -0.08 | pyhf SLSQP premature stop |
| tttt | 249 | -0.01 | pyhf SLSQP premature stop |

> Negative ΔNLL means NextStat finds a **better** (lower) minimum.

---

## 2. Parity levels

### Level 1: Objective parity (P0, required)

NLL(params) matches between NextStat and pyhf at the same params.

- Tolerance: rtol=1e-6, atol=1e-8 (see `docs/plans/standards.md` §2.2)
- Verified by: golden tests on fixture workspaces
- Status: **done** for all 7 fixture workspaces

### Level 2: Fit parity (P1, conditional)

Best-fit parameters match within tolerances.

- Tolerance: atol=2e-4 on parameters, atol=5e-4 on uncertainties
- **Expected behavior**: full agreement on small models (<50 params); mismatches on large models due
  to different optimizers
- **Not a defect** if NS NLL <= pyhf NLL

### Level 3: Optimizer compatibility (not implemented, not planned)

Intentionally degrading the optimizer to match SLSQP is **rejected**: it is an artificial constraint
with no scientific value.

---

## 3. How to verify

### For users

```python
import nextstat, json

ws = json.load(open("workspace.json"))
model = nextstat.from_pyhf(json.dumps(ws))
result = nextstat.fit(model)

# NLL at the minimum: lower is better
print(f"NLL: {result.nll}")
print(f"||grad||: ...")  # available via diagnostics
```

### For developers (parity checks)

```bash
# Objective parity (must always pass)
make pyhf-audit-nll

# Fit parity (may differ on large models; expected)
make pyhf-audit-fit
```

### Diagnostic script

```bash
# Cross-eval: verifies that NS and pyhf compute the same NLL at each other's best-fit points
python tests/diagnose_optimizer.py workspace.json
```

---

## 4. Warm-start for pyhf reproducibility

If a **specific** use case requires matching pyhf (e.g. reproducing a published result), use a
warm-start from the pyhf best-fit point:

```python
import pyhf, nextstat, json

# 1. Fit in pyhf
ws = json.load(open("workspace.json"))
model = pyhf.Workspace(ws).model()
pyhf_pars, _ = pyhf.infer.mle.fit(model.config.suggested_init(), model, return_uncertainties=True)

# 2. Warm-start NextStat from the pyhf point
ns_model = nextstat.from_pyhf(json.dumps(ws))
result = nextstat.fit(ns_model, init_pars=pyhf_pars.tolist())
# result.nll <= pyhf NLL (guaranteed)
```

This is **not** a "compatibility mode": NextStat can still go lower.
But if SLSQP truly found the minimum, NextStat will confirm it.

---

## 5. L-BFGS-B vs SLSQP: technical details

| Aspect | L-BFGS-B (NextStat) | SLSQP (pyhf/scipy) |
|--------|---------------------|---------------------|
| Hessian | Quasi-Newton (m=10 history) | Rank-1 update |
| Bounds | Native box constraints | Native box constraints |
| Constraints | Via reparameterization | Native linear/nonlinear |
| Convergence | ||proj_grad|| < ftol | ||grad|| threshold (tunable) |
| Scaling | O(m*n) per iteration | O(n^2) per iteration |
| Large models (>100 params) | Robust | Often premature stop |

L-BFGS-B with m=10 keeps 10 (s, y) pairs to approximate the Hessian. This often provides better
curvature information for N>100 than SLSQP's rank-1 updates.

---

---

## 6. Profile scan evidence: ROOT vs NextStat vs pyhf

Beyond MLE fits, profile likelihood scans provide independent validation. Three
canonical HistFactory fixtures were tested through ROOT/RooFit, pyhf, and NextStat
(`tests/validate_root_profile_scan.py --include-pyhf`).

### Results

| Fixture | NS vs pyhf max |dq(mu)| | NS vs ROOT max |dq(mu)| | ROOT free fit |
|---------|----------------------------|-----------------------------|---------------|
| xmlimport | 1e-7 | 0.051 | Converged |
| multichannel | 4e-7 | 3.4e-8 | Converged |
| coupled_histosys | 5e-6 | 22.5 | **FAILED (status=-1)** |

### Key observations

1. **NextStat and pyhf produce identical profile scans** (< 1e-5), confirming
   that both implement the same likelihood function.

2. **ROOT's conditional fits show optimizer effects** on xmlimport: constant NLL
   offset (11.06) proves identical model; growing q(mu) delta (up to 0.051) at
   tail is Minuit2 converging to slightly higher conditional NLL.

3. **ROOT's coupled_histosys failure is a model-level divergence**: the NLL offset
   between ROOT and NextStat grows from 420.74 (free fit) to 432.0 (mu=3.0),
   ruling out pure optimizer differences and indicating that ROOT evaluates the
   coupled HistoSys likelihood differently.

Full analysis: `docs/references/root-histfactory-comparison.md`

---

## References

- ROOT/HistFactory comparison: `docs/references/root-histfactory-comparison.md`
- Optimizer diagnostic report: `audit/2026-02-07_pyhf-optimizer-diagnostic.md`
- Precision standards: `docs/plans/standards.md`
- Parity modes: `docs/plans/2026-02-07_pyhf-spec-parity-plan.md`
- Diagnostic scripts: `tests/diagnose_optimizer.py`, `tests/repeat_mle_fits.py`
