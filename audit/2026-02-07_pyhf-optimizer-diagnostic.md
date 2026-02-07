# pyhf vs NextStat — Optimizer Diagnostic (SLSQP) and Objective Parity

Date: 2026-02-07

Goal: validate **NextStat as a tool** against **pyhf** (trusted reference for HistFactory likelihood), and determine whether observed MLE differences come from:

1) different likelihood/objective implementations (bug), or  
2) optimizer convergence quality / initialization (expected).

This audit focuses on the two “large” public fixtures:

- `tests/fixtures/workspace_tHu.json` (184 params)
- `tests/fixtures/tttt-prod_workspace.json` (249 params)

`tchannel_workspace.json` is intentionally excluded here (it belongs to a separate test).

---

## Environment / Config

Repro commands were executed with single-thread settings for determinism/fairness:

```bash
export PYTHONPATH=bindings/ns-py/python
```

pyhf:

- version: `0.7.6`
- backend: `numpy`
- optimizer: `scipy_optimizer`
- MLE config used in repeats: `method=SLSQP`, `maxiter=100000`, `tolerance=1e-8`, `do_grad=0`, `do_stitch=0`

NextStat:

- version: `0.1.0`
- MLE config used in repeats: `max_iter=3000`, `tol=1e-6`, `m=10`

---

## Result A — Objective parity (cross-eval)

We use a “cross-eval” check to separate “objective mismatch” from “optimizer landed elsewhere”:

- Evaluate **pyhf NLL** at **NextStat best-fit** parameters.
- Evaluate **NextStat NLL** at **pyhf best-fit** parameters.

From the repeated-fit audit (10 runs, warmup 2, `--fit-order alternate`), the maximum absolute objective delta at the two optima is tiny:

| Workspace | max\|Δobj\| @ optima |
|---|---:|
| `workspace_tHu.json` | `~9.38e-13` |
| `tttt-prod_workspace.json` | `~1.17e-11` |

Interpretation: **the likelihood objectives match** (within numerical precision).

Raw artifacts:

- `tmp/repeat_mle_fits_verify.jsonl`
- `tmp/repeat_mle_fits_verify_audit.md`
- `tmp/repeat_mle_fits_verify_audit.json`

Repro:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/repeat_mle_fits.py \
  --workspace tests/fixtures/workspace_tHu.json \
  --workspace tests/fixtures/tttt-prod_workspace.json \
  --threads 1 \
  --fit-order alternate \
  --n-warmup 2 \
  --n-runs 10 \
  --pyhf-method SLSQP \
  --pyhf-maxiter 100000 \
  --pyhf-tolerance 1e-8 \
  --pyhf-do-grad 0 \
  --pyhf-do-stitch 0 \
  --nextstat-max-iter 3000 \
  --nextstat-tol 1e-6 \
  --out-jsonl tmp/repeat_mle_fits_verify.jsonl

./.venv/bin/python tests/analyze_repeat_mle_fits.py \
  --in-jsonl tmp/repeat_mle_fits_verify.jsonl \
  --out-json tmp/repeat_mle_fits_verify_audit.json \
  --out-md tmp/repeat_mle_fits_verify_audit.md
```

---

## Result B — Optimizer quality (SLSQP in pyhf)

We run an optimizer diagnostic that measures:

1) Gradient norms (computed by NextStat) at both optima  
2) Cross-init behavior:
   - `pyhf(init=NS_hat)`
   - `NextStat(init=pyhf_hat)`
3) Multi-start behavior for pyhf (random initializations within bounds)

### Reproduced diagnostic numbers (this run)

`workspace_tHu.json`:

- pyhf default NLL: `179.4853177926`
- NextStat NLL: `179.4043376656`
- ΔNLL (NS − pyhf): `-0.0809801270`
- ‖∇‖ @ pyhf hat (NextStat grad): `4.6334467317`
- ‖∇‖ @ NS hat (NextStat grad): `0.0203911770`
- pyhf(init=NS_hat) NLL: `179.4043321466` (improvement `+0.0809856459`)
- pyhf multi-start (20): `n_ok=5`, `n_failed=15`, best NLL `179.4076767018`
- verdict: `MATCH` (pyhf reaches NS-level with better init)

`tttt-prod_workspace.json`:

- pyhf default NLL: `287.5123044831`
- NextStat NLL: `287.5021329245`
- ΔNLL (NS − pyhf): `-0.0101715586`
- ‖∇‖ @ pyhf hat (NextStat grad): `1.4352898945`
- ‖∇‖ @ NS hat (NextStat grad): `0.0080540066`
- pyhf(init=NS_hat) NLL: `287.5021301802` (improvement `+0.0101743029`)
- pyhf multi-start (20): `n_ok=2`, `n_failed=18`, best NLL `287.5021458765`
- verdict: `MATCH` (pyhf reaches NS-level with better init)

Interpretation:

- NextStat finds a deeper minimum than pyhf’s default SLSQP run.
- **pyhf can reach that deeper minimum when warm-started from NextStat’s solution**.
- Gradient norms confirm: **pyhf’s default “hat” is not near-stationary**, while NextStat’s is.
- Multi-start with random inits is highly unreliable here (many failures) and does not reliably discover the better minimum.

Raw artifacts:

- `tmp/diagnose_optimizer_repro.json`
- `tmp/diagnose_optimizer_repro.log` (verbose)

Repro:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=bindings/ns-py/python ./.venv/bin/python tests/diagnose_optimizer.py \
  --workspace tests/fixtures/workspace_tHu.json \
  --workspace tests/fixtures/tttt-prod_workspace.json \
  --threads 1 \
  --multi-start 20 \
  --seed 42 \
  --pyhf-method SLSQP \
  --pyhf-maxiter 100000 \
  --pyhf-tolerance 1e-8 \
  --ns-max-iter 3000 \
  --ns-tol 1e-6 \
  --ns-m 10 \
  --out-json tmp/diagnose_optimizer_repro.json
```

---

## Conclusion

For these large HistFactory workspaces:

- **Likelihood parity is confirmed** (cross-eval objective deltas ~1e-13…1e-11).
- The observed MLE discrepancies are explained by **pyhf(SLSQP) optimizer convergence / initialization quality**, not by a NextStat likelihood bug.

Practical implication for “NextStat vs pyhf” validation:

- Treat “pyhf default MLE” as a *tooling baseline*, not a guaranteed global optimum on large models.
- For correctness, prefer:
  - objective parity checks at shared points, and/or
  - cross-init (pyhf warm-started from NextStat) as a confirmation step.

