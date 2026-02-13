#!/usr/bin/env python3
"""Survival vs binary classification bias benchmark.

Ground truth Cox PH with varying censoring (40/60/80%).
Shows bias of sklearn LogisticRegression vs unbiased Cox PH.

The key insight: logistic regression on a binary "churned within horizon"
outcome is biased when censoring is present, because censored observations
are misclassified as non-events. Cox PH handles censoring correctly.

Usage:
    python scripts/benchmarks/bench_survival_vs_classification.py [--out-dir bench_results]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def generate_data_with_censoring(
    n: int, censoring_fraction: float, seed: int = 42
) -> dict:
    """Generate survival data with controlled censoring fraction.

    True coefficients: beta = [0.5, -0.3, 0.8, -0.1] (log hazard ratios).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.integers(0, 2, size=n).astype(float)
    x4 = rng.standard_normal(n)

    true_beta = np.array([0.5, -0.3, 0.8, -0.1])
    lp = x1 * true_beta[0] + x2 * true_beta[1] + x3 * true_beta[2] + x4 * true_beta[3]
    baseline_hazard = 0.1
    scale = 1.0 / (baseline_hazard * np.exp(lp))
    event_times = rng.exponential(scale)

    # Tune censoring rate by adjusting censoring distribution scale.
    # Binary search for the right scale to achieve target censoring fraction.
    lo, hi = 0.01, 1000.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        censor_times_trial = rng.exponential(mid * np.median(event_times), size=n)
        actual_cens = np.mean(event_times > censor_times_trial)
        if actual_cens < censoring_fraction:
            hi = mid
        else:
            lo = mid
    # Use final scale.
    rng2 = np.random.default_rng(seed + 1000)
    censor_scale = (lo + hi) / 2.0
    censor_times = rng2.exponential(censor_scale * np.median(event_times), size=n)
    events = event_times <= censor_times
    obs_times = np.minimum(event_times, censor_times)

    actual_censoring = 1.0 - events.mean()

    return {
        "times": obs_times,
        "events": events,
        "event_times": event_times,  # Ground truth (for oracle).
        "x": np.column_stack([x1, x2, x3, x4]),
        "true_beta": true_beta,
        "actual_censoring": actual_censoring,
    }


# ---------------------------------------------------------------------------
# Cox PH via NextStat (unbiased)
# ---------------------------------------------------------------------------

def _try_import_nextstat():
    try:
        import nextstat
        if callable(getattr(nextstat, "churn_risk_model", None)):
            return nextstat
        return None
    except ImportError:
        return None


def _find_nextstat_binary() -> str | None:
    candidates = [
        Path(__file__).resolve().parents[2] / "target" / "release" / "nextstat",
        Path(__file__).resolve().parents[2] / "target" / "debug" / "nextstat",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def fit_cox_ph_nextstat(data: dict) -> dict:
    ns = _try_import_nextstat()
    covariates = [row.tolist() for row in data["x"]]
    names = ["x1", "x2", "x3", "x4"]
    times_list = data["times"].tolist()
    events_list = data["events"].tolist()

    if ns is not None:
        result = ns.churn_risk_model(times_list, events_list, covariates, names, conf_level=0.95)
        return {
            "coefficients": result["coefficients"],
            "hazard_ratios": result["hazard_ratios"],
            "se": result["se"],
        }

    # CLI fallback.
    nextstat_bin = _find_nextstat_binary()
    if nextstat_bin is None:
        raise RuntimeError("nextstat binary not found")

    import tempfile
    payload = {"times": times_list, "events": events_list,
               "covariates": covariates, "covariate_names": names}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        tmp_path = f.name
    r = subprocess.run(
        [nextstat_bin, "churn", "risk-model", "-i", tmp_path],
        capture_output=True, text=True, check=True,
    )
    Path(tmp_path).unlink(missing_ok=True)
    out = json.loads(r.stdout)
    return {
        "coefficients": [c["coefficient"] for c in out["coefficients"]],
        "hazard_ratios": [c["hazard_ratio"] for c in out["coefficients"]],
        "se": [c["se"] for c in out["coefficients"]],
    }


# ---------------------------------------------------------------------------
# Logistic regression via sklearn (biased under censoring)
# ---------------------------------------------------------------------------

def fit_logreg_sklearn(data: dict, horizon: float) -> dict:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None

    # Binary outcome: event observed before horizon.
    # BUG: censored observations before horizon are labeled 0 (retained),
    # even though some of them would have churned if observed longer.
    y_binary = ((data["events"]) & (data["times"] <= horizon)).astype(int)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=np.inf)
    clf.fit(data["x"], y_binary)

    return {
        "coefficients": clf.coef_[0].tolist(),
        "odds_ratios": np.exp(clf.coef_[0]).tolist(),
    }


# ---------------------------------------------------------------------------
# Oracle logistic regression (no censoring, uses true event times)
# ---------------------------------------------------------------------------

def fit_logreg_oracle(data: dict, horizon: float) -> dict:
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None

    # Oracle: use true event times (no censoring).
    y_binary = (data["event_times"] <= horizon).astype(int)

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=np.inf)
    clf.fit(data["x"], y_binary)

    return {
        "coefficients": clf.coef_[0].tolist(),
        "odds_ratios": np.exp(clf.coef_[0]).tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(results: list[dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed", file=sys.stderr)
        return

    true_beta = np.array([0.5, -0.3, 0.8, -0.1])
    covariate_names = ["x1", "x2", "x3", "x4"]
    censoring_levels = [r["censoring_target"] for r in results]

    fig, axes = plt.subplots(1, len(true_beta), figsize=(16, 5), sharey=False)

    for j, (ax, name) in enumerate(zip(axes, covariate_names)):
        # True value.
        ax.axhline(y=true_beta[j], color="black", linestyle="--", linewidth=1.5,
                    label="True β" if j == 0 else None, alpha=0.7)

        cox_betas = [r["cox_ph"]["coefficients"][j] for r in results]
        lr_betas = [r["logreg"]["coefficients"][j] for r in results if r.get("logreg")]
        oracle_betas = [r["oracle"]["coefficients"][j] for r in results if r.get("oracle")]

        x_pos = np.arange(len(censoring_levels))
        width = 0.25

        ax.bar(x_pos - width, cox_betas, width, color="#2563eb", alpha=0.8,
               label="Cox PH (NextStat)" if j == 0 else None)
        if lr_betas:
            ax.bar(x_pos, lr_betas, width, color="#dc2626", alpha=0.8,
                   label="LogReg (sklearn)" if j == 0 else None)
        if oracle_betas:
            ax.bar(x_pos + width, oracle_betas, width, color="#16a34a", alpha=0.8,
                   label="LogReg Oracle" if j == 0 else None)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{int(c*100)}%" for c in censoring_levels])
        ax.set_xlabel("Censoring rate", fontsize=11)
        ax.set_ylabel("Estimated β", fontsize=11)
        ax.set_title(f"β({name}), true = {true_beta[j]}", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Survival (Cox PH) vs Binary Classification (LogReg) — Censoring Bias",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


def make_bias_table(results: list[dict]):
    """Print a bias summary table to stdout."""
    true_beta = np.array([0.5, -0.3, 0.8, -0.1])
    names = ["x1", "x2", "x3", "x4"]

    print("\n" + "=" * 80)
    print(f"{'Censoring':>10} | {'Covariate':>10} | {'True β':>8} | {'Cox PH β':>9} | {'LogReg β':>9} | {'Cox bias':>9} | {'LR bias':>9}")
    print("-" * 80)

    for r in results:
        cens = r["censoring_target"]
        for j, name in enumerate(names):
            cox_b = r["cox_ph"]["coefficients"][j]
            lr_b = r["logreg"]["coefficients"][j] if r.get("logreg") else float("nan")
            cox_bias = cox_b - true_beta[j]
            lr_bias = lr_b - true_beta[j]
            print(f"{cens*100:>9.0f}% | {name:>10} | {true_beta[j]:>8.3f} | {cox_b:>9.4f} | {lr_b:>9.4f} | {cox_bias:>+9.4f} | {lr_bias:>+9.4f}")
    print("=" * 80)

    # Summary: mean absolute bias.
    print("\nMean absolute bias across all 4 covariates:")
    for r in results:
        cens = r["censoring_target"]
        cox_mab = np.mean(np.abs(np.array(r["cox_ph"]["coefficients"]) - true_beta))
        lr_mab = np.mean(np.abs(np.array(r["logreg"]["coefficients"]) - true_beta)) if r.get("logreg") else float("nan")
        ratio = lr_mab / cox_mab if cox_mab > 0 and not np.isnan(lr_mab) else float("nan")
        print(f"  {cens*100:.0f}% censoring: Cox PH MAB = {cox_mab:.4f}, LogReg MAB = {lr_mab:.4f} ({ratio:.1f}× more biased)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Survival vs classification bias benchmark")
    parser.add_argument("--n-obs", type=int, default=50_000, help="Number of observations")
    parser.add_argument("--n-repeats", type=int, default=20, help="Monte Carlo repeats for stable estimates")
    parser.add_argument("--out-dir", type=str, default="bench_results", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    censoring_targets = [0.40, 0.60, 0.80]
    true_beta = np.array([0.5, -0.3, 0.8, -0.1])

    results = []

    for cens_target in censoring_targets:
        print(f"\n=== Censoring target: {cens_target*100:.0f}% ===")

        cox_betas_all = []
        lr_betas_all = []
        oracle_betas_all = []

        for rep in range(args.n_repeats):
            data = generate_data_with_censoring(args.n_obs, cens_target, seed=42 + rep * 100)
            horizon = float(np.median(data["event_times"]))

            if rep == 0:
                print(f"  Actual censoring: {data['actual_censoring']*100:.1f}%, horizon={horizon:.2f}")

            cox_result = fit_cox_ph_nextstat(data)
            cox_betas_all.append(cox_result["coefficients"])

            lr_result = fit_logreg_sklearn(data, horizon)
            if lr_result:
                lr_betas_all.append(lr_result["coefficients"])

            oracle_result = fit_logreg_oracle(data, horizon)
            if oracle_result:
                oracle_betas_all.append(oracle_result["coefficients"])

        # Average across repeats.
        cox_mean = np.mean(cox_betas_all, axis=0).tolist()
        lr_mean = np.mean(lr_betas_all, axis=0).tolist() if lr_betas_all else [float("nan")] * 4
        oracle_mean = np.mean(oracle_betas_all, axis=0).tolist() if oracle_betas_all else [float("nan")] * 4

        cox_mab = float(np.mean(np.abs(np.array(cox_mean) - true_beta)))
        lr_mab = float(np.mean(np.abs(np.array(lr_mean) - true_beta)))

        print(f"  Cox PH  mean β: {[f'{b:.4f}' for b in cox_mean]}  MAB={cox_mab:.4f}")
        print(f"  LogReg  mean β: {[f'{b:.4f}' for b in lr_mean]}  MAB={lr_mab:.4f}")
        if lr_mab > 0 and cox_mab > 0:
            print(f"  LogReg is {lr_mab/cox_mab:.1f}× more biased than Cox PH")

        entry = {
            "censoring_target": cens_target,
            "n_obs": args.n_obs,
            "n_repeats": args.n_repeats,
            "cox_ph": {"coefficients": cox_mean, "mab": cox_mab},
            "logreg": {"coefficients": lr_mean, "mab": lr_mab},
            "oracle": {"coefficients": oracle_mean, "mab": float(np.mean(np.abs(np.array(oracle_mean) - true_beta)))},
            "true_beta": true_beta.tolist(),
        }
        results.append(entry)

    # Bias table.
    make_bias_table(results)

    # Save JSON.
    json_path = out_dir / "survival_vs_classification.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {json_path}")

    # Plot.
    make_plot(results, out_dir / "survival_vs_classification.png")


if __name__ == "__main__":
    main()
