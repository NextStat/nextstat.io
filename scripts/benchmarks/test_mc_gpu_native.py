"""Test multi-channel GPU-native L-BFGS vs lockstep on CUDA."""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(path, values, obs_name, bounds):
    table = pa.table(
        {obs_name: pa.array(values.astype(np.float64), type=pa.float64())}
    )
    table = table.replace_schema_metadata(
        {
            b"nextstat.schema_version": b"nextstat_unbinned_events_v1",
            b"nextstat.observables": json.dumps(
                [{"name": obs_name, "bounds": list(bounds)}],
                separators=(",", ":"),
            ).encode(),
        }
    )
    pq.write_table(table, str(path))


def gen_channel(n_sig, n_bkg, seed):
    rng = np.random.default_rng(seed)
    sig = rng.normal(91.0, 2.5, size=n_sig * 2)
    sig = sig[(sig >= 60.0) & (sig <= 120.0)][:n_sig]
    u = rng.uniform(0.0, 1.0, size=n_bkg)
    lo, hi, lam = 60.0, 120.0, -0.03
    bkg = np.log(np.exp(lam * lo) + u * (np.exp(lam * hi) - np.exp(lam * lo))) / lam
    return np.concatenate([sig, bkg])


def main():
    cli = os.environ.get(
        "NS_CLI_BIN", "./target/release/nextstat"
    )
    n_toys = 200
    seed = 42

    with tempfile.TemporaryDirectory() as td:
        d1 = os.path.join(td, "ch1.parquet")
        d2 = os.path.join(td, "ch2.parquet")
        write_parquet(d1, gen_channel(120, 360, 17), "mass", (60.0, 120.0))
        write_parquet(d2, gen_channel(60, 180, 99), "mass", (60.0, 120.0))

        spec = {
            "schema_version": "nextstat_unbinned_spec_v0",
            "model": {
                "poi": "mu",
                "parameters": [
                    {"name": "mu", "init": 1.0, "bounds": [0.0, 5.0]},
                    {"name": "mu_sig", "init": 91.0, "bounds": [85.0, 95.0]},
                    {"name": "sigma_sig", "init": 2.5, "bounds": [0.5, 10.0]},
                    {"name": "lambda_bkg", "init": -0.03, "bounds": [-0.1, -0.001]},
                ],
            },
            "channels": [
                {
                    "name": "SR1",
                    "data": {"file": d1},
                    "observables": [{"name": "mass", "bounds": [60.0, 120.0]}],
                    "processes": [
                        {
                            "name": "sig1",
                            "pdf": {"type": "gaussian", "observable": "mass", "params": ["mu_sig", "sigma_sig"]},
                            "yield": {"type": "scaled", "base_yield": 120.0, "scale": "mu"},
                        },
                        {
                            "name": "bkg1",
                            "pdf": {"type": "exponential", "observable": "mass", "params": ["lambda_bkg"]},
                            "yield": {"type": "fixed", "value": 360.0},
                        },
                    ],
                },
                {
                    "name": "SR2",
                    "data": {"file": d2},
                    "observables": [{"name": "mass", "bounds": [60.0, 120.0]}],
                    "processes": [
                        {
                            "name": "sig2",
                            "pdf": {"type": "gaussian", "observable": "mass", "params": ["mu_sig", "sigma_sig"]},
                            "yield": {"type": "scaled", "base_yield": 60.0, "scale": "mu"},
                        },
                        {
                            "name": "bkg2",
                            "pdf": {"type": "exponential", "observable": "mass", "params": ["lambda_bkg"]},
                            "yield": {"type": "fixed", "value": 180.0},
                        },
                    ],
                },
            ],
        }
        sp = os.path.join(td, "spec.json")
        with open(sp, "w") as f:
            json.dump(spec, f, indent=2)

        # 1) Lockstep
        print(f"=== LOCKSTEP (2-channel, {n_toys} toys) ===")
        t0 = time.time()
        r1 = subprocess.run(
            [cli, "unbinned-fit-toys", "--config", sp,
             "--n-toys", str(n_toys), "--seed", str(seed), "--gpu", "cuda"],
            capture_output=True, text=True, timeout=120,
        )
        t_ls = time.time() - t0
        if r1.returncode != 0:
            print("FAILED:", r1.stderr[:500])
            sys.exit(1)
        d1r = json.loads(r1.stdout)
        res_ls = d1r["results"]
        nc_ls = res_ls["n_converged"]
        print(f"  Converged: {nc_ls}/{n_toys}, time: {t_ls:.2f}s")

        # 2) GPU-native
        print(f"\n=== GPU-NATIVE (2-channel, {n_toys} toys) ===")
        t0 = time.time()
        r2 = subprocess.run(
            [cli, "unbinned-fit-toys", "--config", sp,
             "--n-toys", str(n_toys), "--seed", str(seed),
             "--gpu", "cuda", "--gpu-native"],
            capture_output=True, text=True, timeout=120,
        )
        t_gn = time.time() - t0
        if r2.returncode != 0:
            print("FAILED:", r2.stderr[:1000])
            sys.exit(1)
        d2r = json.loads(r2.stdout)
        res_gn = d2r["results"]
        nc_gn = res_gn["n_converged"]
        print(f"  Converged: {nc_gn}/{n_toys}, time: {t_gn:.2f}s")

        # Parity comparison
        conv_both = [
            i for i in range(n_toys)
            if res_ls["converged"][i] and res_gn["converged"][i]
        ]
        nll_diffs = [abs(res_gn["nll"][i] - res_ls["nll"][i]) for i in conv_both]
        mu_diffs = [abs(res_gn["poi_hat"][i] - res_ls["poi_hat"][i]) for i in conv_both]

        print(f"\n=== PARITY (both converged: {len(conv_both)}) ===")
        if nll_diffs:
            print(f"  Max |ΔNLL|:  {max(nll_diffs):.2e}")
            print(f"  Mean |ΔNLL|: {np.mean(nll_diffs):.2e}")
            print(f"  Max |Δmu|:   {max(mu_diffs):.2e}")
            print(f"  Mean |Δmu|:  {np.mean(mu_diffs):.2e}")
        print(f"\n  Lockstep:   {t_ls:.2f}s")
        print(f"  GPU-native: {t_gn:.2f}s")
        print(f"  Speedup:    {t_ls / t_gn:.1f}x")


if __name__ == "__main__":
    main()
