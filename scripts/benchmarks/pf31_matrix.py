#!/usr/bin/env python3
"""PF3.1-OPT2 Multi-GPU Scaling Benchmark Matrix.

Tests 1->2->3->4 GPU scaling for device-resident sharded toy pipeline.
"""
import subprocess, json, time, os

from _parse_utils import parse_json_stdout

BIN = "/root/nextstat.io/target/release/nextstat"
SPEC = "/tmp/pf31_bench/spec.json"
OUTDIR = "/tmp/pf31_bench/results"
os.makedirs(OUTDIR, exist_ok=True)

print("=== PF3.1-OPT2 Multi-GPU Scaling Matrix ===")
print("GPUs: 4x A40")
print()

results = {}

for n_toys in [200, 500]:
    tag = "{} toys, ~10K events, device-resident sharded".format(n_toys)
    print("--- {} ---".format(tag))
    results[n_toys] = {}
    for n_gpu in [1, 2, 3, 4]:
        devs = ",".join(str(i) for i in range(n_gpu))
        outfile = os.path.join(OUTDIR, "{}gpu_{}toys.json".format(n_gpu, n_toys))

        cmd = [
            BIN, "unbinned-fit-toys",
            "--config", SPEC,
            "--n-toys", str(n_toys),
            "--gpu", "cuda",
            "--gpu-sample-toys",
            "--gpu-devices", devs,
            "--gpu-shards", str(n_gpu),
            "--seed", "42",
        ]

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall_s = time.time() - t0

        if proc.returncode != 0:
            err = proc.stderr.strip()[:200]
            print("  {} GPU: FAILED -- {}".format(n_gpu, err))
            continue

        with open(outfile, "w") as f:
            f.write(proc.stdout)

        d = parse_json_stdout(proc.stdout)
        r = d["results"]
        conv = r["n_converged"]
        errs = r["n_error"]
        results[n_toys][n_gpu] = wall_s

        print("  {} GPU: {:7.2f}s  ({}/{} converged, {} errors)".format(
            n_gpu, wall_s, conv, n_toys, errs))
    print()

print("=== SCALING SUMMARY ===")
print()
print("{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
    "Workload", "1GPU", "2GPU", "3GPU", "4GPU", "2x_spd", "4x_spd"))
print("-" * 72)
for n_toys in [200, 500]:
    row = results.get(n_toys, {})
    t1 = row.get(1)
    vals = []
    spdups = []
    for ng in [1, 2, 3, 4]:
        t = row.get(ng)
        vals.append("{:.2f}s".format(t) if t else "-")
        if t and t1:
            spdups.append("{:.2f}x".format(t1 / t))
        else:
            spdups.append("-")
    label = "{}t x 10Kev".format(n_toys)
    print("{:<20} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        label, vals[0], vals[1], vals[2], vals[3], spdups[1], spdups[3]))

print()
print("=== Done ===")
