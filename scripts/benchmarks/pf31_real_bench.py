#!/usr/bin/env python3
"""PF3.1 Real Benchmark: CB+Exp, ~2M events, RunPod 4x A40.

VRAM constraint: 2M events x 8B = 16MB/toy. A40 48GB -> max ~2500 toys/GPU.
Strategy:
  - CPU baseline: 2000 toys (fair comparison) + 10K toys (throughput)
  - Device-resident GPU: 2000 toys on 1/2/3/4 GPUs (scaling test)
  - Device-resident 4 GPU: 8000 toys (throughput test, 2000/GPU)
"""
import subprocess, json, time, os, sys, multiprocessing

BIN = "/root/nextstat.io/target/release/nextstat"
SPEC = "/tmp/pf31_real/spec.json"
OUTDIR = "/tmp/pf31_real/results"
os.makedirs(OUTDIR, exist_ok=True)

def run_bench(label, cmd, outfile, timeout=7200):
    print("  {} ...".format(label), end="", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        print(" TIMEOUT ({:.0f}s)".format(wall))
        return wall, None
    wall = time.time() - t0

    if proc.returncode != 0:
        err = proc.stderr.strip()[:300]
        print(" FAILED ({:.1f}s) -- {}".format(wall, err))
        with open(outfile + ".err", "w") as f:
            f.write(proc.stderr)
        return wall, None

    with open(outfile, "w") as f:
        f.write(proc.stdout)

    try:
        d = json.loads(proc.stdout)
        r = d["results"]
        conv = r["n_converged"]
        total = r["n_toys"]
        errs = r["n_error"]
        print(" {:.1f}s ({}/{} conv, {} err)".format(wall, conv, total, errs))
        return wall, d
    except (json.JSONDecodeError, KeyError) as e:
        print(" {:.1f}s (parse error: {})".format(wall, e))
        return wall, None


n_cpu = multiprocessing.cpu_count()
print("=" * 70)
print("PF3.1 REAL BENCHMARK: CB+Exp ~2M events")
print("RunPod 4x A40 48GB | {} vCPU".format(n_cpu))
print("=" * 70)

results = {}

# ========== 1. CPU BASELINE (2000 toys — fair GPU comparison) ==========
N_FAIR = 2000
print("\n--- CPU BASELINE ({} toys) ---".format(N_FAIR))
cmd = [BIN, "unbinned-fit-toys", "--config", SPEC,
       "--n-toys", str(N_FAIR), "--seed", "42"]
wall, d = run_bench("CPU Rayon ({} cores)".format(n_cpu), cmd,
                    os.path.join(OUTDIR, "cpu_{}.json".format(N_FAIR)))
results["cpu_2k"] = wall

# ========== 2. DEVICE-RESIDENT GPU SCALING (2000 toys) ==========
print("\n--- DEVICE-RESIDENT GPU ({} toys, scaling test) ---".format(N_FAIR))
for n_gpu in [1, 2, 3, 4]:
    devs = ",".join(str(i) for i in range(n_gpu))
    label = "Device-resident {} GPU [{}]".format(n_gpu, devs)
    outfile = os.path.join(OUTDIR, "dr_{}gpu_{}t.json".format(n_gpu, N_FAIR))
    cmd = [BIN, "unbinned-fit-toys", "--config", SPEC,
           "--n-toys", str(N_FAIR), "--seed", "42",
           "--gpu", "cuda", "--gpu-sample-toys",
           "--gpu-devices", devs, "--gpu-shards", str(n_gpu)]
    wall, d = run_bench(label, cmd, outfile, timeout=3600)
    results["dr_{}gpu_2k".format(n_gpu)] = wall

# ========== 3. CPU BASELINE (10K toys — throughput reference) ==========
# Already done from previous run: 5590.8s, 6438/10000 conv
# Re-read from saved file if available
cpu_10k_file = os.path.join(OUTDIR, "cpu_10000.json")
if os.path.exists(cpu_10k_file):
    print("\n--- CPU 10K toys: using cached result ---")
    results["cpu_10k"] = 5590.8
    print("  CPU 10K: 5590.8s (cached from previous run)")
else:
    print("\n--- CPU BASELINE (10000 toys) ---")
    cmd = [BIN, "unbinned-fit-toys", "--config", SPEC,
           "--n-toys", "10000", "--seed", "42"]
    wall, d = run_bench("CPU Rayon ({} cores)".format(n_cpu), cmd,
                        cpu_10k_file)
    results["cpu_10k"] = wall

# ========== 4. DEVICE-RESIDENT 4 GPU MAX THROUGHPUT (8000 toys) ==========
N_MAX = 8000
print("\n--- DEVICE-RESIDENT 4 GPU MAX THROUGHPUT ({} toys, 2000/GPU) ---".format(N_MAX))
cmd = [BIN, "unbinned-fit-toys", "--config", SPEC,
       "--n-toys", str(N_MAX), "--seed", "42",
       "--gpu", "cuda", "--gpu-sample-toys",
       "--gpu-devices", "0,1,2,3", "--gpu-shards", "4"]
wall, d = run_bench("4 GPU x 2000 toys", cmd,
                    os.path.join(OUTDIR, "dr_4gpu_{}t.json".format(N_MAX)), timeout=3600)
results["dr_4gpu_8k"] = wall

# ========== SUMMARY ==========
print("\n" + "=" * 70)
print("SCALING SUMMARY: CB+Exp ~2M events, RunPod 4x A40")
print("=" * 70)

cpu2k = results.get("cpu_2k")
cpu10k = results.get("cpu_10k")
dr1 = results.get("dr_1gpu_2k")

print("\n{:<40} {:>10} {:>12} {:>12}".format(
    "Configuration", "Wall (s)", "vs CPU", "vs 1GPU"))
print("-" * 76)

if cpu2k:
    print("{:<40} {:>10.1f} {:>12} {:>12}".format(
        "CPU Rayon {} cores (2K toys)".format(n_cpu), cpu2k, "baseline", "-"))

for n_gpu in [1, 2, 3, 4]:
    key = "dr_{}gpu_2k".format(n_gpu)
    t = results.get(key)
    if not t:
        continue
    vs_cpu = "{:.2f}x".format(cpu2k / t) if cpu2k else "-"
    vs_1gpu = "{:.2f}x".format(dr1 / t) if dr1 else "-"
    print("{:<40} {:>10.1f} {:>12} {:>12}".format(
        "Device-res {} GPU (2K toys)".format(n_gpu), t, vs_cpu, vs_1gpu))

print()
if cpu10k:
    print("{:<40} {:>10.1f} {:>12} {:>12}".format(
        "CPU Rayon {} cores (10K toys)".format(n_cpu), cpu10k, "baseline", "-"))
t_8k = results.get("dr_4gpu_8k")
if t_8k and cpu10k:
    # Normalize: 8K toys GPU vs 10K toys CPU -> per-toy rate comparison
    rate_cpu = cpu10k / 10000
    rate_gpu = t_8k / 8000
    eff = rate_cpu / rate_gpu
    print("{:<40} {:>10.1f} {:>12} {:>12}".format(
        "Device-res 4 GPU (8K toys)", t_8k,
        "{:.2f}x eff".format(eff), "-"))
elif t_8k:
    print("{:<40} {:>10.1f} {:>12} {:>12}".format(
        "Device-res 4 GPU (8K toys)", t_8k, "-", "-"))

print("\nDone. Results in: {}".format(OUTDIR))
