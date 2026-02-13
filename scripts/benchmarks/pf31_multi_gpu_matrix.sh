#!/usr/bin/env bash
# PF3.1-OPT2 Multi-GPU Scaling Benchmark Matrix
# Tests 1→2→3→4 GPU scaling for device-resident sharded toy pipeline.
# Expects setup_vastai_multi_gpu.sh to have been run first.
set -euo pipefail

REPO="$HOME/nextstat.io"
BIN="$REPO/target/release/nextstat"
SPEC="/tmp/pf31_bench/spec.json"
OUTDIR="/tmp/pf31_bench/results_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTDIR"

N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "=== PF3.1-OPT2 Multi-GPU Scaling Matrix ==="
echo "GPUs detected: $N_GPU"
echo "Output dir: $OUTDIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Build device list strings: "0", "0,1", "0,1,2", "0,1,2,3"
declare -a DEVICE_LISTS
for ((n=1; n<=N_GPU && n<=4; n++)); do
    devs=""
    for ((d=0; d<n; d++)); do
        if [ -n "$devs" ]; then devs="$devs,"; fi
        devs="$devs$d"
    done
    DEVICE_LISTS+=("$devs")
done

# Benchmark configurations
# Format: n_toys events_approx
CONFIGS=(
    "200 10000"
    "500 10000"
    "200 50000"
)

echo "=== Matrix: ${#DEVICE_LISTS[@]} GPU configs × ${#CONFIGS[@]} workloads ==="
echo ""

for config in "${CONFIGS[@]}"; do
    read -r N_TOYS EVENTS <<< "$config"
    echo "--- Workload: ${N_TOYS} toys, ~${EVENTS} events ---"

    for devs in "${DEVICE_LISTS[@]}"; do
        n_dev=$(echo "$devs" | tr ',' '\n' | wc -l)
        tag="${n_dev}gpu_${N_TOYS}toys_${EVENTS}ev"
        outfile="$OUTDIR/${tag}.json"

        echo -n "  ${n_dev} GPU(s) [${devs}], shards=${n_dev}: "

        # Device-resident sharded path: --gpu-sample-toys --gpu-shards N --gpu-devices ...
        START_S=$(date +%s%N)
        "$BIN" unbinned-fit-toys \
            --config "$SPEC" \
            --n-toys "$N_TOYS" \
            --gpu cuda \
            --gpu-sample-toys \
            --gpu-devices "$devs" \
            --gpu-shards "$n_dev" \
            --seed 42 \
            > "$outfile" 2>/dev/null
        END_S=$(date +%s%N)
        WALL_MS=$(( (END_S - START_S) / 1000000 ))

        # Extract results
        python3 -c "
import json, sys
with open('$outfile') as f:
    d = json.load(f)
r = d['results']
wall = $WALL_MS / 1000.0
print(f'{wall:.2f}s wall, {r[\"n_converged\"]}/{r[\"n_toys\"]} converged, {r[\"n_error\"]} errors')
"
    done
    echo ""
done

# Summary table
echo "=== SCALING SUMMARY ==="
echo ""
python3 << 'PYEOF'
import json, glob, os, sys

outdir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUTDIR", "/tmp/pf31_bench/results")
# Find output dir from env
PYEOF_END=1

python3 -c "
import json, glob, os

outdir = '$OUTDIR'
files = sorted(glob.glob(os.path.join(outdir, '*.json')))
if not files:
    print('No result files found')
    exit(0)

results = []
for f in files:
    base = os.path.basename(f).replace('.json', '')
    parts = base.split('_')
    n_gpu = int(parts[0].replace('gpu', ''))
    n_toys = int(parts[1].replace('toys', ''))
    n_ev = int(parts[2].replace('ev', ''))
    with open(f) as fh:
        d = json.load(fh)
    r = d['results']
    results.append({
        'n_gpu': n_gpu, 'n_toys': n_toys, 'n_ev': n_ev,
        'converged': r['n_converged'], 'total': r['n_toys'],
        'file': f
    })

# Group by workload
from collections import defaultdict
groups = defaultdict(list)
for r in results:
    key = (r['n_toys'], r['n_ev'])
    groups[key].append(r)

print(f'{'Workload':<25} {'1-GPU':>10} {'2-GPU':>10} {'3-GPU':>10} {'4-GPU':>10} {'Speedup 1→4':>12}')
print('-' * 80)
for (n_toys, n_ev), items in sorted(groups.items()):
    items.sort(key=lambda x: x['n_gpu'])
    times = {r['n_gpu']: r for r in items}
    line = f'{n_toys}t × {n_ev}ev'
    # We don't have wall times in the JSON, so just show convergence
    vals = []
    for ng in [1, 2, 3, 4]:
        if ng in times:
            t = times[ng]
            vals.append(f'{t[\"converged\"]}/{t[\"total\"]}')
        else:
            vals.append('-')
    print(f'{line:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}')

print()
print('NOTE: Wall times are printed above in the per-run output.')
print(f'Full results in: {outdir}/')
"

echo ""
echo "=== Benchmark complete ==="
echo "Copy results: scp -r $OUTDIR ."
