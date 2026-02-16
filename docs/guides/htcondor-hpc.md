---
title: "HTCondor & HPC Cluster Deployment"
status: stable
---

# HTCondor & HPC Cluster Deployment

This guide covers deploying NextStat on HTCondor clusters commonly used in HEP (LXPLUS, Tier-2, institutional batch systems). The same patterns apply to other batch schedulers (Slurm, PBS/Torque) with minor syntax changes.

## Do I Need to Compile NextStat?

**No.** Pre-built wheels are published to PyPI for every release:

```bash
pip install nextstat
```

Supported platforms and Python versions:

| Platform | Architectures | Python |
|----------|---------------|--------|
| Linux (manylinux_2_17) | x86_64, aarch64 | 3.11, 3.12, 3.13, 3.14 |
| macOS | x86_64, aarch64 (Apple Silicon) | 3.11, 3.12, 3.13, 3.14 |
| Windows | x86_64 | 3.11, 3.12, 3.13, 3.14 |

manylinux_2_17 wheels work on any Linux with glibc >= 2.17, which includes:

- AlmaLinux 9 / RHEL 9 (glibc 2.34)
- CentOS 7 (glibc 2.17)
- Ubuntu 20.04+ (glibc 2.31+)
- LXPLUS (CentOS/AlmaLinux based)

If `pip install nextstat` takes more than 30 seconds, it is likely building from source. Check:

```bash
pip install nextstat --only-binary :all:
```

This forces pip to use a pre-built wheel and will fail fast if no compatible wheel exists.

## Deployment Strategies

Choose the strategy that matches your cluster's storage topology.

### Strategy A: Shared Filesystem (Recommended)

Best when: submit and worker nodes share a filesystem (NFS, AFS, CVMFS, EOS).

```bash
# On submit node — install once
python3 -m venv /shared/path/nextstat-env
source /shared/path/nextstat-env/bin/activate
pip install nextstat

# Verify
nextstat --version
python3 -c "import nextstat; print(nextstat.__version__)"
```

HTCondor jobs use the same path:

```
# job.sub
executable = /shared/path/nextstat-env/bin/python3
arguments  = my_analysis.py
getenv     = False
environment = "PATH=/shared/path/nextstat-env/bin:$$PATH"
queue 1
```

Or with the CLI:

```
executable = /shared/path/nextstat-env/bin/nextstat
arguments  = fit --input workspace.json --output fit_$(Process).json
queue 1
```

### Strategy B: Portable Virtualenv via File Transfer

Best when: worker nodes have no shared filesystem with submit node.

```bash
# 1. Create self-contained venv on submit node
python3 -m venv --copies nextstat-env
source nextstat-env/bin/activate
pip install nextstat

# 2. Package it
tar czf nextstat-env.tar.gz nextstat-env/

# 3. Create wrapper script (see docs/examples/htcondor/wrapper.sh)
```

HTCondor transfers and unpacks on each worker:

```
# job.sub
universe   = vanilla
executable = wrapper.sh
arguments  = my_analysis.py $(Process)

transfer_input_files = nextstat-env.tar.gz, my_analysis.py, workspace.json
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

request_cpus   = 1
request_memory = 4GB
request_disk   = 2GB

queue 100
```

See `docs/examples/htcondor/wrapper.sh` for the wrapper script.

**Note on portability:** The venv must be created on a machine with the same OS family and Python version as the worker nodes. An AlmaLinux 9 venv works on AlmaLinux 9 workers but may not work on Ubuntu workers.

### Strategy C: Apptainer / Singularity Container

Best when: you need full environment reproducibility, or workers have a different OS.

```bash
# Build container (submit node or build machine)
apptainer build nextstat.sif containers/nextstat.def

# Test locally
apptainer exec nextstat.sif nextstat --version
apptainer exec nextstat.sif python3 -c "import nextstat; print(nextstat.__version__)"
```

HTCondor with Apptainer:

```
# job.sub
universe   = vanilla
executable = wrapper_apptainer.sh
arguments  = my_analysis.py $(Process)

transfer_input_files = nextstat.sif, my_analysis.py, workspace.json
should_transfer_files = YES
when_to_transfer_output = ON_EXIT

# Apptainer needs more disk for the .sif
request_cpus   = 1
request_memory = 4GB
request_disk   = 4GB

queue 100
```

Wrapper script for Apptainer:

```bash
#!/bin/bash
apptainer exec nextstat.sif python3 "$1" --process-id "$2"
```

See `containers/nextstat.def` for the container definition.

## Common HTCondor Patterns

### Array Job: Profile Scan Points

Split a profile scan across workers, one mu-value per job:

```
# scan_array.sub
executable = wrapper.sh
arguments  = scan_point.py $(Process)

transfer_input_files = nextstat-env.tar.gz, scan_point.py, workspace.json
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = scan_$(Process).json

request_cpus   = 1
request_memory = 2GB
queue 201
```

Where `scan_point.py`:

```python
import sys, json, nextstat

process_id = int(sys.argv[1])
mu = 0.0 + process_id * (5.0 / 200)

with open("workspace.json") as f:
    ws_str = f.read()

model = nextstat.from_pyhf(ws_str)
result = nextstat.profile_scan(model, [mu])

with open(f"scan_{process_id}.json", "w") as f:
    json.dump({"mu": mu, **result}, f)
```

### Batch Toys with DAGMan

Run toy-based CLs in parallel, then merge:

```
# toys.dag
JOB TOYS toys_batch.sub
JOB MERGE merge_toys.sub
PARENT TOYS CHILD MERGE
```

See `docs/examples/htcondor/toys_dag.dag` for the complete DAGMan workflow.

### Multi-CPU Fit

For large models (100+ parameters), request multiple CPUs:

```
request_cpus = 8
environment = "RAYON_NUM_THREADS=8"
```

NextStat automatically uses Rayon for parallel gradient evaluation when `RAYON_NUM_THREADS > 1`.

## Troubleshooting

### `pip install nextstat` compiles from source

Possible causes:
- Python version not supported (< 3.11 or > 3.14)
- Platform not in wheel matrix (e.g., 32-bit Linux)
- Old pip that cannot parse modern wheel tags

Fix:

```bash
python3 --version          # Must be 3.11-3.14
pip install --upgrade pip  # Upgrade pip first
pip install nextstat --only-binary :all:
```

### Worker node has no internet access

Pre-download the wheel on the submit node:

```bash
pip download nextstat -d ./wheels/
# Transfer wheels/ to workers, then:
pip install --no-index --find-links=./wheels/ nextstat
```

### `ImportError: libpython3.12.so` on worker

The venv was created with `--copies` but the worker has a different Python. Solutions:
- Use `--copies` and ensure same Python version on workers
- Use Apptainer container (Strategy C) for full isolation
- Use shared filesystem (Strategy A)

### Apptainer: `FATAL: kernel too old`

The container was built on a newer kernel. Rebuild on a machine with the same kernel version as workers, or use `--fakeroot`.

## Performance Tips

- **Single-point fits**: 1 CPU, 2 GB RAM is sufficient for most models
- **Profile scans**: Parallelize across mu-values (1 job per point), not across CPUs
- **Toy-based CLs**: Use `--n-toys-per-job` to batch toys within each job, reducing job overhead
- **Large models (100+ params)**: Request 4-8 CPUs, set `RAYON_NUM_THREADS`
- **GPU toys**: If workers have GPUs, add `request_gpus = 1` and use `--gpu-sample-toys`
