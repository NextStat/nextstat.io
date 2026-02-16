# NextStat Container Definitions

Apptainer/Singularity definition files for deploying NextStat on HPC clusters.

## Files

| File | Base Image | GPU | Description |
|------|-----------|-----|-------------|
| `nextstat.def` | AlmaLinux 9 | No | CPU-only, minimal (~500 MB) |
| `nextstat-gpu.def` | nvidia/cuda:12.4.1-runtime-almalinux9 | CUDA 12.4 | GPU batch toys, differentiable inference |

## Build

```bash
# CPU-only
apptainer build nextstat.sif nextstat.def

# GPU (CUDA)
apptainer build nextstat-gpu.sif nextstat-gpu.def
```

## Run

```bash
# CLI
apptainer run nextstat.sif fit --input workspace.json

# Python script
apptainer exec nextstat.sif python3.12 my_analysis.py

# GPU (pass --nv to expose host drivers)
apptainer run --nv nextstat-gpu.sif hypotest-toys \
    --input workspace.json --mu 1.0 --n-toys 10000 --gpu-sample-toys
```

## HTCondor Integration

See `docs/guides/htcondor-hpc.md` (Strategy C) for using containers with HTCondor.
