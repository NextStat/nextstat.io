# HTCondor Examples

Example submit files and scripts for running NextStat on HTCondor batch clusters.

See the full deployment guide: `docs/guides/htcondor-hpc.md`

## Files

| File | Description |
|------|-------------|
| `wrapper.sh` | Wrapper script for portable venv (Strategy B) |
| `simple_fit.sub` | Single fit job (Strategy A: shared filesystem) |
| `scan_array.sub` | Profile scan array job (1 mu-value per job) |
| `scan_point.py` | Worker script for scan array |
| `merge_scan.py` | Merge scan results on submit node |
| `toys_batch.sub` | Batch toy CLs (20 jobs x 500 toys) |
| `toys_job.py` | Worker script for toy batch |
| `toys_dag.dag` | DAGMan: toys -> merge pipeline |
| `merge_toys.sub` | Merge job — runs on submit node (universe=local) |
| `merge_toys_local.sh` | Wrapper for local merge |
| `merge_toys.py` | Merge toy results |
| `trex_hwfsdp.sub` | Production-style TRExFitter replicas |

## Quick Start

```bash
# 1. Create portable venv
python3 -m venv --copies nextstat-env
source nextstat-env/bin/activate
pip install nextstat
tar czf nextstat-env.tar.gz nextstat-env/

# 2. Prepare workspace
cp /path/to/workspace.json .

# 3. Submit
mkdir -p logs
condor_submit scan_array.sub

# 4. After completion, merge
python3 merge_scan.py scan_*.json -o scan_merged.json
```
