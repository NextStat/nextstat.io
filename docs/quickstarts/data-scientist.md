# Quickstart (Data Scientist): Logistic Regression in 5 Minutes

What you'll do:

- generate a small synthetic classification dataset
- fit a logistic regression with NextStat
- print coefficients + basic metrics (accuracy, log-loss)

## Install

```bash
pip install nextstat
```

Repo dev mode (no pip install):

```bash
cd /Users/andresvlc/WebDev/nextstat.io
export PYTHONPATH=bindings/ns-py/python
```

## Run

```bash
python docs/quickstarts/code/ds_logistic.py
```

Output:

- prints fitted coefficients + standard errors
- prints accuracy + log-loss

Next steps:

- use formula interface: `nextstat.glm.logistic.from_formula(...)`
- add ridge/MAP for separation stability: `l2=...`

