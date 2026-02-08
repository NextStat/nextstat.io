# Quickstart (Biologist): One-Compartment Oral PK Fit

What you'll do:

- generate a small synthetic PK dataset (oral dosing, first-order absorption)
- fit parameters by MLE
- compare predictions to observations

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
python docs/quickstarts/code/bio_pk_1c_oral.py
```

Output:

- prints estimated params + uncertainties
- writes a small JSON artifact: `docs/quickstarts/out/bio_pk_1c_oral_result.json`

Next steps:

- try population PK (NLME): `OneCompartmentOralPkNlmeModel(...)`

