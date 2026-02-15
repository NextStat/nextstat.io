.PHONY: venv install hep hep-fit pharma pharma-fit bayesian ml econometrics validate snapshot

PY ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

venv:
	$(PY) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r env/python/requirements.txt

hep: install
	$(PYBIN) suites/hep/suite.py --deterministic --out-dir out/hep

hep-fit: install
	$(PYBIN) suites/hep/suite.py --deterministic --fit --fit-repeat 3 --out-dir out/hep_fit

pharma: install
	$(PYBIN) suites/pharma/suite.py --deterministic --out-dir out/pharma

pharma-fit: install
	$(PYBIN) suites/pharma/suite.py --deterministic --fit --fit-repeat 3 --out-dir out/pharma_fit

bayesian: install
	$(PYBIN) suites/bayesian/suite.py --deterministic --out-dir out/bayesian

ml: install
	$(PYBIN) suites/ml/suite.py --deterministic --out-dir out/ml

econometrics: install
	$(PYBIN) suites/econometrics/suite.py --deterministic --out-dir out/econometrics

validate: install
	@if [ -d out ]; then $(PYBIN) scripts/validate_artifacts.py --strict out; fi
	@if [ -d manifests/snapshots ]; then $(PYBIN) scripts/validate_artifacts.py --strict manifests/snapshots; fi

snapshot: install
	$(PYBIN) scripts/publish_snapshot.py --deterministic --fit --fit-repeat 3
