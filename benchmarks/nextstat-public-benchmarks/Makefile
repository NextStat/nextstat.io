.PHONY: venv install hep hep-fit pharma pharma-fit validate snapshot

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

validate: install
	$(PYBIN) - <<'PY'
		import json
		from pathlib import Path
		import jsonschema
		schema_case = json.loads(Path("manifests/schema/benchmark_result_v1.schema.json").read_text())
		schema_suite = json.loads(Path("manifests/schema/benchmark_suite_result_v1.schema.json").read_text())
		suite = json.loads(Path("out/hep/hep_suite.json").read_text())
		jsonschema.validate(suite, schema_suite)
		for e in suite["cases"]:
		    inst = json.loads((Path("out/hep") / e["path"]).read_text())
		    jsonschema.validate(inst, schema_case)
		print("benchmark_suite_result_v1: schema ok")
		schema_pharma_case = json.loads(Path("manifests/schema/pharma_benchmark_result_v1.schema.json").read_text())
		schema_pharma_suite = json.loads(Path("manifests/schema/pharma_benchmark_suite_result_v1.schema.json").read_text())
		psuite = json.loads(Path("out/pharma/pharma_suite.json").read_text())
		jsonschema.validate(psuite, schema_pharma_suite)
		for e in psuite["cases"]:
		    inst = json.loads((Path("out/pharma") / e["path"]).read_text())
		    jsonschema.validate(inst, schema_pharma_case)
		print("pharma_benchmark_suite_result_v1: schema ok")

		schema_baseline = json.loads(Path("manifests/schema/baseline_manifest_v1.schema.json").read_text())
		schema_index = json.loads(Path("manifests/schema/snapshot_index_v1.schema.json").read_text())
		schema_rep = json.loads(Path("manifests/schema/replication_report_v1.schema.json").read_text())
		snap_root = Path("manifests/snapshots")
		if snap_root.exists():
		    n = 0
		    for d in sorted(p for p in snap_root.iterdir() if p.is_dir()):
		        baseline = d / "baseline_manifest.json"
		        if baseline.exists():
		            jsonschema.validate(json.loads(baseline.read_text()), schema_baseline)
		        index = d / "snapshot_index.json"
		        if index.exists():
		            jsonschema.validate(json.loads(index.read_text()), schema_index)
		        rep1 = d / "replication_report.json"
		        rep2 = d / "replication" / "replication_report.json"
		        rep = rep1 if rep1.exists() else rep2 if rep2.exists() else None
		        if rep is not None:
		            jsonschema.validate(json.loads(rep.read_text()), schema_rep)
		        n += 1
		    if n:
		        print(f"snapshots: schema ok ({n})")
		PY

snapshot: install
	$(PYBIN) scripts/publish_snapshot.py --deterministic --fit --fit-repeat 3
