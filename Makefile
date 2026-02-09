.PHONY: \
	apex2-baseline-record \
	apex2-baseline-compare \
	apex2-pre-release-gate \
	validation-pack \
	rust-slow-tests \
	rust-very-slow-tests \
	trex-spec-validate \
	trex-spec-run \
	trex-spec-baseline-record \
	trex-spec-baseline-compare \
	hepdata-fetch \
	hepdata-pytest \
	hepdata-root-profile-smoke \
	pyhf-audit \
	apex2-root-prereq \
	apex2-root-baseline-record \
	apex2-root-cases \
	apex2-condor-render-root-array-sub \
	apex2-root-aggregate \
	apex2-root-suite-compare-perf \
	apex2-root-suite-compare-latest \
	playground-build-wasm \
	playground-serve

PY ?= python3
PYTHONPATH ?= bindings/ns-py/python
RECORD_ARGS ?=
COMPARE_ARGS ?=
TREX_SPEC ?= docs/specs/trex/canonical/histfactory_fixture_baseline.yaml
TREX_SCHEMA ?= docs/schemas/trex/analysis_spec_v0.schema.json
TREX_RECORD_OUT_DIR ?= tmp/baselines
TREX_RECORD_ARGS ?=
TREX_COMPARE_MANIFEST ?= tmp/baselines/latest_trex_analysis_spec_manifest.json
TREX_COMPARE_OUT ?= tmp/trex_analysis_spec_compare_report.json
TREX_COMPARE_ARGS ?=
TREX_RUN_ARGS ?=
ROOT_SEARCH_DIR ?=
ROOT_CASES_OUT ?= tmp/apex2_root_cases.json
ROOT_CASES_ARGS ?=
ROOT_CASES_JSON ?=
ROOT_BASELINE_OUT_DIR ?= tmp/baselines
ROOT_BASELINE_ARGS ?=
ROOT_CONDOR_INITIALDIR ?= $(CURDIR)
ROOT_CONDOR_SUB_OUT ?= apex2_root_suite_array.sub
ROOT_RESULTS_DIR ?=
ROOT_AGG_GLOB ?= apex2_root_case_*.json
ROOT_AGG_OUT ?= tmp/apex2_root_suite_aggregate.json
ROOT_AGG_ARGS ?=
ROOT_BASELINE_SUITE ?=
ROOT_CURRENT_SUITE ?=
ROOT_PERF_OUT ?= tmp/root_suite_perf_compare.json
ROOT_PERF_ARGS ?=
ROOT_ROOT_MANIFEST ?= tmp/baselines/latest_root_manifest.json
PYHF_AUDIT_ARGS ?=
PYHF_AUDIT_FIT_ARGS ?=
PLAYGROUND_PORT ?= 8000
VALIDATION_PACK_OUT_DIR ?= tmp/validation_pack
VALIDATION_PACK_WORKSPACE ?= tests/fixtures/complex_workspace.json
VALIDATION_PACK_ARGS ?=

apex2-baseline-record:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/record_baseline.py $(RECORD_ARGS)

apex2-baseline-compare:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/compare_with_latest_baseline.py $(COMPARE_ARGS)

apex2-pre-release-gate:
	bash scripts/apex2/pre_release_gate.sh

validation-pack:
	bash validation-pack/render_validation_pack.sh \
		--out-dir "$(VALIDATION_PACK_OUT_DIR)" \
		--workspace "$(VALIDATION_PACK_WORKSPACE)" \
		$(VALIDATION_PACK_ARGS)

rust-slow-tests:
	cargo test -p ns-inference -- --ignored --skip test_fit_toys_pull_distribution

rust-very-slow-tests:
	cargo test -p ns-inference --release test_fit_toys_pull_distribution -- --ignored

trex-spec-validate:
	"$(PY)" scripts/trex/validate_analysis_spec.py --spec "$(TREX_SPEC)" --schema "$(TREX_SCHEMA)"

trex-spec-run:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" scripts/trex/run_analysis_spec.py --spec "$(TREX_SPEC)" $(TREX_RUN_ARGS)

trex-spec-baseline-record:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/record_trex_analysis_spec_baseline.py \
		--spec "$(TREX_SPEC)" \
		--schema "$(TREX_SCHEMA)" \
		--out-dir "$(TREX_RECORD_OUT_DIR)" \
		$(TREX_RECORD_ARGS)

trex-spec-baseline-compare:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/compare_trex_analysis_spec_with_latest_baseline.py \
		--manifest "$(TREX_COMPARE_MANIFEST)" \
		--schema "$(TREX_SCHEMA)" \
		--out "$(TREX_COMPARE_OUT)" \
		$(TREX_COMPARE_ARGS)

hepdata-fetch:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/hepdata/fetch_workspaces.py

hepdata-pytest:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" -m pytest -k hepdata_workspaces

hepdata-root-profile-smoke:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/hepdata/root_profile_smoke.py --fetch --keep

pyhf-audit:
	PYTHONUNBUFFERED=1 PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/audit_pyhf_parity.py $(PYHF_AUDIT_ARGS) --out-json tmp/pyhf_parity_audit.json --out-md tmp/pyhf_parity_audit.md

.PHONY: pyhf-audit-fit
pyhf-audit-fit:
	PYTHONUNBUFFERED=1 PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/audit_pyhf_parity.py $(PYHF_AUDIT_FIT_ARGS) --fit --fit-max-params 600 --out-json tmp/pyhf_parity_audit_fit.json --out-md tmp/pyhf_parity_audit_fit.md

apex2-root-prereq:
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/apex2_root_suite_report.py --prereq-only --out tmp/apex2_root_prereq.json

apex2-root-baseline-record:
	@test -n "$(ROOT_SEARCH_DIR)" || (echo "Set ROOT_SEARCH_DIR=/abs/path/to/trex/output" >&2; exit 2)
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/record_baseline.py \
		--only root \
		--out-dir "$(ROOT_BASELINE_OUT_DIR)" \
		--root-search-dir "$(ROOT_SEARCH_DIR)" \
		$(ROOT_BASELINE_ARGS)

apex2-root-cases:
	@test -n "$(ROOT_SEARCH_DIR)" || (echo "Set ROOT_SEARCH_DIR=/abs/path/to/trex/output" >&2; exit 2)
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/generate_apex2_root_cases.py \
		--search-dir "$(ROOT_SEARCH_DIR)" \
		--out "$(ROOT_CASES_OUT)" \
		$(ROOT_CASES_ARGS)

apex2-condor-render-root-array-sub:
	@test -n "$(ROOT_CASES_JSON)" || (echo "Set ROOT_CASES_JSON=/abs/path/to/apex2_root_cases.json" >&2; exit 2)
	"$(PY)" scripts/condor/render_apex2_root_suite_array_sub.py \
		--cases "$(ROOT_CASES_JSON)" \
		--initialdir "$(ROOT_CONDOR_INITIALDIR)" \
		--out "$(ROOT_CONDOR_SUB_OUT)"

apex2-root-aggregate:
	@test -n "$(ROOT_RESULTS_DIR)" || (echo "Set ROOT_RESULTS_DIR=/abs/path/to/results_dir" >&2; exit 2)
	PYTHONPATH="$(PYTHONPATH)" "$(PY)" tests/aggregate_apex2_root_suite_reports.py \
		--in-dir "$(ROOT_RESULTS_DIR)" \
		--glob "$(ROOT_AGG_GLOB)" \
		--out "$(ROOT_AGG_OUT)" \
		$(ROOT_AGG_ARGS)

apex2-root-suite-compare-perf:
	@test -n "$(ROOT_BASELINE_SUITE)" || (echo "Set ROOT_BASELINE_SUITE=/abs/path/to/root_suite_baseline.json" >&2; exit 2)
	@test -n "$(ROOT_CURRENT_SUITE)" || (echo "Set ROOT_CURRENT_SUITE=/abs/path/to/root_suite_current.json" >&2; exit 2)
	"$(PY)" tests/compare_apex2_root_suite_to_baseline.py \
		--baseline "$(ROOT_BASELINE_SUITE)" \
		--current "$(ROOT_CURRENT_SUITE)" \
		--out "$(ROOT_PERF_OUT)" \
		$(ROOT_PERF_ARGS)

apex2-root-suite-compare-latest:
	@test -n "$(ROOT_CURRENT_SUITE)" || (echo "Set ROOT_CURRENT_SUITE=/abs/path/to/apex2_root_suite_aggregate.json" >&2; exit 2)
	"$(PY)" tests/compare_apex2_root_suite_to_latest_baseline.py \
		--manifest "$(ROOT_ROOT_MANIFEST)" \
		--current "$(ROOT_CURRENT_SUITE)" \
		--out "$(ROOT_PERF_OUT)" \
		$(ROOT_PERF_ARGS)

playground-build-wasm:
	bash scripts/playground_build_wasm.sh

playground-serve:
	bash scripts/playground_serve.sh "$(PLAYGROUND_PORT)"
