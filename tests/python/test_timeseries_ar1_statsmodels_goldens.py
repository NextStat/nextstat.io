"""AR(1) Kalman goldens (statsmodels-derived).

This is a non-optional regression harness:
- The fixture was generated from statsmodels once and checked into the repo.
- The test compares NextStat outputs against the fixture without importing statsmodels.

Regenerate:
  PYTHONPATH=bindings/ns-py/python ./.venv/bin/python \
    tests/python/generate_statsmodels_ar1_kalman_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import nextstat


def _flatten_1d_means(xs) -> list[float]:
    return [float(v[0]) for v in xs]


def _flatten_1x1_covs(ps) -> list[float]:
    return [float(v[0][0]) for v in ps]


def test_ar1_kalman_matches_statsmodels_goldens():
    path = Path("tests/fixtures/statsmodels_ar1_kalman_goldens.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert int(raw.get("version")) == 1
    cases = raw.get("cases")
    assert isinstance(cases, list) and cases

    for c in cases:
        name = str(c["name"])
        phi = float(c["phi"])
        q = float(c["q"])
        r = float(c["r"])
        m0 = float(c["m0"])
        p0 = float(c["p0"])
        ys = [float(v) for v in c["ys"]]
        ref = c["ref"]

        model = nextstat.timeseries.ar1_model(phi=phi, q=q, r=r, m0=m0, p0=p0)

        ns_f = nextstat.timeseries.kalman_filter(model, [[y] for y in ys])
        assert float(ns_f["log_likelihood"]) == pytest.approx(
            float(ref["log_likelihood"]), rel=1e-6, abs=1e-6
        ), name
        assert _flatten_1d_means(ns_f["filtered_means"]) == pytest.approx(
            _flatten_1d_means(ref["filtered_means"]), rel=0.0, abs=1e-6
        ), name
        assert _flatten_1x1_covs(ns_f["filtered_covs"]) == pytest.approx(
            _flatten_1x1_covs(ref["filtered_covs"]), rel=0.0, abs=1e-6
        ), name

        if ref.get("predicted_means") is not None and ref.get("predicted_covs") is not None:
            # Compare only the overlapping prefix (statsmodels may expose n or n+1).
            n_pred = min(len(ns_f["predicted_means"]), len(ref["predicted_means"]))
            assert _flatten_1d_means(ns_f["predicted_means"][:n_pred]) == pytest.approx(
                _flatten_1d_means(ref["predicted_means"][:n_pred]), rel=0.0, abs=1e-6
            ), name
            assert _flatten_1x1_covs(ns_f["predicted_covs"][:n_pred]) == pytest.approx(
                _flatten_1x1_covs(ref["predicted_covs"][:n_pred]), rel=0.0, abs=1e-6
            ), name

        ns_s = nextstat.timeseries.kalman_smooth(model, [[y] for y in ys])
        assert _flatten_1d_means(ns_s["smoothed_means"]) == pytest.approx(
            _flatten_1d_means(ref["smoothed_means"]), rel=0.0, abs=1e-6
        ), name
        assert _flatten_1x1_covs(ns_s["smoothed_covs"]) == pytest.approx(
            _flatten_1x1_covs(ref["smoothed_covs"]), rel=0.0, abs=1e-6
        ), name

