from __future__ import annotations

from pathlib import Path

import nextstat.unbinned as ns_unbinned


class _FakeModel:
    def __init__(self) -> None:
        self._names = ["mu", "theta"]
        self._init = [1.0, 0.0]
        self._bounds = [(0.0, 5.0), (-5.0, 5.0)]
        self._poi_index = 0
        self._schema_version = "nextstat_unbinned_spec_v0"
        self.with_fixed_calls: list[tuple[int, float]] = []

    def parameter_names(self) -> list[str]:
        return list(self._names)

    def suggested_init(self) -> list[float]:
        return list(self._init)

    def suggested_bounds(self) -> list[tuple[float, float]]:
        return list(self._bounds)

    def poi_index(self) -> int | None:
        return self._poi_index

    def n_params(self) -> int:
        return len(self._names)

    def schema_version(self) -> str:
        return self._schema_version

    def with_fixed_param(self, idx: int, value: float) -> _FakeModel:
        self.with_fixed_calls.append((idx, value))
        fixed = _FakeModel()
        fixed._poi_index = self._poi_index
        return fixed


def test_from_config_builds_analysis(monkeypatch) -> None:
    called: dict[str, str] = {}

    class _FakeUnbinnedClass:
        @staticmethod
        def from_config(path: str):
            called["path"] = path
            return _FakeModel()

    monkeypatch.setattr(ns_unbinned, "UnbinnedModel", _FakeUnbinnedClass)

    config_path = Path("tmp/model.yaml")
    analysis = ns_unbinned.from_config(config_path)
    assert isinstance(analysis, ns_unbinned.UnbinnedAnalysis)
    assert called["path"] == str(config_path)


def test_analysis_methods_delegate_to_core(monkeypatch) -> None:
    model = _FakeModel()
    analysis = ns_unbinned.UnbinnedAnalysis(model)
    calls: dict[str, tuple] = {}

    def _fake_fit(m, *, init_pars=None):
        calls["fit"] = (m, init_pars)
        return {"kind": "fit"}

    def _fake_scan(m, mu_values):
        calls["scan"] = (m, list(mu_values))
        return {"kind": "scan"}

    def _fake_fit_toys(m, params, **kwargs):
        calls["fit_toys"] = (m, list(params), kwargs)
        return [{"kind": "fit_toys"}]

    def _fake_hypotest(mu_test, m):
        calls["hypotest"] = (mu_test, m)
        return {"kind": "hypotest"}

    def _fake_hypotest_toys(poi_test, m, **kwargs):
        calls["hypotest_toys"] = (poi_test, m, kwargs)
        return {"kind": "hypotest_toys"}

    def _fake_ranking(m):
        calls["ranking"] = (m,)
        return [{"name": "theta"}]

    monkeypatch.setattr(ns_unbinned, "_fit", _fake_fit)
    monkeypatch.setattr(ns_unbinned, "_profile_scan", _fake_scan)
    monkeypatch.setattr(ns_unbinned, "_fit_toys", _fake_fit_toys)
    monkeypatch.setattr(ns_unbinned, "_hypotest", _fake_hypotest)
    monkeypatch.setattr(ns_unbinned, "_hypotest_toys", _fake_hypotest_toys)
    monkeypatch.setattr(ns_unbinned, "_ranking", _fake_ranking)

    assert analysis.fit(init_pars=(1.0, -0.2)) == {"kind": "fit"}
    assert analysis.fit_toys(params=[1.1, 0.0], n_toys=25, seed=7) == [{"kind": "fit_toys"}]
    assert analysis.scan([0.0, 1.0, 2.0]) == {"kind": "scan"}
    assert analysis.hypotest(1.0) == {"kind": "hypotest"}
    assert analysis.hypotest_toys(1.0, n_toys=25, seed=7, expected_set=True) == {
        "kind": "hypotest_toys"
    }
    assert analysis.ranking() == [{"name": "theta"}]

    assert calls["fit"] == (model, [1.0, -0.2])
    assert calls["fit_toys"] == (model, [1.1, 0.0], {"n_toys": 25, "seed": 7})
    assert calls["scan"] == (model, [0.0, 1.0, 2.0])
    assert calls["hypotest"] == (1.0, model)
    assert calls["hypotest_toys"][0] == 1.0
    assert calls["hypotest_toys"][1] is model
    assert calls["hypotest_toys"][2]["n_toys"] == 25
    assert calls["hypotest_toys"][2]["seed"] == 7
    assert calls["hypotest_toys"][2]["expected_set"] is True
    assert calls["ranking"] == (model,)


def test_fit_toys_defaults_to_suggested_init(monkeypatch) -> None:
    model = _FakeModel()
    analysis = ns_unbinned.UnbinnedAnalysis(model)
    captured: dict[str, object] = {}

    def _fake_fit_toys(m, params, **kwargs):
        captured["model"] = m
        captured["params"] = list(params)
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(ns_unbinned, "_fit_toys", _fake_fit_toys)

    assert analysis.fit_toys(n_toys=3, seed=11) == []
    assert captured["model"] is model
    assert captured["params"] == [1.0, 0.0]
    assert captured["kwargs"] == {"n_toys": 3, "seed": 11}


def test_parameter_index_and_with_fixed_param() -> None:
    model = _FakeModel()
    analysis = ns_unbinned.UnbinnedAnalysis(model)

    assert analysis.parameter_index("mu") == 0
    assert analysis.parameter_index(1) == 1

    fixed = analysis.with_fixed_param("theta", 0.5)
    assert isinstance(fixed, ns_unbinned.UnbinnedAnalysis)
    assert model.with_fixed_calls == [(1, 0.5)]


def test_summary_shape() -> None:
    model = _FakeModel()
    analysis = ns_unbinned.UnbinnedAnalysis(model)

    summary = analysis.summary()
    assert summary["schema_version"] == "nextstat_unbinned_spec_v0"
    assert summary["n_params"] == 2
    assert summary["poi_index"] == 0
    assert summary["parameters"][0]["name"] == "mu"
    assert summary["parameters"][0]["is_poi"] is True
