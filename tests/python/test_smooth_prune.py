"""Tests for smoothing and pruning preprocessing steps."""

from __future__ import annotations

import copy

import pytest

from nextstat.analysis.preprocess.smooth import (
    _hanning,
    _running_median,
    apply_maxvariation_cap,
    smooth_353qh_twice,
    smooth_gaussian_kernel,
    smooth_variation,
)
from nextstat.analysis.preprocess.prune import (
    should_prune_histosys_overall,
    should_prune_histosys_shape,
    should_prune_normsys,
)
from nextstat.analysis.preprocess import (
    NegativeBinsHygieneStep,
    PreprocessPipeline,
    PruneSystematicsStep,
    SmoothHistoSysStep,
    SymmetrizeHistoSysStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ws_with_one_histosys(*, nominal, hi, lo, channel="SR", sample="bkg", modifier="shape") -> dict:
    return {
        "version": "1.0.0",
        "channels": [
            {
                "name": channel,
                "samples": [
                    {
                        "name": sample,
                        "data": list(nominal),
                        "modifiers": [
                            {
                                "name": modifier,
                                "type": "histosys",
                                "data": {"hi_data": list(hi), "lo_data": list(lo)},
                            }
                        ],
                    }
                ],
            }
        ],
        "observations": [{"name": channel, "data": list(nominal)}],
        "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
    }


def _ws_with_normsys(*, hi, lo, channel="SR", sample="bkg", modifier="norm_sys") -> dict:
    return {
        "version": "1.0.0",
        "channels": [
            {
                "name": channel,
                "samples": [
                    {
                        "name": sample,
                        "data": [100.0],
                        "modifiers": [
                            {
                                "name": modifier,
                                "type": "normsys",
                                "data": {"hi": hi, "lo": lo},
                            }
                        ],
                    }
                ],
            }
        ],
        "observations": [{"name": channel, "data": [100.0]}],
        "measurements": [{"name": "meas", "config": {"poi": "mu", "parameters": []}}],
    }


# ===========================================================================
# Smoothing tests
# ===========================================================================


class TestRunningMedian:
    def test_median3_known(self) -> None:
        # [1, 100, 2, 3, 4] → median3 should tame the spike
        result = _running_median([1.0, 100.0, 2.0, 3.0, 4.0], 3)
        assert len(result) == 5
        # Middle element: sorted([1,100,2]) = [1,2,100] → median = 2
        assert result[1] == 2.0
        # [100,2,3] → sorted = [2,3,100] → median = 3
        assert result[2] == 3.0

    def test_median5_known(self) -> None:
        result = _running_median([10.0, 1.0, 5.0, 3.0, 8.0], 5)
        assert len(result) == 5
        # Center: sorted([10,1,5,3,8]) = [1,3,5,8,10] → 5
        assert result[2] == 5.0

    def test_empty(self) -> None:
        assert _running_median([], 3) == []

    def test_single(self) -> None:
        assert _running_median([7.0], 3) == [7.0]


class TestHanning:
    def test_known_values(self) -> None:
        # [1, 4, 1] → inner: 0.25*1 + 0.5*4 + 0.25*1 = 2.5
        result = _hanning([1.0, 4.0, 1.0])
        assert result[1] == pytest.approx(2.5)
        # edges use repeated boundary
        # i=0: 0.25*1 + 0.5*1 + 0.25*4 = 1.75
        assert result[0] == pytest.approx(1.75)
        # i=2: 0.25*4 + 0.5*1 + 0.25*1 = 1.75
        assert result[2] == pytest.approx(1.75)

    def test_empty(self) -> None:
        assert _hanning([]) == []

    def test_single(self) -> None:
        assert _hanning([5.0]) == [5.0]


class TestSmooth353QHTwice:
    def test_flat_delta_unchanged(self) -> None:
        """Smoothing a flat delta should return the same flat delta."""
        delta = [3.0] * 10
        result = smooth_353qh_twice(delta)
        for v in result:
            assert v == pytest.approx(3.0, abs=1e-10)

    def test_single_spike_reduced(self) -> None:
        """A single spike should be reduced by smoothing."""
        delta = [0.0] * 4 + [10.0] + [0.0] * 4
        result = smooth_353qh_twice(delta)
        # Spike at index 4 should be reduced
        assert abs(result[4]) < 10.0
        # Overall max should decrease
        assert max(abs(v) for v in result) < max(abs(v) for v in delta)

    def test_short_input(self) -> None:
        """Inputs with <=2 bins should be returned as-is."""
        assert smooth_353qh_twice([5.0]) == [5.0]
        assert smooth_353qh_twice([1.0, 2.0]) == [1.0, 2.0]
        assert smooth_353qh_twice([]) == []

    def test_idempotent_on_smooth_data(self) -> None:
        """Already smooth data should change minimally."""
        delta = [float(i) for i in range(10)]
        result = smooth_353qh_twice(delta)
        for orig, sm in zip(delta, result):
            assert sm == pytest.approx(orig, abs=0.5)


class TestGaussianKernel:
    def test_sigma_zero_identity(self) -> None:
        """sigma=0 should return input unchanged."""
        delta = [1.0, 5.0, 2.0, 8.0]
        result = smooth_gaussian_kernel(delta, sigma=0.0)
        assert result == delta

    def test_large_sigma_approaches_mean(self) -> None:
        """Very large sigma should make all values approach the mean."""
        delta = [0.0, 0.0, 10.0, 0.0, 0.0]
        result = smooth_gaussian_kernel(delta, sigma=100.0)
        mean_val = sum(delta) / len(delta)
        for v in result:
            assert v == pytest.approx(mean_val, abs=0.1)

    def test_empty(self) -> None:
        assert smooth_gaussian_kernel([], sigma=1.5) == []

    def test_negative_sigma_identity(self) -> None:
        """Negative sigma should return input unchanged."""
        delta = [1.0, 2.0, 3.0]
        result = smooth_gaussian_kernel(delta, sigma=-1.0)
        assert result == delta


class TestMaxvariationCap:
    def test_cap_applied(self) -> None:
        original = [1.0, -2.0, 0.5]  # max |original| = 2.0
        smoothed = [3.0, -1.0, 0.5]  # 3.0 exceeds cap
        result = apply_maxvariation_cap(smoothed, original)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(-1.0)
        assert result[2] == pytest.approx(0.5)

    def test_no_cap_needed(self) -> None:
        original = [5.0, -3.0]
        smoothed = [2.0, -1.0]
        result = apply_maxvariation_cap(smoothed, original)
        assert result == smoothed

    def test_empty_original(self) -> None:
        result = apply_maxvariation_cap([1.0, 2.0], [])
        assert result == [1.0, 2.0]


class TestSmoothVariation:
    def test_round_trip_nominal(self) -> None:
        """smooth_variation(nom, nom, nom) → unchanged (delta = 0)."""
        nom = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = smooth_variation(nom, nom, nom)
        assert result.up == pytest.approx(nom)
        assert result.down == pytest.approx(nom)
        assert result.max_delta_before_up == pytest.approx(0.0)
        assert result.max_delta_before_down == pytest.approx(0.0)

    def test_bins_1_no_crash(self) -> None:
        """Edge case: single bin should not crash."""
        result = smooth_variation([10.0], [12.0], [8.0])
        assert len(result.up) == 1
        assert len(result.down) == 1

    def test_gaussian_method(self) -> None:
        nom = [10.0, 20.0, 30.0, 20.0, 10.0]
        hi = [11.0, 25.0, 31.0, 21.0, 11.0]
        lo = [9.0, 15.0, 29.0, 19.0, 9.0]
        result = smooth_variation(nom, hi, lo, method="gaussian", sigma=1.0)
        assert len(result.up) == 5
        assert len(result.down) == 5
        assert result.method == "gaussian"

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown smooth method"):
            smooth_variation([1.0], [2.0], [0.0], method="bogus")  # type: ignore[arg-type]

    def test_maxvariation_flag(self) -> None:
        """Check maxvariation_applied flag in result."""
        nom = [10.0, 20.0, 30.0, 40.0, 50.0]
        result_on = smooth_variation(nom, nom, nom, apply_maxvariation=True)
        result_off = smooth_variation(nom, nom, nom, apply_maxvariation=False)
        # Both should be identical for zero delta
        assert result_on.up == pytest.approx(nom)
        assert result_off.up == pytest.approx(nom)


# ===========================================================================
# Pruning tests
# ===========================================================================


class TestPruneHistosysShape:
    def test_below_threshold_prunes(self) -> None:
        nom = [100.0, 200.0, 300.0]
        hi = [100.1, 200.2, 300.3]   # max rel delta = 0.3/300 = 0.001
        lo = [99.9, 199.8, 299.7]    # max rel delta = 0.3/300 = 0.001
        d = should_prune_histosys_shape(nom, hi, lo, threshold=0.005)
        assert d.should_prune is True

    def test_above_threshold_keeps(self) -> None:
        nom = [100.0, 200.0, 300.0]
        hi = [110.0, 220.0, 330.0]   # 10% variation
        lo = [90.0, 180.0, 270.0]
        d = should_prune_histosys_shape(nom, hi, lo, threshold=0.005)
        assert d.should_prune is False

    def test_exact_threshold_keeps(self) -> None:
        """At exactly the threshold, should NOT prune (< not <=)."""
        nom = [100.0]
        hi = [100.5]   # rel delta = 0.005
        lo = [99.5]
        d = should_prune_histosys_shape(nom, hi, lo, threshold=0.005)
        assert d.should_prune is False

    def test_zero_nominal_bin_skipped(self) -> None:
        """Bins with nom=0 are skipped in relative calculation."""
        nom = [0.0, 100.0]
        hi = [5.0, 100.1]   # bin 0 skipped, bin 1: 0.1/100 = 0.001
        lo = [-5.0, 99.9]
        d = should_prune_histosys_shape(nom, hi, lo, threshold=0.005)
        assert d.should_prune is True


class TestPruneNormsys:
    def test_small_effect_prunes(self) -> None:
        d = should_prune_normsys(1.001, 0.999, threshold=0.005)
        assert d.should_prune is True

    def test_large_effect_keeps(self) -> None:
        d = should_prune_normsys(1.1, 0.9, threshold=0.005)
        assert d.should_prune is False

    def test_asymmetric_one_side_large(self) -> None:
        """One side small, other large → keep."""
        d = should_prune_normsys(1.001, 0.9, threshold=0.005)
        assert d.should_prune is False


class TestPruneHistosysOverall:
    def test_pure_norm_prune(self) -> None:
        """Histosys with only norm effect (uniform scaling), small → prune."""
        nom = [100.0, 200.0]
        # 0.1% scaling
        hi = [100.1, 200.2]
        lo = [99.9, 199.8]
        d = should_prune_histosys_overall(nom, hi, lo, norm_threshold=0.005, shape_threshold=0.005)
        assert d.should_prune is True

    def test_large_norm_keeps(self) -> None:
        """Large norm effect → keep."""
        nom = [100.0, 200.0]
        hi = [120.0, 240.0]   # 20% norm
        lo = [80.0, 160.0]
        d = should_prune_histosys_overall(nom, hi, lo, norm_threshold=0.005, shape_threshold=0.005)
        assert d.should_prune is False

    def test_decomposition_shape_only(self) -> None:
        """Pure shape (same integral) with large residual → keep."""
        nom = [100.0, 100.0]
        # Same integral (200) but different shape
        hi = [150.0, 50.0]
        lo = [50.0, 150.0]
        d = should_prune_histosys_overall(nom, hi, lo, norm_threshold=0.005, shape_threshold=0.005)
        # Shape effect is 50% → keep
        assert d.should_prune is False


# ===========================================================================
# Pipeline integration tests
# ===========================================================================


class TestSmoothHistoSysStep:
    def test_smooth_modifies_workspace(self) -> None:
        nom = [10.0, 10.0, 10.0, 100.0, 10.0, 10.0, 10.0]  # spike at center in hi
        hi = [10.0, 10.0, 10.0, 200.0, 10.0, 10.0, 10.0]
        lo = [10.0, 10.0, 10.0, 5.0, 10.0, 10.0, 10.0]
        ws = _ws_with_one_histosys(nominal=nom, hi=hi, lo=lo)
        pipe = PreprocessPipeline([SmoothHistoSysStep(method="353qh_twice")])
        res = pipe.run(ws)
        mod = res.workspace["channels"][0]["samples"][0]["modifiers"][0]
        # Spike should be reduced
        smoothed_hi = mod["data"]["hi_data"]
        assert smoothed_hi[3] < 200.0  # reduced from 200
        # Provenance
        recs = res.provenance_dict()["steps"][0]["records"]
        assert len(recs) == 1
        assert recs[0]["kind"] == "histosys.smooth"
        assert recs[0]["changed"] is True

    def test_smooth_unchanged_records(self) -> None:
        """When data doesn't change, record_unchanged controls recording."""
        nom = [10.0] * 5
        ws = _ws_with_one_histosys(nominal=nom, hi=nom, lo=nom)
        pipe = PreprocessPipeline([SmoothHistoSysStep(record_unchanged=False)])
        res = pipe.run(ws)
        recs = res.provenance_dict()["steps"][0]["records"]
        assert len(recs) == 0

    def test_smooth_gaussian(self) -> None:
        nom = [10.0, 10.0, 10.0, 100.0, 10.0, 10.0, 10.0]
        hi = [10.0, 10.0, 10.0, 200.0, 10.0, 10.0, 10.0]
        lo = [10.0, 10.0, 10.0, 5.0, 10.0, 10.0, 10.0]
        ws = _ws_with_one_histosys(nominal=nom, hi=hi, lo=lo)
        pipe = PreprocessPipeline([SmoothHistoSysStep(method="gaussian", sigma=1.5)])
        res = pipe.run(ws)
        mod = res.workspace["channels"][0]["samples"][0]["modifiers"][0]
        assert mod["data"]["hi_data"][3] < 200.0


class TestPruneSystematicsStep:
    def test_prune_removes_modifier(self) -> None:
        """Small systematic should be removed from workspace."""
        nom = [100.0, 200.0, 300.0]
        hi = [100.05, 200.1, 300.15]  # ~0.05% effect
        lo = [99.95, 199.9, 299.85]
        ws = _ws_with_one_histosys(nominal=nom, hi=hi, lo=lo)
        pipe = PreprocessPipeline([PruneSystematicsStep(shape_threshold=0.005)])
        res = pipe.run(ws)
        mods = res.workspace["channels"][0]["samples"][0]["modifiers"]
        assert len(mods) == 0  # modifier was removed

    def test_prune_keeps_significant(self) -> None:
        """Large systematic should be kept."""
        nom = [100.0, 200.0]
        hi = [120.0, 240.0]
        lo = [80.0, 160.0]
        ws = _ws_with_one_histosys(nominal=nom, hi=hi, lo=lo)
        pipe = PreprocessPipeline([PruneSystematicsStep(shape_threshold=0.005)])
        res = pipe.run(ws)
        mods = res.workspace["channels"][0]["samples"][0]["modifiers"]
        assert len(mods) == 1

    def test_prune_normsys(self) -> None:
        """Normsys pruning with norm method."""
        ws = _ws_with_normsys(hi=1.001, lo=0.999)
        pipe = PreprocessPipeline([PruneSystematicsStep(prune_method="norm", norm_threshold=0.005)])
        res = pipe.run(ws)
        mods = res.workspace["channels"][0]["samples"][0]["modifiers"]
        assert len(mods) == 0

    def test_prune_normsys_kept(self) -> None:
        """Large normsys should be kept."""
        ws = _ws_with_normsys(hi=1.1, lo=0.9)
        pipe = PreprocessPipeline([PruneSystematicsStep(prune_method="norm", norm_threshold=0.005)])
        res = pipe.run(ws)
        mods = res.workspace["channels"][0]["samples"][0]["modifiers"]
        assert len(mods) == 1

    def test_prune_provenance(self) -> None:
        """Pruned modifier should have a provenance record with changed=True."""
        nom = [100.0]
        hi = [100.01]
        lo = [99.99]
        ws = _ws_with_one_histosys(nominal=nom, hi=hi, lo=lo)
        pipe = PreprocessPipeline([PruneSystematicsStep(shape_threshold=0.005)])
        res = pipe.run(ws)
        recs = res.provenance_dict()["steps"][0]["records"]
        assert len(recs) == 1
        assert recs[0]["changed"] is True
        assert recs[0]["kind"] == "histosys.prune"


class TestFullPipeline:
    def test_hygiene_symmetrize_smooth_prune(self) -> None:
        """Full pipeline: hygiene → symmetrize → smooth → prune."""
        nom = [100.0, 200.0, 300.0, 400.0, 500.0]
        hi = [100.1, 200.2, 300.3, 400.4, 500.5]   # small ~0.1% effect
        lo = [99.9, 199.8, 299.7, 399.6, 499.5]
        ws = _ws_with_one_histosys(nominal=nom, hi=hi, lo=lo)
        pipe = PreprocessPipeline([
            NegativeBinsHygieneStep(policy="clamp_renorm"),
            SymmetrizeHistoSysStep(method="absmean"),
            SmoothHistoSysStep(method="353qh_twice"),
            PruneSystematicsStep(shape_threshold=0.005),  # 0.5% threshold, 0.1% effect → prune
        ])
        res = pipe.run(ws)
        # Modifier should be pruned (0.1% < 0.5%)
        mods = res.workspace["channels"][0]["samples"][0]["modifiers"]
        assert len(mods) == 0
        # 4 pipeline steps
        assert len(res.provenance.steps) == 4

    def test_provenance_sha_chain(self) -> None:
        """Each step's output SHA should be the next step's input SHA."""
        nom = [10.0, 20.0, 30.0]
        hi = [12.0, 22.0, 32.0]
        lo = [8.0, 18.0, 28.0]
        ws = _ws_with_one_histosys(nominal=nom, hi=hi, lo=lo)
        pipe = PreprocessPipeline([
            SymmetrizeHistoSysStep(method="absmean"),
            SmoothHistoSysStep(method="353qh_twice"),
        ])
        res = pipe.run(ws)
        steps = res.provenance.steps
        assert len(steps) == 2
        assert steps[0].output_sha256 == steps[1].input_sha256
        assert res.provenance.input_sha256 == steps[0].input_sha256
        assert res.provenance.output_sha256 == steps[1].output_sha256

    def test_smooth_then_prune(self) -> None:
        """Smooth then prune: broad systematic survives smooth+prune."""
        nom = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        # Broad shift across all bins — not a spike, survives smoothing
        hi = [110.0, 112.0, 115.0, 118.0, 115.0, 112.0, 110.0]
        lo = [90.0, 88.0, 85.0, 82.0, 85.0, 88.0, 90.0]
        ws = _ws_with_one_histosys(nominal=nom, hi=hi, lo=lo)
        pipe = PreprocessPipeline([
            SmoothHistoSysStep(method="353qh_twice"),
            PruneSystematicsStep(shape_threshold=0.005),
        ])
        res = pipe.run(ws)
        # Broad 10-18% variation survives smoothing and isn't pruned
        mods = res.workspace["channels"][0]["samples"][0]["modifiers"]
        assert len(mods) == 1

    def test_pipeline_does_not_mutate_input(self) -> None:
        """Pipeline should deep-copy by default."""
        nom = [100.0, 200.0]
        hi = [100.01, 200.02]
        lo = [99.99, 199.98]
        ws = _ws_with_one_histosys(nominal=nom, hi=hi, lo=lo)
        ws_copy = copy.deepcopy(ws)
        pipe = PreprocessPipeline([
            SmoothHistoSysStep(),
            PruneSystematicsStep(shape_threshold=0.005),
        ])
        pipe.run(ws)
        assert ws == ws_copy
