from __future__ import annotations

import pytest

from nextstat.analysis.preprocess.symmetrize import symmetrize_shapes


def test_symmetrize_absmean_golden():
    nominal = [10.0, 20.0]
    up = [12.0, 18.0]
    down = [9.0, 25.0]

    r = symmetrize_shapes(nominal, up=up, down=down, method="absmean")
    assert r.up == pytest.approx([11.5, 23.5], abs=1e-12)
    assert r.down == pytest.approx([8.5, 16.5], abs=1e-12)


def test_symmetrize_max_golden():
    nominal = [10.0, 20.0]
    up = [12.0, 18.0]
    down = [9.0, 25.0]

    r = symmetrize_shapes(nominal, up=up, down=down, method="max")
    assert r.up == pytest.approx([12.0, 25.0], abs=1e-12)
    assert r.down == pytest.approx([8.0, 15.0], abs=1e-12)


def test_symmetrize_twosided_golden():
    nominal = [10.0, 20.0]
    up = [12.0, 18.0]
    down = [9.0, 25.0]

    r = symmetrize_shapes(nominal, up=up, down=down, method="twosided")
    assert r.up == pytest.approx([11.5, 16.5], abs=1e-12)
    assert r.down == pytest.approx([8.5, 23.5], abs=1e-12)


def test_symmetrize_onesided_reflects_missing_side():
    nominal = [10.0, 20.0]
    up = [12.0, 18.0]

    r = symmetrize_shapes(nominal, up=up, down=None, method="onesided")
    assert r.up == pytest.approx([12.0, 18.0], abs=1e-12)
    assert r.down == pytest.approx([8.0, 22.0], abs=1e-12)


def test_symmetrize_negative_policy_error_raises():
    nominal = [1.0]
    up = [-1.0]

    with pytest.raises(ValueError, match="negative bin"):
        symmetrize_shapes(nominal, up=up, down=None, method="onesided", negative_policy="error")


def test_symmetrize_negative_policy_clamp():
    nominal = [1.0]
    up = [-1.0]

    r = symmetrize_shapes(nominal, up=up, down=None, method="onesided", negative_policy="clamp")
    assert r.up == pytest.approx([0.0], abs=1e-12)
    assert r.down == pytest.approx([3.0], abs=1e-12)


def test_symmetrize_requires_both_sides_for_twosided():
    nominal = [1.0, 2.0]
    up = [1.1, 1.9]
    with pytest.raises(ValueError, match="requires both up and down"):
        symmetrize_shapes(nominal, up=up, down=None, method="twosided")

