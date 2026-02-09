from __future__ import annotations

import math


def test_missing_drop_rows_drops_any_row_with_missing_x_or_y():
    import nextstat.missing as missing

    X = [
        [1.0, 2.0],
        [None, 3.0],
        [4.0, float("nan")],
        [5.0, 6.0],
    ]
    y = [0, 1, None, 1]

    r = missing.apply_policy(X, y, policy="drop_rows")
    assert r.n_kept == 2
    assert r.n_dropped == 2
    assert r.kept_row_mask == [True, False, False, True]
    assert r.x == [[1.0, 2.0], [5.0, 6.0]]
    assert r.y == [0, 1]


def test_missing_impute_mean_imputes_x_and_drops_missing_y():
    import nextstat.missing as missing

    X = [
        [1.0, 2.0],
        [None, 3.0],
        [4.0, float("nan")],
        [5.0, 6.0],
    ]
    y = [0, 1, None, 1]

    r = missing.apply_policy(X, y, policy="impute_mean")

    # One row dropped due to missing y (index 2).
    assert r.kept_row_mask == [True, True, False, True]
    assert r.n_kept == 3
    assert r.y == [0, 1, 1]

    # Column means computed from observed (including rows that may be dropped due to y):
    # col0 observed: 1,4,5 => mean=10/3
    # col1 observed: 2,3,6 => mean=11/3
    m0 = 10.0 / 3.0
    m1 = 11.0 / 3.0

    got = r.x
    assert len(got) == 3
    assert got[0] == [1.0, 2.0]
    assert abs(got[1][0] - m0) < 1e-12 and got[1][1] == 3.0
    assert got[2] == [5.0, 6.0]
    assert all(math.isfinite(v) for row in got for v in row)


def test_missing_impute_mean_rejects_all_missing_column():
    import nextstat.missing as missing

    X = [
        [None, 1.0],
        [None, 2.0],
    ]
    y = [0, 1]

    try:
        missing.apply_policy(X, y, policy="impute_mean")
    except ValueError as e:
        assert "cannot impute column 0" in str(e)
    else:
        raise AssertionError("expected ValueError")
