from __future__ import annotations


def test_logistic_separation_emits_warning_unregularized():
    import nextstat

    # Perfectly separable toy.
    x = [[-3.0], [-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0], [3.0]]
    y = [0, 0, 0, 0, 1, 1, 1, 1]

    r = nextstat.glm.logistic.fit(x, y, include_intercept=True)
    assert isinstance(r.warnings, list)
    # MLE may still "converge" numerically depending on the optimizer; warn heuristically.
    assert (
        ("possible_separation_use_l2" in r.warnings)
        or ("large_coefficients_possible_separation" in r.warnings)
        or (not r.converged)
    )


def test_logistic_separation_regularized_has_no_separation_warning():
    import nextstat

    x = [[-3.0], [-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0], [3.0]]
    y = [0, 0, 0, 0, 1, 1, 1, 1]

    r = nextstat.glm.logistic.fit(x, y, include_intercept=True, l2=10.0, penalize_intercept=False)
    assert "possible_separation_use_l2" not in r.warnings


def test_logistic_separation_can_retry_with_fallback_l2():
    import nextstat

    x = [[-3.0], [-2.0], [-1.5], [-1.0], [1.0], [1.5], [2.0], [3.0]]
    y = [0, 0, 0, 0, 1, 1, 1, 1]

    r = nextstat.glm.logistic.fit(
        x,
        y,
        include_intercept=True,
        fallback_l2=10.0,
        fallback_on_separation=True,
    )
    assert r.converged
    assert "fallback_l2_applied" in r.warnings
    assert "possible_separation_use_l2" not in r.warnings
