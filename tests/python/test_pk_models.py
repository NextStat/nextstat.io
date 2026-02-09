import math

import nextstat


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def test_pk_model_predict_nll_grad_and_fit_smoke():
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    true_params = [1.2, 15.0, 2.0]  # (cl, v, ka)

    # Build a base model to generate deterministic synthetic observations.
    base = nextstat.OneCompartmentOralPkModel(
        times,
        [0.0] * len(times),
        dose=100.0,
        bioavailability=1.0,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )
    y = base.predict(true_params)
    assert len(y) == len(times)
    assert all(_is_finite(v) and v >= 0.0 for v in y)

    model = nextstat.OneCompartmentOralPkModel(
        times,
        y,
        dose=100.0,
        bioavailability=1.0,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )

    assert model.n_params() == 3
    assert len(model.parameter_names()) == 3
    assert len(model.suggested_init()) == 3
    assert len(model.suggested_bounds()) == 3

    nll = model.nll(true_params)
    g = model.grad_nll(true_params)
    assert _is_finite(nll)
    assert len(g) == 3
    assert all(_is_finite(v) for v in g)

    mle = nextstat.MaximumLikelihoodEstimator(max_iter=100, tol=1e-6, m=10)
    fit = mle.fit(model)
    assert len(fit.bestfit) == 3
    assert _is_finite(fit.nll)
    assert all(_is_finite(v) and v > 0.0 for v in fit.bestfit)


def test_nlme_model_constructs_and_has_finite_nll_grad():
    n_subjects = 2
    times_per = [0.25, 0.5, 1.0, 2.0, 4.0]

    base = nextstat.OneCompartmentOralPkModel(
        times_per,
        [0.0] * len(times_per),
        dose=100.0,
        bioavailability=1.0,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )

    cl_pop, v_pop, ka_pop = 1.2, 15.0, 2.0
    y_per = base.predict([cl_pop, v_pop, ka_pop])
    times = []
    y = []
    subject_idx = []
    for sid in range(n_subjects):
        for j, t in enumerate(times_per):
            times.append(t)
            y.append(y_per[j])
            subject_idx.append(sid)

    model = nextstat.OneCompartmentOralPkNlmeModel(
        times,
        y,
        subject_idx,
        n_subjects,
        dose=100.0,
        bioavailability=1.0,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )

    dim = model.n_params()
    assert dim == 6 + 3 * n_subjects

    init = model.suggested_init()
    assert len(init) == dim

    nll = model.nll(init)
    g = model.grad_nll(init)
    assert _is_finite(nll)
    assert len(g) == dim
    assert all(_is_finite(v) for v in g)
