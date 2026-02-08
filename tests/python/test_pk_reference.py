import math

import nextstat


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def _oral_1c_analytic(times: list[float], *, cl: float, v: float, ka: float, dose: float, f: float) -> list[float]:
    # One-compartment oral dosing with first-order absorption and elimination.
    # C(t) = (F*D*ka/(V*(ka-ke))) * (exp(-ke t) - exp(-ka t))
    #
    # For ka ~ ke, use the limit:
    # C(t) = (F*D/V) * ka * t * exp(-ka t)
    ke = float(cl) / float(v)
    out: list[float] = []
    for t in times:
        t = float(t)
        if t < 0:
            raise AssertionError("time must be >= 0")
        if abs(float(ka) - ke) < 1e-10:
            c = (float(f) * float(dose) / float(v)) * float(ka) * t * math.exp(-float(ka) * t)
        else:
            pref = (float(f) * float(dose) * float(ka)) / (float(v) * (float(ka) - ke))
            c = pref * (math.exp(-ke * t) - math.exp(-float(ka) * t))
        out.append(float(c))
    return out


def test_one_compartment_oral_predict_matches_analytic_reference():
    times = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    cl, v, ka = 1.2, 15.0, 2.0
    dose, f = 100.0, 1.0

    analytic = _oral_1c_analytic(times, cl=cl, v=v, ka=ka, dose=dose, f=f)
    assert len(analytic) == len(times)
    assert all(_is_finite(x) and x >= 0.0 for x in analytic)

    model = nextstat.OneCompartmentOralPkModel(
        times,
        [0.0] * len(times),
        dose=dose,
        bioavailability=f,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )
    pred = model.predict([cl, v, ka])
    assert len(pred) == len(times)
    assert all(_is_finite(x) and x >= 0.0 for x in pred)

    # Tight absolute tolerance: this is a reference formula check.
    for a, b in zip(analytic, pred):
        assert abs(float(a) - float(b)) <= 1e-10


def test_one_compartment_oral_predict_stable_near_ka_equals_ke():
    # Stress the (ka - ke) denominator numerics.
    times = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    cl, v = 1.0, 10.0
    ke = cl / v
    ka = ke + 1e-12
    dose, f = 50.0, 0.9

    model = nextstat.OneCompartmentOralPkModel(
        times,
        [0.0] * len(times),
        dose=dose,
        bioavailability=f,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )
    pred = model.predict([cl, v, ka])
    assert all(_is_finite(x) and x >= 0.0 for x in pred)


def test_one_compartment_oral_grad_nll_near_zero_on_self_generated_data():
    times = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    true_params = [1.2, 15.0, 2.0]  # (cl, v, ka)

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

    model = nextstat.OneCompartmentOralPkModel(
        times,
        y,
        dose=100.0,
        bioavailability=1.0,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )
    g = model.grad_nll(true_params)
    assert len(g) == 3
    assert all(_is_finite(v) for v in g)
    # Not exact zero due to implementation details; keep a pragmatic threshold.
    assert max(abs(float(v)) for v in g) <= 1e-6

