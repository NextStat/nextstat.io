import math

import nextstat


def _is_finite(x: float) -> bool:
    return math.isfinite(float(x))


def _subject_params(theta: list[float], subject: int, n_subjects: int) -> list[float]:
    # Deterministic mild between-subject variability so omega estimation is well-posed.
    centered = subject - 0.5 * (n_subjects - 1)
    out: list[float] = []
    for p, val in enumerate(theta):
        eta = 0.04 * centered * (p + 1) / len(theta)
        out.append(float(val) * math.exp(eta))
    return out


def _simulate_population(
    model: str,
    theta: list[float],
    n_subjects: int = 4,
    dose: float = 100.0,
    bioavailability: float = 1.0,
) -> tuple[list[float], list[float], list[int], list[float]]:
    times_per = [0.5, 1.0, 2.0, 4.0, 8.0]

    if model == "1cpt_iv":
        base = None
    elif model == "2cpt_iv":
        base = nextstat.TwoCompartmentIvPkModel(
            times_per,
            [0.0] * len(times_per),
            dose=dose,
            error_model="additive",
            sigma=0.05,
        )
    elif model == "2cpt_oral":
        base = nextstat.TwoCompartmentOralPkModel(
            times_per,
            [0.0] * len(times_per),
            dose=dose,
            bioavailability=bioavailability,
            error_model="additive",
            sigma=0.05,
        )
    elif model == "3cpt_iv":
        base = nextstat.ThreeCompartmentIvPkModel(
            times_per,
            [0.0] * len(times_per),
            dose=dose,
            error_model="additive",
            sigma=0.05,
        )
    elif model == "3cpt_oral":
        base = nextstat.ThreeCompartmentOralPkModel(
            times_per,
            [0.0] * len(times_per),
            dose=dose,
            bioavailability=bioavailability,
            error_model="additive",
            sigma=0.05,
        )
    else:
        raise AssertionError(f"unsupported model: {model}")

    times: list[float] = []
    y: list[float] = []
    subject_idx: list[int] = []
    for sid in range(n_subjects):
        if model == "1cpt_iv":
            cl, v = _subject_params(theta, sid, n_subjects)
            pred = [(dose / v) * math.exp(-(cl / v) * t) for t in times_per]
        else:
            pred = base.predict(_subject_params(theta, sid, n_subjects))
        for j, t in enumerate(times_per):
            # Deterministic observation perturbation to avoid a perfect fit corner case.
            obs = max(float(pred[j]) * (1.0 + 0.015 * ((sid + j) % 3 - 1)), 1e-8)
            times.append(float(t))
            y.append(obs)
            subject_idx.append(sid)
    return times, y, subject_idx, [dose]


def _assert_common_result_shape(res: dict, n_subjects: int, n_theta: int) -> None:
    assert isinstance(res, dict)
    assert {"theta", "omega", "eta", "ofv", "converged", "n_iter"} <= set(res.keys())
    assert len(res["theta"]) == n_theta
    assert len(res["omega"]) == n_theta
    assert len(res["eta"]) == n_subjects
    assert len(res["eta"][0]) == n_theta
    assert _is_finite(res["ofv"])
    assert isinstance(res["converged"], bool)
    assert isinstance(res["n_iter"], int)


def test_nlme_foce_multicpt_new_methods_dispatch_smoke():
    cases = [
        ("2cpt_iv", "fo", [1.4, 18.0, 1.0, 24.0]),
        ("2cpt_oral", "its", [1.4, 18.0, 1.0, 24.0, 1.8]),
        ("3cpt_iv", "imp", [1.2, 16.0, 0.9, 22.0, 0.6, 35.0]),
        ("3cpt_oral", "fo", [1.2, 16.0, 0.9, 22.0, 0.6, 35.0, 1.5]),
    ]

    for model, method, theta_init in cases:
        n_subjects = 4
        times, y, subject_idx, doses = _simulate_population(
            model=model,
            theta=theta_init,
            n_subjects=n_subjects,
            dose=120.0,
            bioavailability=1.0,
        )

        kwargs = dict(
            model=model,
            method=method,
            doses=doses,
            bioavailability=1.0,
            error_model="additive",
            sigma=0.08,
            theta_init=theta_init,
            omega_init=[0.25] * len(theta_init),
            max_outer_iter=10,
            max_inner_iter=10,
            tol=1e-4,
            its_max_iter=6,
            its_max_individual_iter=20,
            its_tol=1e-4,
            its_omega_damping=0.3,
            imp_n_iter=4,
            imp_n_samples=80,
            imp_proposal_scale=1.0,
            imp_seed=42,
            imp_tol=1e-4,
            imp_e_only=False,
        )
        if model == "2cpt_iv":
            # Also exercise correlated omega path in Python binding.
            kwargs["omega_matrix"] = [
                [0.25**2, 0.0, 0.0, 0.0],
                [0.0, 0.25**2, 0.0, 0.0],
                [0.0, 0.0, 0.25**2, 0.0],
                [0.0, 0.0, 0.0, 0.25**2],
            ]

        res = nextstat.nlme_foce(times, y, subject_idx, n_subjects, **kwargs)
        _assert_common_result_shape(res, n_subjects=n_subjects, n_theta=len(theta_init))

        if method == "imp":
            imp = res.get("imp")
            assert isinstance(imp, dict)
            assert len(imp["ofv_trace"]) >= 1
            assert len(imp["ess_fraction_trace"]) >= 1
            assert len(imp["max_weight_trace"]) >= 1
        else:
            assert res.get("imp") is None


def test_nlme_foce_1cpt_iv_methods_dispatch_smoke():
    n_subjects = 4
    theta_init = [1.3, 18.0]
    times, y, subject_idx, doses = _simulate_population(
        model="1cpt_iv",
        theta=theta_init,
        n_subjects=n_subjects,
        dose=120.0,
    )

    for method in ["fo", "its", "imp"]:
        res = nextstat.nlme_foce(
            times,
            y,
            subject_idx,
            n_subjects,
            model="1cpt_iv",
            method=method,
            doses=doses,
            error_model="additive",
            sigma=0.08,
            theta_init=theta_init,
            omega_init=[0.25, 0.25],
            max_outer_iter=10,
            max_inner_iter=10,
            tol=1e-4,
            its_max_iter=6,
            its_max_individual_iter=20,
            its_tol=1e-4,
            its_omega_damping=0.3,
            imp_n_iter=4,
            imp_n_samples=80,
            imp_proposal_scale=1.0,
            imp_seed=42,
            imp_tol=1e-4,
            imp_e_only=False,
        )
        _assert_common_result_shape(res, n_subjects=n_subjects, n_theta=2)
        if method == "imp":
            assert isinstance(res.get("imp"), dict)
        else:
            assert res.get("imp") is None


def test_nlme_foce_regimens_and_logit_normal_dispatch_smoke():
    n_subjects = 4
    times_per = [0.5, 1.0, 2.0, 4.0, 8.0]
    subject_idx = [sid for sid in range(n_subjects) for _ in times_per]
    times = times_per * n_subjects

    # 2-cpt IV data for regimen dispatch.
    base_iv = nextstat.TwoCompartmentIvPkModel(
        times_per,
        [0.0] * len(times_per),
        dose=100.0,
        error_model="additive",
        sigma=0.05,
    )
    y_iv: list[float] = []
    for sid in range(n_subjects):
        p = [1.3 * math.exp(0.03 * sid), 17.0, 0.9, 23.0]
        pred = base_iv.predict(p)
        y_iv.extend([max(v * (1.0 + 0.01 * ((sid + i) % 2)), 1e-8) for i, v in enumerate(pred)])

    regimens = [
        {
            "events": [
                {"time": 0.0, "amount": 100.0, "route": "infusion", "duration": 2.0},
                {"time": 12.0, "amount": 100.0, "route": "iv_bolus"},
            ]
        }
        for _ in range(n_subjects)
    ]
    res_reg = nextstat.nlme_foce(
        times,
        y_iv,
        subject_idx,
        n_subjects,
        model="2cpt_iv",
        method="focei",
        doses=[100.0],
        regimens=regimens,
        error_model="additive",
        sigma=0.1,
        theta_init=[1.3, 17.0, 0.9, 23.0],
        omega_init=[0.2, 0.2, 0.2, 0.2],
        max_outer_iter=8,
        max_inner_iter=10,
        tol=1e-4,
    )
    _assert_common_result_shape(res_reg, n_subjects=n_subjects, n_theta=4)

    # 1-cpt IV data for regimen dispatch.
    y_1cpt_iv: list[float] = []
    for sid in range(n_subjects):
        cl = 1.2 * math.exp(0.03 * sid)
        v = 15.0
        pred = [(100.0 / v) * math.exp(-(cl / v) * t) for t in times_per]
        y_1cpt_iv.extend([max(vv * (1.0 + 0.01 * ((sid + i) % 2)), 1e-8) for i, vv in enumerate(pred)])

    res_reg_1cpt = nextstat.nlme_foce(
        times,
        y_1cpt_iv,
        subject_idx,
        n_subjects,
        model="1cpt_iv",
        method="focei",
        doses=[100.0],
        regimens=regimens,
        error_model="additive",
        sigma=0.1,
        theta_init=[1.2, 15.0],
        omega_init=[0.2, 0.2],
        max_outer_iter=8,
        max_inner_iter=10,
        tol=1e-4,
    )
    _assert_common_result_shape(res_reg_1cpt, n_subjects=n_subjects, n_theta=2)

    # 1-cpt oral data for logit-normal transform dispatch.
    base_oral = nextstat.OneCompartmentOralPkModel(
        times_per,
        [0.0] * len(times_per),
        dose=120.0,
        bioavailability=1.0,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )
    y_oral: list[float] = []
    for sid in range(n_subjects):
        p = [1.8, 20.0, 1.2 + 0.1 * sid]
        pred = base_oral.predict(p)
        y_oral.extend([max(v, 1e-8) for v in pred])

    res_tr = nextstat.nlme_foce(
        times,
        y_oral,
        subject_idx,
        n_subjects,
        model="1cpt_oral",
        method="focei",
        doses=[120.0],
        bioavailability=1.0,
        error_model="additive",
        sigma=0.08,
        theta_init=[1.8, 20.0, 1.0],
        omega_init=[0.2, 0.2, 0.2],
        random_effect_transforms=[
            "lognormal",
            "lognormal",
            {"type": "logit_normal", "lower": 0.2, "upper": 3.0},
        ],
        max_outer_iter=10,
        max_inner_iter=10,
        tol=1e-4,
    )
    _assert_common_result_shape(res_tr, n_subjects=n_subjects, n_theta=3)
    assert 0.2 < float(res_tr["theta"][2]) < 3.0

    # 3-cpt oral logit-normal transform dispatch.
    t3, y3, idx3, d3 = _simulate_population(
        model="3cpt_oral",
        theta=[1.2, 16.0, 0.9, 22.0, 0.6, 35.0, 1.4],
        n_subjects=n_subjects,
        dose=120.0,
        bioavailability=1.0,
    )
    res_tr3 = nextstat.nlme_foce(
        t3,
        y3,
        idx3,
        n_subjects,
        model="3cpt_oral",
        method="focei",
        doses=d3,
        bioavailability=1.0,
        error_model="additive",
        sigma=0.08,
        theta_init=[1.2, 16.0, 0.9, 22.0, 0.6, 35.0, 1.2],
        omega_init=[0.2] * 7,
        random_effect_transforms=[
            "lognormal",
            "lognormal",
            "lognormal",
            "lognormal",
            "lognormal",
            "lognormal",
            {"type": "logit_normal", "lower": 0.2, "upper": 3.0},
        ],
        max_outer_iter=10,
        max_inner_iter=10,
        tol=1e-4,
    )
    _assert_common_result_shape(res_tr3, n_subjects=n_subjects, n_theta=7)
    assert 0.2 < float(res_tr3["theta"][6]) < 3.0


def test_nlme_saem_regimens_and_logit_normal_dispatch_smoke():
    n_subjects = 4
    times_per = [0.5, 1.0, 2.0, 4.0, 8.0]
    subject_idx = [sid for sid in range(n_subjects) for _ in times_per]
    times = times_per * n_subjects

    base_iv = nextstat.TwoCompartmentIvPkModel(
        times_per,
        [0.0] * len(times_per),
        dose=100.0,
        error_model="additive",
        sigma=0.05,
    )
    y_iv: list[float] = []
    for sid in range(n_subjects):
        pred = base_iv.predict([1.2 * math.exp(0.02 * sid), 16.0, 0.8, 21.0])
        y_iv.extend([max(v, 1e-8) for v in pred])

    regimens = [
        {
            "events": [
                {"time": 0.0, "amount": 100.0, "route": "infusion", "duration": 2.0},
                {"time": 12.0, "amount": 100.0, "route": "iv_bolus"},
            ]
        }
        for _ in range(n_subjects)
    ]

    res_reg = nextstat.nlme_saem(
        times,
        y_iv,
        subject_idx,
        n_subjects,
        model="2cpt_iv",
        doses=[100.0],
        regimens=regimens,
        error_model="additive",
        sigma=0.1,
        theta_init=[1.2, 16.0, 0.8, 21.0],
        omega_init=[0.2, 0.2, 0.2, 0.2],
        n_burn=5,
        n_iter=5,
        n_chains=1,
        seed=42,
        tol=1e-4,
    )
    assert len(res_reg["theta"]) == 4
    assert math.isfinite(float(res_reg["ofv"]))

    y_1cpt_iv: list[float] = []
    for sid in range(n_subjects):
        cl = 1.1 * math.exp(0.03 * sid)
        v = 14.0
        pred = [(100.0 / v) * math.exp(-(cl / v) * t) for t in times_per]
        y_1cpt_iv.extend([max(vv, 1e-8) for vv in pred])
    res_reg_1cpt = nextstat.nlme_saem(
        times,
        y_1cpt_iv,
        subject_idx,
        n_subjects,
        model="1cpt_iv",
        doses=[100.0],
        regimens=regimens,
        error_model="additive",
        sigma=0.1,
        theta_init=[1.1, 14.0],
        omega_init=[0.2, 0.2],
        n_burn=5,
        n_iter=5,
        n_chains=1,
        seed=42,
        tol=1e-4,
    )
    assert len(res_reg_1cpt["theta"]) == 2
    assert math.isfinite(float(res_reg_1cpt["ofv"]))

    base_oral = nextstat.OneCompartmentOralPkModel(
        times_per,
        [0.0] * len(times_per),
        dose=120.0,
        bioavailability=1.0,
        sigma=0.05,
        lloq=None,
        lloq_policy="censored",
    )
    y_oral: list[float] = []
    for sid in range(n_subjects):
        pred = base_oral.predict([1.7, 19.0, 1.1 + 0.08 * sid])
        y_oral.extend([max(v, 1e-8) for v in pred])

    res_tr = nextstat.nlme_saem(
        times,
        y_oral,
        subject_idx,
        n_subjects,
        model="1cpt_oral",
        doses=[120.0],
        bioavailability=1.0,
        error_model="additive",
        sigma=0.08,
        theta_init=[1.7, 19.0, 1.0],
        omega_init=[0.2, 0.2, 0.2],
        random_effect_transforms=[
            "lognormal",
            "lognormal",
            {"type": "logit_normal", "lower": 0.2, "upper": 3.0},
        ],
        n_burn=5,
        n_iter=5,
        n_chains=1,
        seed=42,
        tol=1e-4,
    )
    assert len(res_tr["theta"]) == 3
    assert 0.2 < float(res_tr["theta"][2]) < 3.0

    t2, y2, idx2, d2 = _simulate_population(
        model="2cpt_oral",
        theta=[1.3, 17.0, 0.9, 23.0, 1.5],
        n_subjects=n_subjects,
        dose=120.0,
        bioavailability=1.0,
    )
    res_tr2 = nextstat.nlme_saem(
        t2,
        y2,
        idx2,
        n_subjects,
        model="2cpt_oral",
        doses=d2,
        bioavailability=1.0,
        error_model="additive",
        sigma=0.08,
        theta_init=[1.3, 17.0, 0.9, 23.0, 1.2],
        omega_init=[0.2] * 5,
        random_effect_transforms=[
            "lognormal",
            "lognormal",
            "lognormal",
            "lognormal",
            {"type": "logit_normal", "lower": 0.2, "upper": 3.0},
        ],
        n_burn=5,
        n_iter=5,
        n_chains=1,
        seed=42,
        tol=1e-4,
    )
    assert len(res_tr2["theta"]) == 5
    assert 0.2 < float(res_tr2["theta"][4]) < 3.0
