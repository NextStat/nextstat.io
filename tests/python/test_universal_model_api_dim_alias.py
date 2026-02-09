"""Universal Model API (Phase 5): `dim()` alias contract.

Rust core uses `LogDensityModel.dim()`. Python historically exposed `n_params()`.
We expose both to keep Python code generic across model families.
"""

from __future__ import annotations

from pathlib import Path

import nextstat


REPO = Path(__file__).resolve().parents[2]


def test_dim_alias_matches_n_params_across_models():
    # HistFactory
    ws = (REPO / "tests" / "fixtures" / "simple_workspace.json").read_text()
    hf = nextstat.HistFactoryModel.from_workspace(ws)

    # Simple non-HEP models
    funnel = nextstat.FunnelModel()
    gm = nextstat.GaussianMeanModel([1.0, 2.0, 3.0], 1.0)
    sn = nextstat.StdNormalModel(dim=3)

    # GLM
    x = [[1.0], [2.0], [3.0], [4.0]]
    y_lin = [1.0, 2.0, 3.0, 4.0]
    y_logit = [0, 0, 1, 1]
    y_pois = [1, 2, 3, 4]
    y_nb = [1, 2, 3, 4]

    lr = nextstat.LinearRegressionModel(x, y_lin, include_intercept=True)
    logit = nextstat.LogisticRegressionModel(x, y_logit, include_intercept=True)
    pois = nextstat.PoissonRegressionModel(x, y_pois, include_intercept=True)
    nb = nextstat.NegativeBinomialRegressionModel(x, y_nb, include_intercept=True)

    # Ordinal
    ord_x = [[-1.0], [0.0], [1.0], [2.0]]
    ord_y = [0, 1, 2, 2]
    ologit = nextstat.OrderedLogitModel(ord_x, ord_y, n_levels=3)
    oprobit = nextstat.OrderedProbitModel(ord_x, ord_y, n_levels=3)

    # Composed (builder)
    cglm = nextstat.ComposedGlmModel.linear_regression(x, y_lin, include_intercept=True)

    # LMM
    lmm = nextstat.LmmMarginalModel(x, y_lin, include_intercept=True, group_idx=[0, 0, 1, 1])

    # Survival
    times = [1.0, 2.0, 3.0, 4.0]
    events = [True, False, True, True]
    exp = nextstat.ExponentialSurvivalModel(times, events)
    weib = nextstat.WeibullSurvivalModel(times, events)
    lnaft = nextstat.LogNormalAftModel(times, events)
    cox = nextstat.CoxPhModel(times, events, x, ties="efron")

    # PK
    pk_times = [0.5, 1.0, 2.0, 4.0]
    pk_y = [0.2, 0.4, 0.3, 0.1]
    pk = nextstat.OneCompartmentOralPkModel(pk_times, pk_y, dose=10.0, sigma=0.1)
    pk_nlme = nextstat.OneCompartmentOralPkNlmeModel(
        pk_times,
        pk_y,
        subject_idx=[0, 0, 1, 1],
        n_subjects=2,
        dose=10.0,
        sigma=0.1,
    )

    models = [
        hf,
        funnel,
        gm,
        sn,
        lr,
        logit,
        pois,
        nb,
        ologit,
        oprobit,
        cglm,
        lmm,
        exp,
        weib,
        lnaft,
        cox,
        pk,
        pk_nlme,
    ]

    for m in models:
        assert m.dim() == m.n_params()
