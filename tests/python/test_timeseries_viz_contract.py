from __future__ import annotations


def test_kalman_viz_artifact_contract_smoke():
    import nextstat

    model = nextstat.timeseries.local_level_model(q=0.1, r=0.2, m0=0.0, p0=1.0)
    ys = [[0.9], [1.2], [None], [1.1]]

    fit = nextstat.timeseries.kalman_fit(model, ys, max_iter=5, tol=1e-9, forecast_steps=2)
    art = nextstat.timeseries.kalman_viz_artifact(fit, ys, level=0.9)

    assert set(art.keys()) >= {"level", "t_obs", "ys", "smooth", "forecast"}
    assert len(art["t_obs"]) == len(ys)
    assert len(art["ys"]) == len(ys)
    assert art["ys"][2][0] is None

    smooth = art["smooth"]
    assert smooth is not None
    assert len(smooth["state_mean"]) == len(ys)
    assert len(smooth["state_lo"]) == len(ys)
    assert len(smooth["state_hi"]) == len(ys)

    # In kalman_fit we have access to H/R via EM output, so obs bands should be present.
    assert smooth["obs_mean"] is not None
    assert smooth["obs_lo"] is not None
    assert smooth["obs_hi"] is not None
    assert len(smooth["obs_mean"]) == len(ys)

    fc = art["forecast"]
    assert fc is not None
    assert len(fc["t"]) == 2
    assert len(fc["obs_mean"]) == 2
    assert len(fc["obs_lo"]) == 2
    assert len(fc["obs_hi"]) == 2


def test_timeseries_plot_helpers_optional_matplotlib():
    import nextstat
    import pytest

    model = nextstat.timeseries.local_level_model(q=0.1, r=0.2, m0=0.0, p0=1.0)
    ys = [[0.9], [1.2], [None], [1.1]]
    fit = nextstat.timeseries.kalman_fit(model, ys, max_iter=2, tol=1e-9, forecast_steps=1)
    art = nextstat.timeseries.kalman_viz_artifact(fit, ys, level=0.9)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as _plt  # noqa: F401
    except Exception:
        with pytest.raises(ImportError):
            nextstat.timeseries.plot_kalman_obs(art)
        with pytest.raises(ImportError):
            nextstat.timeseries.plot_kalman_states(art)
    else:
        ax = nextstat.timeseries.plot_kalman_obs(art, title="Smoke")
        assert ax is not None

        axs = nextstat.timeseries.plot_kalman_states(art, state_indices=[0], title="Smoke")
        assert axs is not None
