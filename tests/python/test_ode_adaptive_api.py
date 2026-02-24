import math

import nextstat


def test_lsoda_linear_smoke():
    a = [[-0.3]]
    y0 = [2.0]
    out = nextstat.ode.lsoda(a, y0, 0.0, 4.0, rtol=1e-8, atol=1e-10)
    assert "t" in out and "y" in out
    assert len(out["t"]) == len(out["y"])
    assert len(out["t"]) >= 2
    y_final = float(out["y"][-1][0])
    expected = 2.0 * math.exp(-0.3 * 4.0)
    assert abs(y_final - expected) < 5e-4


def test_forward_sensitivity_linear_smoke():
    # dy/dt = p * y, y(0)=1 => y(t)=exp(p t), dy/dp = t exp(p t)
    a_params = [[[1.0]]]
    params = [0.4]
    out = nextstat.ode.forward_sensitivity_solve(
        a_params,
        params,
        [1.0],
        0.0,
        3.0,
        solver="lsoda",
        rtol=1e-8,
        atol=1e-10,
    )
    assert "t" in out and "y" in out and "sens" in out
    assert len(out["t"]) == len(out["y"]) == len(out["sens"])
    y_final = float(out["y"][-1][0])
    s_final = float(out["sens"][-1][0][0])
    expected_y = math.exp(0.4 * 3.0)
    expected_s = 3.0 * expected_y
    assert abs(y_final - expected_y) < 5e-4
    assert abs(s_final - expected_s) < 2e-3


def test_lsoda_callback_smoke():
    def rhs(t: float, y: list[float]) -> list[float]:
        return [-0.3 * y[0]]

    def jac(t: float, y: list[float]) -> list[list[float]]:
        return [[-0.3]]

    out = nextstat.ode.lsoda(rhs, [2.0], 0.0, 4.0, jac=jac, rtol=1e-8, atol=1e-10)
    y_final = float(out["y"][-1][0])
    expected = 2.0 * math.exp(-0.3 * 4.0)
    assert abs(y_final - expected) < 5e-4


def test_forward_sensitivity_callback_smoke():
    # dy/dt = p * y, y(0)=1 => y(t)=exp(p t), dy/dp = t exp(p t)
    def rhs(t: float, y: list[float], params: list[float]) -> list[float]:
        return [params[0] * y[0]]

    def jac_y(t: float, y: list[float], params: list[float]) -> list[list[float]]:
        return [[params[0]]]

    def jac_params(t: float, y: list[float], params: list[float]) -> list[list[float]]:
        return [[y[0]]]

    out = nextstat.ode.forward_sensitivity_solve(
        rhs,
        [0.4],
        [1.0],
        0.0,
        3.0,
        jac_y=jac_y,
        jac_params=jac_params,
        solver="lsoda",
        rtol=1e-8,
        atol=1e-10,
    )
    y_final = float(out["y"][-1][0])
    s_final = float(out["sens"][-1][0][0])
    expected_y = math.exp(0.4 * 3.0)
    expected_s = 3.0 * expected_y
    assert abs(y_final - expected_y) < 5e-4
    assert abs(s_final - expected_s) < 2e-3
