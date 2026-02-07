from __future__ import annotations

import json
from pathlib import Path

import nextstat


def _load_signal_nominal_from_workspace(path: Path, *, channel: str, sample: str) -> list[float]:
    ws = json.loads(path.read_text(encoding="utf-8"))
    for ch in ws["channels"]:
        if ch["name"] != channel:
            continue
        for s in ch["samples"]:
            if s["name"] == sample:
                return [float(x) for x in s["data"]]
    raise KeyError(f"Missing channel/sample: {channel}/{sample}")


def test_q0_like_loss_and_grad_nominal_finite_diff() -> None:
    ws_path = Path("tests/fixtures/simple_workspace.json")
    ws_json = ws_path.read_text(encoding="utf-8")
    model = nextstat.HistFactoryModel.from_workspace(ws_json)

    nominal = _load_signal_nominal_from_workspace(ws_path, channel="singlechannel", sample="signal")
    mle = nextstat.MaximumLikelihoodEstimator(max_iter=400, tol=1e-6, m=10)

    q0, grad = mle.q0_like_loss_and_grad_nominal(
        model, channel="singlechannel", sample="signal", nominal=nominal
    )
    assert q0 == q0  # not NaN
    assert len(grad) == len(nominal)

    idx = 0
    for i, v in enumerate(nominal):
        if v > 0.5:
            idx = i
            break

    eps = max(1e-3, 1e-3 * abs(nominal[idx]))
    plus = list(nominal)
    plus[idx] += eps
    q0_p, _ = mle.q0_like_loss_and_grad_nominal(
        model, channel="singlechannel", sample="signal", nominal=plus
    )

    minus = list(nominal)
    minus[idx] -= eps
    q0_m, _ = mle.q0_like_loss_and_grad_nominal(
        model, channel="singlechannel", sample="signal", nominal=minus
    )

    fd = (q0_p - q0_m) / (2.0 * eps)
    g = grad[idx]
    denom = max(abs(fd), abs(g), 1e-6)
    rel = abs(fd - g) / denom
    assert rel < 5e-2

