from __future__ import annotations

from pathlib import Path

import pytest


def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401

        return True
    except Exception:
        return False


def _has_gym() -> bool:
    try:
        import gymnasium  # noqa: F401

        return True
    except Exception:
        try:
            import gym  # noqa: F401

            return True
        except Exception:
            return False


_REQUIRES_GYM = pytest.mark.skipif(
    not (_has_numpy() and _has_gym()),
    reason="requires numpy + gymnasium (preferred) or gym",
)


@_REQUIRES_GYM
def test_histfactory_env_reset_step_shapes() -> None:
    import numpy as np

    import nextstat
    from nextstat.gym import make_histfactory_env

    ws_json = Path("tests/fixtures/simple_workspace.json").read_text(encoding="utf-8")
    env = make_histfactory_env(
        ws_json,
        channel="singlechannel",
        sample="signal",
        reward_metric="nll",
        max_steps=3,
        init_noise=0.0,
    )

    out = env.reset(seed=123)
    obs = out[0] if isinstance(out, tuple) else out
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1
    assert obs.shape == env.observation_space.shape

    action = np.zeros(env.action_space.shape, dtype=np.float64)
    step_out = env.step(action)

    # Gymnasium: (obs, reward, terminated, truncated, info)
    # Gym:       (obs, reward, done, info)
    assert isinstance(step_out[0], np.ndarray)
    assert step_out[0].shape == env.observation_space.shape
    assert isinstance(step_out[1], float)


@_REQUIRES_GYM
def test_histfactory_env_determinism_fixed_seed() -> None:
    import numpy as np

    from nextstat.gym import make_histfactory_env

    ws_json = Path("tests/fixtures/simple_workspace.json").read_text(encoding="utf-8")

    def rollout(seed: int):
        env = make_histfactory_env(
            ws_json,
            channel="singlechannel",
            sample="signal",
            reward_metric="nll",
            max_steps=5,
            init_noise=0.05,
            action_scale=0.01,
        )
        out = env.reset(seed=seed)
        obs0 = out[0] if isinstance(out, tuple) else out

        actions = [
            np.zeros(env.action_space.shape, dtype=np.float64),
            np.ones(env.action_space.shape, dtype=np.float64) * 0.25,
            np.ones(env.action_space.shape, dtype=np.float64) * -0.5,
        ]

        traj = [(obs0.copy(), None)]
        for a in actions:
            step_out = env.step(a)
            obs = step_out[0]
            reward = step_out[1]
            traj.append((obs.copy(), float(reward)))
        return traj

    t1 = rollout(123)
    t2 = rollout(123)

    assert len(t1) == len(t2)
    for (o1, r1), (o2, r2) in zip(t1, t2):
        assert np.allclose(o1, o2)
        if r1 is None:
            assert r2 is None
        else:
            assert r2 is not None
            assert abs(float(r1) - float(r2)) < 1e-12


@_REQUIRES_GYM
def test_histfactory_env_perf_smoke_runs() -> None:
    import numpy as np

    from nextstat.gym import make_histfactory_env

    ws_json = Path("tests/fixtures/simple_workspace.json").read_text(encoding="utf-8")
    env = make_histfactory_env(
        ws_json,
        channel="singlechannel",
        sample="signal",
        reward_metric="nll",
        max_steps=50,
        init_noise=0.0,
        action_scale=0.01,
    )
    env.reset(seed=7)
    action = np.zeros(env.action_space.shape, dtype=np.float64)
    for _ in range(50):
        env.step(action)

