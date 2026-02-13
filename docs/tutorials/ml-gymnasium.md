---
title: "ML: Gymnasium RL Environment"
status: experimental
---

# ML: Gymnasium RL Environment (`nextstat.gym`)

NextStat includes an **optional** Gymnasium/Gym wrapper to treat a HistFactory workspace as an RL/DOE environment.

It is designed as a playground: you propose updates to a sample's nominal yields (e.g. signal histogram)
and receive a NextStat metric as reward.

## Install

You need:
- `nextstat` (Python package)
- `gymnasium` (preferred) or `gym`
- `numpy`

## Minimal example (maximize discovery q₀)

```python
from pathlib import Path

import numpy as np

from nextstat.gym import make_histfactory_env

ws_json = Path("tests/fixtures/simple_workspace.json").read_text()

env = make_histfactory_env(
    ws_json,
    channel="singlechannel",
    sample="signal",
    reward_metric="q0",      # maximize q0
    max_steps=64,
    action_scale=0.02,
    action_mode="logmul",
    init_noise=0.0,
)

reset_out = env.reset(seed=123)
if isinstance(reset_out, tuple) and len(reset_out) == 2:
    obs, info = reset_out
else:
    obs, info = reset_out, {}
total = 0.0
for _ in range(64):
    action = env.action_space.sample()  # baseline: random
    step_out = env.step(action)
    if isinstance(step_out, tuple) and len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = step_out
    total += float(reward)
    if done:
        break

print("episode reward:", total)
```

## Reward metrics

`HistFactoryEnv` returns reward **to maximize**:

- `"nll"`: `-NLL(params_fixed)` (fast; no profiling)
- `"q0"` / `"z0"`: discovery test statistic / significance (profiled)
- `"qmu"` / `"zmu"`: `-qμ(μ_test)` / `-sqrt(qμ)` (upper-limit style; profiled)

## Notes

- `reward_metric="nll"` is the fast mode intended for “many steps quickly”.
- Profiled rewards (`q0/qmu`) run optimization internally; they are much heavier per step.
- The env modifies the model in-place by overriding one sample’s nominal yields.
