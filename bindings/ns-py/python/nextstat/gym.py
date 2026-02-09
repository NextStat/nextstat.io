"""Gymnasium/Gym environments for NextStat (optional).

This module is intentionally optional and only imports `gymnasium` / `gym`
and `numpy` when used.

Primary use case: quick RL / DOE playgrounds where the "environment" reward
is a NextStat metric computed on a HistFactory workspace (NLL / q0 / qmu / Z).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Optional


def _require_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "`nextstat.gym` requires numpy (usually installed with gymnasium)."
        ) from e


def _require_gym():
    try:
        import gymnasium as gym  # type: ignore

        return gym, True
    except Exception:
        try:
            import gym  # type: ignore

            return gym, False
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "`nextstat.gym` requires gymnasium (preferred) or gym."
            ) from e


def _load_sample_nominal_from_workspace_json(
    workspace_json: str,
    *,
    channel: str,
    sample: str,
) -> list[float]:
    ws = json.loads(workspace_json)
    for ch in ws.get("channels", []):
        if ch.get("name") != channel:
            continue
        for s in ch.get("samples", []):
            if s.get("name") == sample:
                return [float(x) for x in s.get("data", [])]
    raise KeyError(f"Missing channel/sample: {channel}/{sample}")


RewardMetric = Literal["nll", "q0", "z0", "qmu", "zmu"]
ActionMode = Literal["add", "logmul"]


@dataclass(frozen=True)
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class HistFactoryEnv:
    """Minimal Gymnasium/Gym env for optimizing a single sample's nominal yields.

    Observation: current signal nominal yields (vector).
    Action: per-bin delta (vector), applied as:
      - `action_mode="add"`   : s <- clamp(s + action_scale * a)
      - `action_mode="logmul"`: s <- clamp(s * exp(action_scale * a))

    Reward (to maximize):
      - `reward_metric="nll"` : `-model.nll(fixed_params)`  (fast; no profiling)
      - `reward_metric="q0"`  : discovery test statistic q0 (profiled; slower)
      - `reward_metric="z0"`  : sqrt(q0)
      - `reward_metric="qmu"` : `-qmu(mu_test)` (upper-limit style; slower)
      - `reward_metric="zmu"` : `-sqrt(qmu(mu_test))`

    Notes:
    - For `reward_metric="nll"` the model must support in-place sample nominal overrides
      via `HistFactoryModel.set_sample_nominal(...)` (NextStat API).
    - For profiled rewards, the env uses `MaximumLikelihoodEstimator.*_like_loss_and_grad_nominal`
      which also updates the model's sample nominal internally.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        workspace_json: str,
        channel: str,
        sample: str,
        reward_metric: RewardMetric = "nll",
        mu_test: float = 5.0,
        max_steps: int = 128,
        action_scale: float = 0.05,
        action_mode: ActionMode = "logmul",
        init_noise: float = 0.0,
        clip_min: float = 1e-12,
        clip_max: float = 1e12,
        fixed_params: Optional[list[float]] = None,
        mle_max_iter: int = 200,
        mle_tol: float = 1e-6,
        mle_m: int = 10,
    ) -> None:
        np = _require_numpy()
        gym, is_gymnasium = _require_gym()

        import nextstat  # local import: optional package layout

        if not (max_steps > 0):
            raise ValueError("max_steps must be > 0")
        if not (action_scale > 0.0 and math.isfinite(action_scale)):
            raise ValueError("action_scale must be finite and > 0")
        if not (clip_min > 0.0 and clip_max > clip_min):
            raise ValueError("clip_min/clip_max must satisfy 0 < clip_min < clip_max")
        if reward_metric in ("qmu", "zmu") and not (mu_test >= 0.0 and math.isfinite(mu_test)):
            raise ValueError("mu_test must be finite and >= 0")

        self._np = np
        self._gym = gym
        self._is_gymnasium = is_gymnasium

        self.channel = channel
        self.sample = sample
        self.reward_metric = reward_metric
        self.mu_test = float(mu_test)
        self.max_steps = int(max_steps)
        self.action_scale = float(action_scale)
        self.action_mode = action_mode
        self.init_noise = float(init_noise)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        self._workspace_json = workspace_json
        self._base_nominal = _load_sample_nominal_from_workspace_json(
            workspace_json, channel=channel, sample=sample
        )
        if len(self._base_nominal) == 0:
            raise ValueError("Sample nominal is empty (no bins)")

        self.model = nextstat.HistFactoryModel.from_workspace(workspace_json)
        self._fixed_params = fixed_params or self.model.suggested_init()
        if len(self._fixed_params) != self.model.n_params():
            raise ValueError("fixed_params length must match model.n_params()")

        self._mle = nextstat.MaximumLikelihoodEstimator(
            max_iter=int(mle_max_iter),
            tol=float(mle_tol),
            m=int(mle_m),
        )

        n_bins = len(self._base_nominal)
        self.observation_space = gym.spaces.Box(
            low=np.full((n_bins,), self.clip_min, dtype=np.float64),
            high=np.full((n_bins,), self.clip_max, dtype=np.float64),
            shape=(n_bins,),
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(n_bins,), dtype=np.float64
        )

        self._rng = None
        self._step = 0
        self._signal = np.array(self._base_nominal, dtype=np.float64)

    def _obs(self):
        return self._signal.copy()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        np = self._np
        del options

        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            try:
                self.action_space.seed(int(seed))
            except Exception:
                pass
        elif self._rng is None:
            self._rng = np.random.default_rng()

        self._step = 0
        base = np.array(self._base_nominal, dtype=np.float64)
        if self.init_noise > 0.0:
            noise = self._rng.normal(0.0, self.init_noise, size=base.shape)
            base = base * np.exp(noise)
        self._signal = np.clip(base, self.clip_min, self.clip_max)

        info: dict[str, Any] = {"seed": seed}
        if self._is_gymnasium:
            return self._obs(), info
        return self._obs()

    def step(self, action):
        np = self._np

        a = np.asarray(action, dtype=np.float64)
        if a.shape != self._signal.shape:
            raise ValueError(f"Action shape {a.shape} != expected {self._signal.shape}")

        if self.action_mode == "add":
            updated = self._signal + (self.action_scale * a)
        elif self.action_mode == "logmul":
            updated = self._signal * np.exp(self.action_scale * a)
        else:  # pragma: no cover
            raise ValueError(f"Unknown action_mode={self.action_mode!r}")

        self._signal = np.clip(updated, self.clip_min, self.clip_max)
        self._step += 1

        reward, info = self._compute_reward()

        terminated = False
        truncated = self._step >= self.max_steps

        obs = self._obs()

        if self._is_gymnasium:
            return obs, float(reward), terminated, truncated, info
        done = bool(terminated or truncated)
        return obs, float(reward), done, info

    def _compute_reward(self) -> tuple[float, dict[str, Any]]:
        signal_list = [float(x) for x in self._signal.tolist()]
        info: dict[str, Any] = {"reward_metric": self.reward_metric}

        if self.reward_metric == "nll":
            # Fast path: override sample nominal + evaluate NLL at fixed params.
            self.model.set_sample_nominal(
                channel=self.channel,
                sample=self.sample,
                nominal=signal_list,
            )
            nll_val = float(self.model.nll(self._fixed_params))
            info["nll"] = nll_val
            return -nll_val, info

        if self.reward_metric == "q0":
            q0, _grad = self._mle.q0_like_loss_and_grad_nominal(
                self.model,
                channel=self.channel,
                sample=self.sample,
                nominal=signal_list,
            )
            info["q0"] = float(q0)
            return float(q0), info

        if self.reward_metric == "z0":
            q0, _grad = self._mle.q0_like_loss_and_grad_nominal(
                self.model,
                channel=self.channel,
                sample=self.sample,
                nominal=signal_list,
            )
            z0 = math.sqrt(max(float(q0), 0.0))
            info["q0"] = float(q0)
            info["z0"] = z0
            return z0, info

        if self.reward_metric == "qmu":
            qmu, _grad = self._mle.qmu_like_loss_and_grad_nominal(
                self.model,
                mu_test=self.mu_test,
                channel=self.channel,
                sample=self.sample,
                nominal=signal_list,
            )
            info["qmu"] = float(qmu)
            return -float(qmu), info

        if self.reward_metric == "zmu":
            qmu, _grad = self._mle.qmu_like_loss_and_grad_nominal(
                self.model,
                mu_test=self.mu_test,
                channel=self.channel,
                sample=self.sample,
                nominal=signal_list,
            )
            zmu = math.sqrt(max(float(qmu), 0.0))
            info["qmu"] = float(qmu)
            info["zmu"] = zmu
            return -zmu, info

        raise ValueError(f"Unknown reward_metric={self.reward_metric!r}")


def make_histfactory_env(
    workspace_json: str,
    *,
    channel: str,
    sample: str,
    reward_metric: RewardMetric = "nll",
    **kwargs: Any,
) -> HistFactoryEnv:
    """Convenience factory returning `HistFactoryEnv`."""
    return HistFactoryEnv(
        workspace_json=workspace_json,
        channel=channel,
        sample=sample,
        reward_metric=reward_metric,
        **kwargs,
    )

