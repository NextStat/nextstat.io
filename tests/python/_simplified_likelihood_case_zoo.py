from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimplifiedLikelihoodCaseConfig:
    name: str
    n_channels: int
    bins_per_channel: int
    latent_rank: int
    full_nuisance_count: int
    seed: int = 0


@dataclass(frozen=True)
class SimplifiedLikelihoodBenchmarkCase:
    name: str
    measurement: str
    workspace: dict[str, Any]
    simplified_workspace: dict[str, Any]
    full_nuisance_count: int
    latent_rank: int


_SUITES: dict[str, list[SimplifiedLikelihoodCaseConfig]] = {
    "smoke": [
        SimplifiedLikelihoodCaseConfig(
            name="synthetic_covariance_smoke",
            n_channels=2,
            bins_per_channel=3,
            latent_rank=4,
            full_nuisance_count=24,
            seed=7,
        ),
    ],
    "ci": [
        SimplifiedLikelihoodCaseConfig(
            name="synthetic_covariance_smoke",
            n_channels=2,
            bins_per_channel=3,
            latent_rank=4,
            full_nuisance_count=24,
            seed=7,
        ),
        SimplifiedLikelihoodCaseConfig(
            name="synthetic_covariance_medium",
            n_channels=3,
            bins_per_channel=4,
            latent_rank=6,
            full_nuisance_count=72,
            seed=11,
        ),
    ],
    "bench": [
        SimplifiedLikelihoodCaseConfig(
            name="synthetic_covariance_medium",
            n_channels=3,
            bins_per_channel=4,
            latent_rank=6,
            full_nuisance_count=72,
            seed=11,
        ),
        SimplifiedLikelihoodCaseConfig(
            name="synthetic_covariance_large",
            n_channels=4,
            bins_per_channel=6,
            latent_rank=8,
            full_nuisance_count=128,
            seed=23,
        ),
    ],
}


def available_suite_names() -> list[str]:
    return sorted(_SUITES)


def make_suite(name: str) -> list[SimplifiedLikelihoodBenchmarkCase]:
    try:
        configs = _SUITES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown simplified-likelihood suite '{name}'. Available: {available_suite_names()}") from exc
    return [make_case(config) for config in configs]


def make_case(config: SimplifiedLikelihoodCaseConfig) -> SimplifiedLikelihoodBenchmarkCase:
    channel_names = [f"SR{i}" for i in range(config.n_channels)]
    total_bins = config.n_channels * config.bins_per_channel

    background_nominal: list[float] = []
    signal_nominal: list[float] = []
    flattened_bins: list[dict[str, str]] = []

    for channel_index, channel_name in enumerate(channel_names):
        for bin_index in range(config.bins_per_channel):
            base_bkg = (
                35.0
                + 2.5 * channel_index
                + 1.1 * bin_index
                + 0.2 * (((channel_index + 1) * (bin_index + 1) + config.seed) % 3)
            )
            base_sig = 1.8 + 0.2 * channel_index + 0.1 * bin_index
            background_nominal.append(round(base_bkg, 6))
            signal_nominal.append(round(base_sig, 6))
            flattened_bins.append({"channel": channel_name, "name": f"bin{bin_index}"})

    latent_modes = _make_latent_modes(
        background_nominal=background_nominal,
        latent_rank=config.latent_rank,
        seed=config.seed,
    )
    shifts = _make_full_nuisance_shifts(
        latent_modes=latent_modes,
        full_nuisance_count=config.full_nuisance_count,
        seed=config.seed,
    )
    total_covariance = _covariance_from_shifts(shifts)

    observed = [
        round(background_nominal[i] + signal_nominal[i], 6) for i in range(total_bins)
    ]
    workspace = _build_full_workspace(
        channel_names=channel_names,
        bins_per_channel=config.bins_per_channel,
        background_nominal=background_nominal,
        signal_nominal=signal_nominal,
        observed=observed,
        shifts=shifts,
    )
    simplified_workspace = _build_simplified_workspace(
        name=config.name,
        bins=flattened_bins,
        observed=observed,
        background_nominal=background_nominal,
        signal_nominal=signal_nominal,
        total_covariance=total_covariance,
    )
    return SimplifiedLikelihoodBenchmarkCase(
        name=config.name,
        measurement="m",
        workspace=workspace,
        simplified_workspace=simplified_workspace,
        full_nuisance_count=config.full_nuisance_count,
        latent_rank=config.latent_rank,
    )


def _make_latent_modes(
    *, background_nominal: list[float], latent_rank: int, seed: int
) -> list[list[float]]:
    modes: list[list[float]] = []
    for mode_index in range(latent_rank):
        mode: list[float] = []
        for bin_index, nominal in enumerate(background_nominal):
            frac = (
                0.015
                + 0.002 * ((bin_index + mode_index + seed) % 5)
                + 0.0005 * (mode_index % 4)
            )
            sign = -1.0 if ((bin_index + seed + (mode_index + 1) * (mode_index + 2)) % (mode_index + 2) == 0) else 1.0
            mode.append(sign * frac * nominal)
        modes.append(mode)
    return modes


def _make_full_nuisance_shifts(
    *, latent_modes: list[list[float]], full_nuisance_count: int, seed: int
) -> list[list[float]]:
    latent_rank = len(latent_modes)
    total_bins = len(latent_modes[0]) if latent_modes else 0
    shifts: list[list[float]] = []

    for nuisance_index in range(full_nuisance_count):
        if nuisance_index < latent_rank:
            weights = [0.0] * latent_rank
            weights[nuisance_index] = 1.0
        else:
            weights = []
            for mode_index in range(latent_rank):
                raw = (
                    ((nuisance_index + 1) * (2 * mode_index + 3) + mode_index + seed) % 13
                ) - 6
                if raw == 0:
                    raw = 2 if ((nuisance_index + mode_index + seed) % 2 == 0) else -3
                weights.append(raw / 7.0)
            norm = math.sqrt(sum(weight * weight for weight in weights)) or 1.0
            weights = [weight / norm for weight in weights]

        scale = 0.7 + 0.01 * ((nuisance_index + seed) % 7)
        shift = [0.0] * total_bins
        for mode_index, weight in enumerate(weights):
            for bin_index in range(total_bins):
                shift[bin_index] += scale * weight * latent_modes[mode_index][bin_index]
        shifts.append(shift)

    return shifts


def _covariance_from_shifts(shifts: list[list[float]]) -> list[list[float]]:
    total_bins = len(shifts[0]) if shifts else 0
    covariance = [[0.0] * total_bins for _ in range(total_bins)]
    for shift in shifts:
        for row in range(total_bins):
            for col in range(total_bins):
                covariance[row][col] += shift[row] * shift[col]
    return covariance


def _build_full_workspace(
    *,
    channel_names: list[str],
    bins_per_channel: int,
    background_nominal: list[float],
    signal_nominal: list[float],
    observed: list[float],
    shifts: list[list[float]],
) -> dict[str, Any]:
    channels: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    cursor = 0

    for channel_name in channel_names:
        channel_background = background_nominal[cursor : cursor + bins_per_channel]
        channel_signal = signal_nominal[cursor : cursor + bins_per_channel]
        channel_observed = observed[cursor : cursor + bins_per_channel]

        background_modifiers = []
        for nuisance_index, shift in enumerate(shifts):
            hi = [
                round(max(1e-6, channel_background[i] + shift[cursor + i]), 6)
                for i in range(bins_per_channel)
            ]
            lo = [
                round(max(1e-6, channel_background[i] - shift[cursor + i]), 6)
                for i in range(bins_per_channel)
            ]
            background_modifiers.append(
                {
                    "name": f"theta_{nuisance_index:03d}",
                    "type": "histosys",
                    "data": {"hi_data": hi, "lo_data": lo},
                }
            )

        channels.append(
            {
                "name": channel_name,
                "samples": [
                    {
                        "name": "signal",
                        "data": channel_signal,
                        "modifiers": [{"name": "mu", "type": "normfactor", "data": None}],
                    },
                    {
                        "name": "background",
                        "data": channel_background,
                        "modifiers": background_modifiers,
                    },
                ],
            }
        )
        observations.append({"name": channel_name, "data": channel_observed})
        cursor += bins_per_channel

    return {
        "channels": channels,
        "observations": observations,
        "measurements": [{"name": "m", "config": {"poi": "mu", "parameters": []}}],
        "version": "1.0.0",
    }


def _build_simplified_workspace(
    *,
    name: str,
    bins: list[dict[str, str]],
    observed: list[float],
    background_nominal: list[float],
    signal_nominal: list[float],
    total_covariance: list[list[float]],
) -> dict[str, Any]:
    return {
        "schema_version": "nextstat_simplified_likelihood_v0",
        "metadata": {
            "experiment": "Synthetic",
            "analysis_id": name,
            "source_format": "covariance",
            "reference": "internal-apex2",
            "description": "Deterministic synthetic covariance-form simplified likelihood for Apex2 fidelity and speedup reporting.",
        },
        "poi": {"name": "mu", "init": 1.0, "bounds": [0.0, 5.0]},
        "bins": bins,
        "observed": observed,
        "background_nominal": background_nominal,
        "signal_nominal": signal_nominal,
        "uncertainty_model": {
            "kind": "covariance",
            "total_covariance": total_covariance,
        },
    }
