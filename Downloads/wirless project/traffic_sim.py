"""Passenger and station-layout simulation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class StationLayout:
    """Rectangular station abstraction used by the RF engine."""

    length_m: float
    width_m: float
    tracks: int
    concourse_area_m2: float


@dataclass
class PassengerTraffic:
    """Generated passenger locations and per-user traffic demand."""

    positions: np.ndarray  # shape: (users, 2), columns are x/y in metres
    demands_mbps: np.ndarray
    profiles: np.ndarray
    weights: np.ndarray


TRAFFIC_PROFILES: Dict[str, Tuple[float, float]] = {
    "Voice": (0.064, 0.15),
    "Telemetry": (0.256, 0.15),
    "Messaging": (0.512, 0.20),
    "Web": (2.0, 0.30),
    "HD video": (8.0, 0.20),
}


def _truncated_normal(
    rng: np.random.Generator,
    mean: np.ndarray,
    std: np.ndarray,
    count: int,
    length: float,
    width: float,
) -> np.ndarray:
    points = rng.normal(mean, std, size=(count, 2))
    points[:, 0] = np.clip(points[:, 0], 0.0, length)
    points[:, 1] = np.clip(points[:, 1], 0.0, width)
    return points


def generate_passengers(
    user_count: int,
    layout: StationLayout,
    seed: int | None = 42,
) -> PassengerTraffic:
    """Create clustered passengers on platforms, tracks, and concourses.

    Platform/trains receive the majority of users, while the central
    concourse gets a broad medium-density distribution.
    """
    if user_count < 1:
        raise ValueError("user_count must be positive")
    if layout.length_m <= 0 or layout.width_m <= 0:
        raise ValueError("station dimensions must be positive")
    if layout.tracks < 1:
        raise ValueError("tracks must be at least one")

    rng = np.random.default_rng(seed)
    platform_share = 0.72
    platform_count = int(user_count * platform_share)
    concourse_count = user_count - platform_count
    platform_y = np.linspace(
        layout.width_m * 0.12, layout.width_m * 0.88, layout.tracks + 1
    )
    platform_y = platform_y[::2] if len(platform_y) > 2 else platform_y
    platform_y = np.maximum(platform_y, layout.width_m * 0.08)

    platform_centres = np.column_stack(
        [
            rng.uniform(layout.length_m * 0.08, layout.length_m * 0.92, len(platform_y)),
            platform_y,
        ]
    )
    platform_assignments = rng.integers(0, len(platform_centres), platform_count)
    platform_points = np.empty((platform_count, 2))
    for cluster_index, centre in enumerate(platform_centres):
        selected = np.flatnonzero(platform_assignments == cluster_index)
        if selected.size:
            platform_points[selected] = _truncated_normal(
                rng,
                centre,
                np.array([layout.length_m * 0.10, max(layout.width_m * 0.035, 2.0)]),
                selected.size,
                layout.length_m,
                layout.width_m,
            )

    concourse_centre = np.array([layout.length_m * 0.50, layout.width_m * 0.50])
    concourse_points = _truncated_normal(
        rng,
        concourse_centre,
        np.array(
            [
                max(layout.length_m * 0.18, np.sqrt(layout.concourse_area_m2) * 0.45),
                max(layout.width_m * 0.18, np.sqrt(layout.concourse_area_m2) * 0.35),
            ]
        ),
        concourse_count,
        layout.length_m,
        layout.width_m,
    )
    positions = np.vstack((platform_points, concourse_points))
    rng.shuffle(positions, axis=0)

    profile_names = np.array(list(TRAFFIC_PROFILES))
    profile_weights = np.array([value[1] for value in TRAFFIC_PROFILES.values()])
    profiles = rng.choice(profile_names, size=user_count, p=profile_weights)
    demands = np.array([TRAFFIC_PROFILES[profile][0] for profile in profiles])
    demand_variation = rng.lognormal(mean=0.0, sigma=0.18, size=user_count)
    return PassengerTraffic(
        positions=positions,
        demands_mbps=demands * demand_variation,
        profiles=profiles,
        weights=profile_weights,
    )


def place_cells(
    layout: StationLayout,
    macro_count: int,
    micro_count: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Place macro cells over the station and micro cells along its platforms."""
    if macro_count < 1 or micro_count < 0:
        raise ValueError("macro_count must be >= 1 and micro_count must be >= 0")
    macro_x = np.linspace(
        layout.length_m * 0.15, layout.length_m * 0.85, macro_count
    )
    macro_y = np.full(macro_count, layout.width_m * 0.50)
    macros = np.column_stack((macro_x, macro_y))

    if micro_count:
        micro_x = np.linspace(
            layout.length_m * 0.10, layout.length_m * 0.90, micro_count
        )
        platform_y = layout.width_m * 0.22 + (
            np.arange(micro_count) % max(layout.tracks, 1)
        ) * layout.width_m * 0.56 / max(layout.tracks - 1, 1)
        micros = np.column_stack((micro_x, platform_y))
    else:
        micros = np.empty((0, 2))
    return macros, micros
