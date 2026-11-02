"""Heuristic RF optimization for dense station scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rf_engine import NetworkResult, RFParameters, simulate_network


@dataclass
class OptimizationResult:
    before: NetworkResult
    after: NetworkResult
    optimized_power_dbm: np.ndarray
    optimized_channels: np.ndarray
    iterations: int


def optimize_network(
    user_positions: np.ndarray,
    user_demands_mbps: np.ndarray,
    cell_positions: np.ndarray,
    params: RFParameters,
    seed: int | None = 42,
    max_iterations: int = 3,
) -> OptimizationResult:
    """Reduce interference and congestion with channel and power heuristics.

    Channels are greedily assigned to spatially close cells. Busy cells are
    then power-boosted while quiet cells are reduced to limit unnecessary CCI.
    """
    cell_count = len(cell_positions)
    before_channels = np.arange(cell_count) % params.reuse_factor
    before = simulate_network(
        user_positions,
        user_demands_mbps,
        cell_positions,
        params,
        channel_ids=before_channels,
        seed=seed,
    )
    channels = before_channels.copy()
    order = np.argsort(-before.cell_users)
    for cell in order:
        neighbour_distance = np.linalg.norm(cell_positions - cell_positions[cell], axis=1)
        neighbour_distance[cell] = np.inf
        nearest = np.argsort(neighbour_distance)[: max(params.reuse_factor - 1, 1)]
        unavailable = set(channels[nearest].tolist())
        available = [channel for channel in range(params.reuse_factor) if channel not in unavailable]
        if available:
            channels[cell] = available[0]

    powers = np.full(cell_count, params.tx_power_dbm)
    after = before
    for iteration in range(1, max_iterations + 1):
        utilization = after.cell_utilization
        powers = np.clip(
            powers + np.where(utilization > 0.90, 3.0, np.where(utilization < 0.35, -2.0, 0.0)),
            params.tx_power_dbm - 6.0,
            params.tx_power_dbm + 6.0,
        )
        candidate = simulate_network(
            user_positions,
            user_demands_mbps,
            cell_positions,
            params,
            tx_power_overrides_dbm=powers,
            channel_ids=channels,
            seed=seed,
        )
        if candidate.total_throughput_mbps >= after.total_throughput_mbps:
            after = candidate
        else:
            break
    return OptimizationResult(
        before=before,
        after=after,
        optimized_power_dbm=powers,
        optimized_channels=channels,
        iterations=iteration if "iteration" in locals() else 0,
    )
