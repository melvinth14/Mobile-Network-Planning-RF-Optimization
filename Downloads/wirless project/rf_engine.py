"""Vectorized RF propagation, association, interference, and capacity models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RFParameters:
    frequency_mhz: float = 3500.0
    tx_power_dbm: float = 43.0
    cell_height_m: float = 12.0
    antenna_gain_dbi: float = 15.0
    noise_floor_dbm: float = -96.0
    bandwidth_mhz: float = 100.0
    path_loss_exponent: float = 2.8
    shadowing_std_db: float = 4.0
    reuse_factor: int = 3
    max_users_per_cell: int = 1200
    max_utilization: float = 0.90


@dataclass
class NetworkResult:
    serving_cell: np.ndarray
    received_power_dbm: np.ndarray
    sinr_db: np.ndarray
    user_throughput_mbps: np.ndarray
    cell_throughput_mbps: np.ndarray
    cell_users: np.ndarray
    cell_utilization: np.ndarray
    dropped: np.ndarray
    overloaded: np.ndarray
    channel_ids: np.ndarray
    distance_m: np.ndarray
    rx_power_matrix_dbm: np.ndarray

    @property
    def coverage_rate(self) -> float:
        return float(np.mean(~self.dropped) * 100.0)

    @property
    def mean_sinr_db(self) -> float:
        return float(np.mean(self.sinr_db))

    @property
    def total_throughput_mbps(self) -> float:
        return float(np.sum(self.user_throughput_mbps))

    @property
    def congestion_rate(self) -> float:
        return float(np.mean(self.cell_utilization >= 1.0) * 100.0)


def dbm_to_mw(value_dbm: np.ndarray | float) -> np.ndarray:
    return np.power(10.0, np.asarray(value_dbm) / 10.0)


def mw_to_dbm(value_mw: np.ndarray | float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(value_mw), 1e-15))


def fspl_db(distance_m: np.ndarray | float, frequency_mhz: float) -> np.ndarray:
    """Free-space path loss, with distance clamped to avoid singularities."""
    distance_km = np.maximum(np.asarray(distance_m), 1.0) / 1000.0
    return 32.44 + 20.0 * np.log10(distance_km) + 20.0 * np.log10(frequency_mhz)


def log_distance_path_loss_db(
    distance_m: np.ndarray | float,
    frequency_mhz: float,
    exponent: float = 2.8,
    reference_distance_m: float = 1.0,
    shadowing_db: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Indoor log-distance model anchored to FSPL at a reference distance."""
    reference_loss = fspl_db(reference_distance_m, frequency_mhz)
    distance = np.maximum(np.asarray(distance_m), reference_distance_m)
    return reference_loss + 10.0 * exponent * np.log10(distance / reference_distance_m) + shadowing_db


def hata_urban_path_loss_db(
    distance_km: np.ndarray | float,
    frequency_mhz: float,
    base_height_m: float = 30.0,
    mobile_height_m: float = 1.5,
) -> np.ndarray:
    """COST-231/Hata urban approximation for macro-cell comparisons."""
    f = float(frequency_mhz)
    d = np.maximum(np.asarray(distance_km), 0.01)
    correction = (1.1 * np.log10(f) - 0.7) * mobile_height_m - (
        1.56 * np.log10(f) - 0.8
    )
    return (
        46.3
        + 33.9 * np.log10(f)
        - 13.82 * np.log10(max(base_height_m, 1.0))
        - correction
        + (44.9 - 6.55 * np.log10(max(base_height_m, 1.0))) * np.log10(d)
    )


def _channel_assignment(cell_count: int, reuse_factor: int) -> np.ndarray:
    if reuse_factor not in (1, 3, 7):
        raise ValueError("reuse_factor must be one of 1, 3, or 7")
    return np.arange(cell_count) % reuse_factor


def compute_rx_power_matrix(
    user_positions: np.ndarray,
    cell_positions: np.ndarray,
    params: RFParameters,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return received power matrix [users, cells] and distance matrix."""
    if cell_positions.shape[0] == 0:
        raise ValueError("at least one cell is required")
    distances = np.linalg.norm(
        user_positions[:, None, :] - cell_positions[None, :, :], axis=2
    )
    rng = np.random.default_rng(seed)
    shadowing = rng.normal(0.0, params.shadowing_std_db, size=distances.shape)
    path_loss = log_distance_path_loss_db(
        distances,
        params.frequency_mhz,
        params.path_loss_exponent,
        shadowing_db=shadowing,
    )
    rx_power = params.tx_power_dbm + params.antenna_gain_dbi - path_loss
    return rx_power, distances


def simulate_network(
    user_positions: np.ndarray,
    user_demands_mbps: np.ndarray,
    cell_positions: np.ndarray,
    params: RFParameters,
    tx_power_overrides_dbm: Iterable[float] | None = None,
    channel_ids: np.ndarray | None = None,
    seed: int | None = 42,
) -> NetworkResult:
    """Associate users and calculate SINR, congestion, and throughput."""
    cell_count = cell_positions.shape[0]
    powers = np.full(cell_count, params.tx_power_dbm)
    if tx_power_overrides_dbm is not None:
        powers = np.asarray(list(tx_power_overrides_dbm), dtype=float)
        if powers.shape != (cell_count,):
            raise ValueError("tx_power_overrides_dbm must match cell count")
    effective_params = RFParameters(
        **{**params.__dict__, "tx_power_dbm": 0.0, "antenna_gain_dbi": 0.0}
    )
    base_rx, distances = compute_rx_power_matrix(
        user_positions, cell_positions, effective_params, seed=seed
    )
    rx_power_matrix = base_rx + powers[None, :] + params.antenna_gain_dbi
    if channel_ids is None:
        channel_ids = _channel_assignment(cell_count, params.reuse_factor)
    channel_ids = np.asarray(channel_ids)

    serving_cell = np.argmax(rx_power_matrix, axis=1)
    received = rx_power_matrix[np.arange(len(user_positions)), serving_cell]
    signal_mw = dbm_to_mw(received)
    all_mw = dbm_to_mw(rx_power_matrix)
    same_channel = channel_ids[None, :] == channel_ids[serving_cell][:, None]
    interference_mw = np.sum(
        np.where(same_channel, all_mw, 0.0), axis=1
    ) - signal_mw
    noise_mw = dbm_to_mw(params.noise_floor_dbm)
    sinr_db = mw_to_dbm(signal_mw / np.maximum(interference_mw + noise_mw, 1e-15))

    cell_users = np.bincount(serving_cell, minlength=cell_count)
    demand_by_cell = np.bincount(
        serving_cell, weights=user_demands_mbps, minlength=cell_count
    )
    spectral_efficiency = np.log2(1.0 + dbm_to_mw(sinr_db))
    raw_user_capacity = params.bandwidth_mhz * spectral_efficiency
    cell_capacity = np.bincount(
        serving_cell, weights=raw_user_capacity, minlength=cell_count
    )
    user_throughput = np.minimum(
        raw_user_capacity,
        np.divide(
            cell_capacity[serving_cell],
            np.maximum(cell_users[serving_cell], 1),
        ),
    )
    cell_utilization = np.maximum(
        np.divide(demand_by_cell, np.maximum(cell_capacity, 1e-6)),
        cell_users / max(params.max_users_per_cell, 1),
    )
    overloaded = cell_utilization > params.max_utilization
    dropped = (sinr_db < -3.0) | overloaded[serving_cell]
    user_throughput = np.where(dropped, 0.0, user_throughput)
    return NetworkResult(
        serving_cell=serving_cell,
        received_power_dbm=received,
        sinr_db=sinr_db,
        user_throughput_mbps=user_throughput,
        cell_throughput_mbps=np.bincount(
            serving_cell, weights=user_throughput, minlength=cell_count
        ),
        cell_users=cell_users,
        cell_utilization=cell_utilization,
        dropped=dropped,
        overloaded=overloaded,
        channel_ids=channel_ids,
        distance_m=distances[np.arange(len(user_positions)), serving_cell],
        rx_power_matrix_dbm=rx_power_matrix,
    )
