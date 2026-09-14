"""Fault injection for a single crossbar.

Four fault models, selected via FaultConfig.failure_model:

- "density" (default): vectorized custom stuck-at, a direct rewrite of the
  original SelfHealingCrossbar.inject_faults with its two Python for-loops
  replaced by tensor indexing - this is the path the Phase 1 smoke test
  actually exercises and must work. Memoryless and purely spatial: picks
  density * total_cells cells fresh each call via `distribution`.
- "weibull" / "exponential": real time-to-failure models (see
  faults/reliability.py) - each cell's own accumulated simulated cycles
  (HealthMonitor.cycles) determines its own conditional failure probability
  this interval, optionally accelerated by Arrhenius temperature scaling.
  Deliberately orthogonal to the spatial `distribution` option - see
  docs/planning/project-setup-plan.md for why they aren't combined here.
- "drift": a real, distinct failure mode found missing this session (see
  the "Catching Up" plan and config.py's FaultConfig docstring) - gradual,
  cumulative conductance shift toward one rail, never marked stuck_mask
  (the cell stays functional, just imprecise) - the failure mode
  mitigation/soft.py's own docstring says it corrects, which this project
  never actually generated before now. Kept as an independently-selectable
  failure_model, not force-combined with stuck-at, so each mitigation
  strategy can first be evaluated against the fault type it targets.

All paths are fully vectorized - no Python per-cell loops, matching this
project's standing discipline (health_monitor.py's own docstring records a
real run killed by exactly that kind of loop).
"""

from __future__ import annotations

import logging

import torch

from neurofault.config import FaultConfig
from neurofault.crossbar.array import CrossbarHandle
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.faults.reliability import (
    arrhenius_acceleration_factor,
    exponential_conditional_failure_probability,
    weibull_conditional_failure_probability,
)

logger = logging.getLogger(__name__)


def inject_faults(handle: CrossbarHandle, monitor: HealthMonitor, fault_config: FaultConfig) -> int:
    if fault_config.failure_model == "density":
        return _custom_stuck_at(handle, monitor, fault_config)
    if fault_config.failure_model == "drift":
        return _drift(handle, monitor, fault_config)
    return _statistical_stuck_at(handle, monitor, fault_config)


def _custom_stuck_at(
    handle: CrossbarHandle, monitor: HealthMonitor, fault_config: FaultConfig
) -> int:
    matrix = handle.accessor.read()
    rows, cols = matrix.shape
    total_devices = rows * cols
    num_faults = int(total_devices * fault_config.density)
    if num_faults == 0:
        return 0

    flat_indices = _select_fault_indices(rows, cols, num_faults, fault_config.distribution)
    fault_rows = flat_indices // cols
    fault_cols = flat_indices % cols

    half = len(flat_indices) // 2
    lrs_rows, lrs_cols = fault_rows[:half], fault_cols[:half]  # stuck at HRS (low conductance)
    hrs_rows, hrs_cols = fault_rows[half:], fault_cols[half:]  # stuck at LRS (high conductance)

    matrix[lrs_rows, lrs_cols] = monitor.g_min
    matrix[hrs_rows, hrs_cols] = monitor.g_max
    handle.accessor.write(matrix)

    monitor.stuck_mask[fault_rows, fault_cols] = True
    monitor.health_scores[fault_rows, fault_cols] = 0
    monitor.stability_index[fault_rows, fault_cols] = 0.1
    monitor.stress[fault_rows, fault_cols] = 0.9

    logger.info(
        "Injected %d faults into %s (%s distribution)",
        len(flat_indices),
        handle.name,
        fault_config.distribution,
    )
    return len(flat_indices)


def _drift(handle: CrossbarHandle, monitor: HealthMonitor, fault_config: FaultConfig) -> int:
    """A real, distinct failure mode found missing this session (see
    config.py's FaultConfig.drift_magnitude/drift_direction docstring):
    cells shift gradually toward one conductance rail instead of being
    pinned there outright - never marked stuck_mask, since the cell stays
    writable/functional, just imprecise. This is the failure mode
    mitigation/soft.py's docstring says it corrects, which this project
    never actually generated before now.
    """
    matrix = handle.accessor.read()
    rows, cols = matrix.shape
    total_devices = rows * cols
    num_drifting = int(total_devices * fault_config.density)
    if num_drifting == 0:
        return 0

    flat_indices = _select_fault_indices(rows, cols, num_drifting, fault_config.distribution)
    drift_rows = flat_indices // cols
    drift_cols = flat_indices % cols

    shift_magnitude = fault_config.drift_magnitude * monitor.g_range
    if fault_config.drift_direction == "toward_hrs":
        shift = torch.full_like(drift_rows, -1.0, dtype=torch.float32) * shift_magnitude
    elif fault_config.drift_direction == "toward_lrs":
        shift = torch.full_like(drift_rows, 1.0, dtype=torch.float32) * shift_magnitude
    else:  # "random_per_cell" - decided once per cell, reused thereafter
        sign = monitor.drift_sign[drift_rows, drift_cols]
        undecided = sign == 0
        if undecided.any():
            new_signs = torch.where(torch.rand(int(undecided.sum())) < 0.5, -1.0, 1.0)
            sign[undecided] = new_signs
            monitor.drift_sign[drift_rows, drift_cols] = sign
        shift = sign * shift_magnitude

    matrix[drift_rows, drift_cols] = torch.clamp(
        matrix[drift_rows, drift_cols] + shift, monitor.g_min, monitor.g_max
    )
    handle.accessor.write(matrix)
    monitor.drift_mask[drift_rows, drift_cols] = True

    logger.info(
        "Drifted %d cell(s) in %s toward %s (%s distribution, magnitude=%.3f)",
        len(flat_indices),
        handle.name,
        fault_config.drift_direction,
        fault_config.distribution,
        fault_config.drift_magnitude,
    )
    return len(flat_indices)


def _statistical_stuck_at(
    handle: CrossbarHandle, monitor: HealthMonitor, fault_config: FaultConfig
) -> int:
    """Weibull/exponential time-to-failure fault injection - see module and
    faults/reliability.py docstrings. Vectorized: one torch.bernoulli() call
    over the whole matrix, no per-cell Python loop.
    """
    surviving = ~monitor.stuck_mask
    monitor.cycles[surviving] += 1

    acceleration = 1.0
    if fault_config.arrhenius_activation_energy_ev is not None:
        acceleration = arrhenius_acceleration_factor(
            fault_config.temperature_kelvin,
            fault_config.reference_temperature_kelvin,
            fault_config.arrhenius_activation_energy_ev,
        )

    delta_cycles = torch.full_like(monitor.cycles, acceleration)
    if fault_config.failure_model == "weibull":
        cycles_before = monitor.cycles - delta_cycles
        fail_prob = weibull_conditional_failure_probability(
            cycles_before, delta_cycles, fault_config.weibull_shape, fault_config.weibull_scale
        )
    elif fault_config.failure_model == "exponential":
        fail_prob = exponential_conditional_failure_probability(
            delta_cycles, fault_config.exponential_rate
        )
    else:
        raise ValueError(f"Unknown fault_config.failure_model: {fault_config.failure_model!r}")

    fail_prob = fail_prob.masked_fill(monitor.stuck_mask, 0.0)
    newly_failed = torch.bernoulli(fail_prob).bool()

    rows, cols = torch.nonzero(newly_failed, as_tuple=True)
    n_new = len(rows)
    if n_new == 0:
        return 0

    matrix = handle.accessor.read()
    # Each newly-failed cell independently stuck at one of the two extremes -
    # unlike _custom_stuck_at's fixed split by position, real stuck-at
    # physics doesn't spatially pattern which rail a device gets stuck at.
    stuck_at_min = torch.rand(n_new) < 0.5
    matrix[rows[stuck_at_min], cols[stuck_at_min]] = monitor.g_min
    matrix[rows[~stuck_at_min], cols[~stuck_at_min]] = monitor.g_max
    handle.accessor.write(matrix)

    monitor.stuck_mask[rows, cols] = True
    monitor.health_scores[rows, cols] = 0
    monitor.stability_index[rows, cols] = 0.1
    monitor.stress[rows, cols] = 0.9

    logger.info(
        "Injected %d faults into %s (%s failure model, max cycles=%d)",
        n_new,
        handle.name,
        fault_config.failure_model,
        int(monitor.cycles.max().item()),
    )
    return n_new


def _select_fault_indices(rows: int, cols: int, num_faults: int, distribution: str) -> torch.Tensor:
    total = rows * cols
    num_faults = min(num_faults, total)

    if distribution == "clustered":
        num_centers = max(1, num_faults // 5)
        centers = torch.randperm(total)[:num_centers]
        center_rows, center_cols = centers // cols, centers % cols
        offsets = torch.tensor([(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)])

        candidate_rows = (center_rows.unsqueeze(1) + offsets[:, 0].unsqueeze(0)).flatten()
        candidate_cols = (center_cols.unsqueeze(1) + offsets[:, 1].unsqueeze(0)).flatten()
        valid = (
            (candidate_rows >= 0)
            & (candidate_rows < rows)
            & (candidate_cols >= 0)
            & (candidate_cols < cols)
        )
        candidate_flat = (candidate_rows[valid] * cols + candidate_cols[valid]).unique()
        return candidate_flat[:num_faults]

    if distribution == "gradient":
        all_rows = torch.arange(rows).repeat_interleave(cols)
        all_cols = torch.arange(cols).repeat(rows)
        distances = all_rows.float() / rows + all_cols.float() / cols
        weights = torch.clamp(1 - distances / 2, min=1e-6)
        return torch.multinomial(weights, num_faults, replacement=False)

    return torch.randperm(total)[:num_faults]  # "random" (default)
