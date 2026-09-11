"""Fault injection for a single crossbar.

Tries memtorch's native memtorch.bh.nonideality primitives first (the only
evidence we have for their call shape is an import that appears, but is
never actually called, in the project's one proven-working notebook run -
so this path is genuinely unverified and must degrade cleanly, never raise
out to the caller). Falls back to a vectorized custom stuck-at
implementation, which is a direct rewrite of the original
SelfHealingCrossbar.inject_faults with its two Python for-loops replaced by
tensor indexing - this is the path the Phase 1 smoke test actually exercises
and must work.
"""

from __future__ import annotations

import logging

import torch

from neurofault.config import FaultConfig
from neurofault.crossbar.array import CrossbarHandle
from neurofault.crossbar.health_monitor import HealthMonitor

logger = logging.getLogger(__name__)


def inject_faults(handle: CrossbarHandle, monitor: HealthMonitor, fault_config: FaultConfig) -> int:
    if fault_config.backend in ("auto", "native"):
        result = _try_native(handle, fault_config)
        if result is not None:
            return result
        if fault_config.backend == "native":
            raise RuntimeError("fault_config.backend='native' but the native path is unavailable")
        logger.warning(
            "memtorch.bh.nonideality path unavailable, falling back to custom stuck-at injection"
        )

    return _custom_stuck_at(handle, monitor, fault_config)


def _try_native(handle: CrossbarHandle, fault_config: FaultConfig) -> int | None:
    try:
        from memtorch.bh.nonideality.NonIdeality import apply_nonidealities  # noqa: F401
    except ImportError:
        return None

    # The exact call signature of apply_nonidealities is unverified in this
    # environment (memtorch's C++ extension doesn't build without a CUDA
    # driver present - see docs/planning/project-setup-plan.md). Deliberately
    # not guessing at a call here: return None so the caller falls back to
    # the verified custom path, rather than risk a confidently-wrong call.
    logger.info(
        "memtorch.bh.nonideality is importable but its call shape is unverified; skipping native path"
    )
    return None


def _custom_stuck_at(
    handle: CrossbarHandle, monitor: HealthMonitor, fault_config: FaultConfig
) -> int:
    matrix = handle.crossbar.conductance_matrix
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
