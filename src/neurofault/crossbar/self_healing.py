"""Redundancy bookkeeping for one crossbar.

Unlike the original SelfHealingCrossbar, this NEVER constructs its own
memtorch.bh.crossbar.Crossbar object. It is pure bookkeeping over the shape
of the CrossbarHandle it's given - there is no second, disconnected physical
array to keep in sync, which is what caused the original codebase's bugs #1
(remapping never affected the real forward pass) and #3 (fault detection
comparing against the wrong object).

This models an idealized, software-only upper bound on redundancy - "restore
a faulty cell to its pre-fault reference value" - not physical spare
row/column hardware. That's an explicit, documented scope decision (see
docs/planning/project-setup-plan.md), not an oversight: real row/column
redundancy would require padding layer dimensions before patch_model runs,
which is deferred to a later phase.
"""

from __future__ import annotations

import logging

from neurofault.crossbar.array import CrossbarHandle

logger = logging.getLogger(__name__)


class RedundancyPool:
    def __init__(self, handle: CrossbarHandle, redundancy_factor: float):
        self.handle = handle
        self.redundancy_factor = redundancy_factor
        shape = handle.crossbar.conductance_matrix.shape
        total_cells = shape[0] * shape[1]

        self.capacity = max(1, int(total_cells * redundancy_factor))
        self.remapping: dict[tuple[int, int], tuple[int, int]] = {}
        # Reference conductance captured once, before any faults - the "known good"
        # value a remap restores. Sourced from the SAME crossbar object, never a copy
        # kept anywhere else.
        self.reference_conductance = handle.crossbar.conductance_matrix.clone().detach()

        logger.info(
            "RedundancyPool for %s: capacity=%d (idealized_per_cell, %.0f%% of %d cells)",
            handle.name,
            self.capacity,
            redundancy_factor * 100,
            total_cells,
        )

    @property
    def available_capacity(self) -> int:
        return self.capacity - len(self.remapping)

    def remap_device(self, row: int, col: int) -> bool:
        """Restore one faulty cell to its pre-fault reference conductance,
        writing directly into the canonical crossbar object (the same one
        the forward pass and health monitor read) - never a separate array.
        """
        pos = (row, col)
        if pos in self.remapping:
            return False
        if self.available_capacity <= 0:
            return False

        self.handle.crossbar.conductance_matrix[row, col] = self.reference_conductance[row, col]
        self.remapping[pos] = pos  # kept for reporting; see module docstring
        return True
