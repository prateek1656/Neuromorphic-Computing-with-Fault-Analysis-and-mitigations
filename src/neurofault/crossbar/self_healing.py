"""Redundancy bookkeeping for one crossbar.

Pure bookkeeping only - no conductance I/O happens here. This never touches
a second, disconnected crossbar (the original codebase's bug #1); it only
tracks which cells have been "remapped" and how much capacity remains. The
actual read/restore-write against the canonical CrossbarHandle happens in
neurofault.mitigation.remap, via the ConductanceAccessor - so this class
works identically regardless of which backend (memtorch/xbtorch/aihwkit)
that accessor wraps.

This models an idealized, software-only upper bound on redundancy - "restore
a faulty cell to its pre-fault reference value" - not physical spare
row/column hardware. That's an explicit, documented scope decision (see
docs/planning/project-setup-plan.md), not an oversight: real row/column
redundancy would require padding layer dimensions before the model is
patched, which is deferred to a later phase.
"""

from __future__ import annotations

import logging

from neurofault.crossbar.array import CrossbarHandle

logger = logging.getLogger(__name__)


class RedundancyPool:
    def __init__(self, handle: CrossbarHandle, redundancy_factor: float):
        self.handle = handle
        self.redundancy_factor = redundancy_factor
        rows, cols = handle.accessor.shape
        total_cells = rows * cols

        self.capacity = max(1, int(total_cells * redundancy_factor))
        self.remapping: dict[tuple[int, int], tuple[int, int]] = {}
        # Reference conductance captured once, before any faults - the "known good"
        # value a remap restores. Read once via the accessor at construction time;
        # never re-derived from a second crossbar object.
        self.reference_conductance = handle.accessor.read().clone().detach()

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
        """Record that (row, col) has been remapped, if capacity allows.

        Pure bookkeeping - does not touch the crossbar. Callers (see
        neurofault.mitigation.remap.apply_remap) are responsible for actually
        restoring the conductance value via the accessor once this returns True.
        """
        pos = (row, col)
        if pos in self.remapping:
            return False
        if self.available_capacity <= 0:
            return False

        self.remapping[pos] = pos  # kept for reporting; see module docstring
        return True
