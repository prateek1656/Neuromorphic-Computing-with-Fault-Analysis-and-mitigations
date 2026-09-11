"""Redundancy remapping: restore a faulty cell's pre-fault reference
conductance, via RedundancyPool's bookkeeping and the canonical
ConductanceAccessor for the actual I/O.

This is the direct fix for the original codebase's bug #1: the write goes
through `handle.accessor` - the same interface health monitoring and fault
injection use for this exact crossbar - never a separate, disconnected
crossbar object. See neurofault.crossbar.self_healing.RedundancyPool's
module docstring for why this is documented as an idealized upper bound
rather than hardware-realistic row/column redundancy.
"""

from __future__ import annotations

import logging

from neurofault.crossbar.array import CrossbarHandle
from neurofault.crossbar.self_healing import RedundancyPool

logger = logging.getLogger(__name__)


def apply_remap(
    handle: CrossbarHandle,
    pool: RedundancyPool,
    critical_devices: list[tuple[int, int]],
) -> int:
    matrix = handle.accessor.read()
    remapped = 0

    for row, col in critical_devices:
        if not pool.remap_device(row, col):
            break  # out of redundancy capacity - stop rather than silently skip the rest
        matrix[row, col] = pool.reference_conductance[row, col]
        remapped += 1

    if remapped > 0:
        handle.accessor.write(matrix)

    logger.info(
        "Remapped %d/%d devices in %s (%d capacity remaining)",
        remapped,
        len(critical_devices),
        handle.name,
        pool.available_capacity,
    )
    return remapped
