"""Redundancy remapping: restore a faulty cell's pre-fault reference
conductance via the RedundancyPool.

This is the direct fix for the original codebase's bug #1: the write goes
through `handle.crossbar.conductance_matrix` - the exact object the forward
pass and health monitor use - never a separate, disconnected crossbar. See
neurofault.crossbar.self_healing.RedundancyPool's module docstring for why
this is documented as an idealized upper bound rather than hardware-realistic
row/column redundancy.
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
    remapped = 0
    for row, col in critical_devices:
        if pool.remap_device(row, col):
            remapped += 1
        else:
            break  # out of redundancy capacity - stop rather than silently skip the rest

    logger.info(
        "Remapped %d/%d devices in %s (%d capacity remaining)",
        remapped,
        len(critical_devices),
        handle.name,
        pool.available_capacity,
    )
    return remapped
