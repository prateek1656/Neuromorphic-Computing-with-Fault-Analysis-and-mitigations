"""Redundancy remapping: restore faulty devices to their pre-fault reference
conductance, via a RedundancyPool's bookkeeping and the canonical
ConductanceAccessor for the actual I/O.

This is the direct fix for the original codebase's bug #1: the write goes
through `handle.accessor` - the same interface health monitoring and fault
injection use for this exact crossbar - never a separate, disconnected
crossbar object.

apply_remap() is the single public entry point regardless of which
redundancy scheme built `pool` - neurofault.mitigation.dispatch.py calls it
generically and never needs to know which scheme is active; the branch on
pool type lives here instead. See neurofault.crossbar.self_healing's module
docstring for what each scheme actually models.
"""

from __future__ import annotations

import logging

from neurofault.crossbar.array import CrossbarHandle
from neurofault.crossbar.self_healing import RedundancyPool, RowColumnRedundancyPool

logger = logging.getLogger(__name__)


def apply_remap(
    handle: CrossbarHandle,
    pool: RedundancyPool | RowColumnRedundancyPool,
    critical_devices: list[tuple[int, int]],
) -> int:
    if isinstance(pool, RowColumnRedundancyPool):
        return _apply_row_col_remap(handle, pool, critical_devices)
    return _apply_per_cell_remap(handle, pool, critical_devices)


def _apply_per_cell_remap(
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


def _apply_row_col_remap(
    handle: CrossbarHandle,
    pool: RowColumnRedundancyPool,
    critical_devices: list[tuple[int, int]],
) -> int:
    """Restores whole condemned rows/columns to reference_conductance in one
    write. Returns len(rows)+len(cols) - the honest unit for this scheme is
    rows/columns repaired, not incidental cell count (a condemned row/column
    touches many cells at once, so a cell-count return would misrepresent
    what "one remapping" means here vs. the per-cell scheme).
    """
    new_rows, new_cols = pool.condemn_rows_and_cols(critical_devices)
    if not new_rows and not new_cols:
        return 0

    matrix = handle.accessor.read()
    for row in new_rows:
        matrix[row, :] = pool.reference_conductance[row, :]
    for col in new_cols:
        matrix[:, col] = pool.reference_conductance[:, col]
    handle.accessor.write(matrix)

    logger.info(
        "Condemned %d row(s), %d column(s) in %s (%d row / %d col capacity remaining)",
        len(new_rows),
        len(new_cols),
        handle.name,
        pool.available_row_capacity,
        pool.available_col_capacity,
    )
    return len(new_rows) + len(new_cols)
