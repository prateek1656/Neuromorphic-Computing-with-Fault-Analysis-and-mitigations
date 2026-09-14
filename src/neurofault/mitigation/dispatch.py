"""Central dispatch for which mitigation strategy runs on a set of critical
devices.

Fixes the original codebase's bug #2: `_apply_runtime_fault_mitigation`'s
final `else` branch called `reset_critical_layer` UNCONDITIONALLY, with no
`enable_layer_reset` check - meaning a "No Mitigations" or "Soft Mitigation
Only" experiment config could still silently trigger a full layer reset.

Every branch here is guarded by its own `mitigation_config.enable_*` flag
with no exception - if a branch's condition is met but its flag is off,
nothing happens at all, full stop. There is no branch that calls a
mitigation function unconditionally.

Also fixes bug #4: soft mitigation is reachable directly from this runtime
path (not only from a separate predictive path gated on history that may
never accumulate).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from neurofault.config import MitigationConfig
from neurofault.crossbar.array import CrossbarHandle
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.crossbar.self_healing import RedundancyPool, RowColumnRedundancyPool
from neurofault.mitigation.remap import apply_remap
from neurofault.mitigation.soft import apply_soft_mitigation

logger = logging.getLogger(__name__)


@dataclass
class MitigationOutcome:
    soft_mitigated: int = 0
    remapped: int = 0
    layer_reset: bool = False
    actions_taken: list[str] = field(default_factory=list)
    # Exact (row, col) positions soft-mitigated this call - system.py needs these to
    # pin the correction through the next synchronize() (see its module docstring):
    # a conductance-only write here would otherwise be silently erased before any
    # forward pass ever benefits from it. Remap doesn't need an equivalent: its
    # corrected positions are already tracked permanently by the pool itself
    # (RedundancyPool.remapping / RowColumnRedundancyPool.condemned_rows/cols).
    soft_mitigated_positions: list[tuple[int, int]] = field(default_factory=list)


def apply_runtime_mitigation(
    handle: CrossbarHandle,
    pool: RedundancyPool | RowColumnRedundancyPool,
    monitor: HealthMonitor,
    critical_devices: list[tuple[int, int]],
    avg_health: float,
    mitigation_config: MitigationConfig,
    reset_fn=None,  # Callable[[], bool] | None - injected so dispatch doesn't need
    #                 to know layer-reset's full signature; None means "not available"
) -> MitigationOutcome:
    outcome = MitigationOutcome()
    if not critical_devices:
        return outcome

    threshold = mitigation_config.health_threshold

    # Real precision fix found this session (see docs/planning/ "Catching Up"
    # plan): get_critical_devices() flags cells via a broad heuristic
    # (health score, failure probability, OR extreme conductance) that can
    # include cells that are simply legitimately-trained near an extreme
    # value, not actually faulty - remapping/layer_reset should only ever
    # touch cells confirmed permanently, physically dead (stuck_mask), never
    # a merely-heuristically-flagged one. This is the mechanistic explanation
    # this session found for why layer_reset kept hurting even once made
    # "targeted": it was still targeting the broad heuristic set, not
    # verified failures.
    stuck_capable = [(row, col) for row, col in critical_devices if monitor.stuck_mask[row, col]]

    if mitigation_config.force_remapping and mitigation_config.enable_remapping:
        outcome.remapped = apply_remap(handle, pool, stuck_capable)
        outcome.actions_taken.append("forced_remap")
        return outcome

    if avg_health > threshold:
        if mitigation_config.enable_soft_mitigation:
            # Real bug found and fixed this session: soft mitigation models
            # correcting drift, not un-sticking a hardware-stuck-at device
            # (see soft.py's own docstring) - but critical_devices here is
            # unfiltered, and a stuck cell trivially satisfies
            # get_critical_devices()'s extreme-conductance check. Verified
            # directly: of 73 critical devices at fault density 0.2, 64 were
            # stuck_mask=True, and all 64 had their conductance changed by
            # "soft mitigation" - a physically impossible repair.
            #
            # Tightened further this session: filtering on "not stuck" still
            # let through cells that are merely legitimately-trained near an
            # extreme conductance value (get_critical_devices()'s heuristic
            # can't tell that apart from real drift) - soft_mitigation would
            # then "correct" (i.e. damage) perfectly good learned weights.
            # Filter on monitor.drift_mask instead - only cells the fault
            # injector has actually, verifiably drifted (failure_model=
            # "drift", see injection.py) are drift_capable now.
            drift_capable = [
                (row, col) for row, col in critical_devices if monitor.drift_mask[row, col]
            ]
            outcome.soft_mitigated = apply_soft_mitigation(handle, monitor, drift_capable)
            outcome.soft_mitigated_positions = drift_capable
            outcome.actions_taken.append("soft_mitigation")
        if mitigation_config.enable_remapping:
            outcome.remapped = apply_remap(handle, pool, stuck_capable)
            outcome.actions_taken.append("remap")
        return outcome

    if avg_health > threshold / 2:
        if mitigation_config.enable_layer_reset and reset_fn is not None:
            outcome.layer_reset = reset_fn()
            outcome.actions_taken.append("layer_reset")
        # If layer reset is disabled here, nothing happens - no fallthrough.
        return outcome

    # Critical health: layer reset is the last resort, but ONLY if enabled.
    # The original codebase called reset_critical_layer here with no flag
    # check at all - that unconditional call is exactly what this dispatch
    # structure makes impossible.
    if mitigation_config.enable_layer_reset and reset_fn is not None:
        outcome.layer_reset = reset_fn()
        outcome.actions_taken.append("layer_reset")

    if not outcome.actions_taken:
        logger.debug(
            "%d critical devices in %s but no mitigation enabled for current health (%.1f%%)",
            len(critical_devices),
            handle.name,
            avg_health,
        )

    return outcome
