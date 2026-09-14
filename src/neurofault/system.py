"""Orchestrator: wires devices -> crossbar patching -> health monitoring ->
fault injection -> mitigation dispatch into one fault-tolerant model.

Builds exactly one CrossbarHandle, one HealthMonitor, and one redundancy pool
per memristive crossbar found after patching - and every downstream module
(faults, mitigation) is given that same handle, never a second copy. This is
the structural fix for the original codebase's bugs #1, #3 and #6 (see
docs/planning/project-setup-plan.md): there is no code path anywhere in this
package that constructs a second Crossbar object for a layer that already
has one, and Conv2d layers are patched by default alongside Linear.
"""

from __future__ import annotations

import logging

import torch

from neurofault.config import ExperimentConfig
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.crossbar.self_healing import (
    RowColumnRedundancyPool,
    build_redundancy_pool,
)
from neurofault.devices.registry import build_device
from neurofault.faults.injection import inject_faults
from neurofault.mitigation.dispatch import MitigationOutcome, apply_runtime_mitigation
from neurofault.mitigation.reset import apply_layer_reset

logger = logging.getLogger(__name__)


def _backend_module(simulator: str):
    """Lazily import the crossbar backend module for the given simulator
    name, so system.py never depends on a backend that isn't in use."""
    if simulator == "crosssim":
        from neurofault.crossbar.backends import crosssim_backend

        return crosssim_backend
    if simulator == "aihwkit":
        from neurofault.crossbar.backends import aihwkit_backend

        return aihwkit_backend
    raise ValueError(f"Unknown simulator backend: {simulator!r}")


class FaultTolerantNeuromorphic:
    def __init__(self, base_model: torch.nn.Module, config: ExperimentConfig):
        self.config = config
        backend = _backend_module(config.simulator)
        self._backend = backend
        self.device_cls, self.device_params = build_device(config.device, config.simulator)

        self.model = backend.patch_layers(
            base_model, config.crossbar, self.device_cls, self.device_params
        )

        self.handles = backend.build_handles(self.model, self.device_params, config.crossbar)

        # Not every backend has an equivalent weight->conductance mapping for
        # re-deriving conductances from weights - layer_reset degrades to
        # "unavailable" (never a crash) for backends that don't define this.
        mapping_routine_for = getattr(backend, "mapping_routine_for", None)
        self._mapping_routine = (
            mapping_routine_for(config.crossbar.scheme) if mapping_routine_for else None
        )
        self.monitors = {name: HealthMonitor(h) for name, h in self.handles.items()}
        self.pools = {
            name: build_redundancy_pool(h, config.crossbar) for name, h in self.handles.items()
        }

        # Original weights per patched layer, captured once, for layer_reset.
        # Membership must be checked against self.handles (built above by the
        # backend, so it already names exactly the patched layers) - a prior
        # version of this check looked for a `.crossbars` attribute that no
        # patched layer in either backend (AnalogLinear/AnalogConv2d for
        # CrossSim, TorchSimulatorTile for AIHWKit) ever actually defines,
        # which silently left _layer_handles empty and layer_reset dead on
        # both backends regardless of config - a real bug found and fixed
        # this session, confirmed via a direct runtime probe on both.
        patched_layers = {h.layer for h in self.handles.values()}
        self._original_weights: dict[str, torch.Tensor] = {}
        self._layer_handles: dict[str, list] = {}
        for name, module in self.model.named_modules():
            if module in patched_layers and getattr(module, "weight", None) is not None:
                self._original_weights[name] = module.weight.data.clone()
                self._layer_handles[name] = [
                    h for hname, h in self.handles.items() if h.layer is module
                ]

        self.total_operations = 0
        self.total_faults_injected = 0
        self.stats = {"soft_mitigations": 0, "remappings": 0, "layer_resets": 0}

        # Conductance values that must be reasserted after every synchronize()
        # (see synchronize()/_reapply_pins() below) because they don't durably
        # survive being re-derived from .weight - a real bug found and fixed
        # this session: this used to be a fault-aware-retraining-only
        # mechanism (_capture_fixed_defects()/reapply_stuck_faults()), but
        # stuck-at faults silently self-healed within one batch in every
        # OTHER mode too, verified directly (86/86 stuck cells reverted after
        # one real optimizer.step()+synchronize()). Two categories:
        # - _stuck_pin_values: permanent, updated incrementally by
        #   inject_faults() below, for every mode.
        # - _pending_one_shot_pins: soft-mitigation's corrective nudge only -
        #   models drift correction, not a permanent structural change, so it
        #   survives exactly the one synchronize() that would otherwise erase
        #   it before any forward pass ever benefits from it, then releases
        #   so ordinary training resumes shaping that cell from the corrected
        #   baseline. Remap needs no equivalent entry here: its corrected
        #   positions are already tracked permanently by the pool itself
        #   (RedundancyPool.remapping / RowColumnRedundancyPool's
        #   condemned_rows/condemned_cols), reapplied directly from there.
        self._stuck_pin_values: dict[str, torch.Tensor] = {}
        self._pending_one_shot_pins: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # layer_reset cooldown bookkeeping - see MitigationConfig.
        # layer_reset_cooldown_checks and _attempt_layer_reset()'s docstring.
        self._health_check_count = 0
        self._last_reset_check: dict[str, int] = {}

        logger.info(
            "Initialized FaultTolerantNeuromorphic with %d memristive crossbar(s)",
            len(self.handles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(self.model.training):
            output = self.model(x)

        with torch.no_grad():
            self.total_operations += 1
            if self.total_operations % self.config.mitigation.health_check_interval == 0:
                self.check_health()

        return output

    def synchronize(self) -> None:
        """Call after every optimizer.step() during training. Some backends
        (CrossSim - see crosssim_backend.py's module docstring) don't
        auto-propagate gradient updates into the analog compute path without
        this; others (AIHWKit) already do and expose a no-op here. Missing
        this call means training silently never reaches the crossbar state
        for backends that need it - a real bug found and fixed this session,
        not a defensive no-op added speculatively.

        Always followed by _reapply_pins(): synchronize_fn() re-derives every
        cell's conductance from .weight, which knows nothing about stuck-at
        faults or mitigation corrections - a real bug found and fixed this
        session (verified directly: 86/86 stuck cells and a 122-cell remap
        both got silently erased by exactly this call). This runs for every
        mode now, not just fault-aware retraining - see _stuck_pin_values'
        docstring in __init__.
        """
        synchronize_fn = getattr(self._backend, "synchronize", None)
        if synchronize_fn is not None:
            synchronize_fn(self.model)
        self._reapply_pins()

    def _reapply_pins(self) -> None:
        for name, handle in self.handles.items():
            monitor = self.monitors[name]
            pool = self.pools[name]
            changed = False
            matrix = handle.accessor.read()

            stuck_values = self._stuck_pin_values.get(name)
            if stuck_values is not None and monitor.stuck_mask.any():
                matrix[monitor.stuck_mask] = stuck_values[monitor.stuck_mask]
                changed = True

            if isinstance(pool, RowColumnRedundancyPool):
                for row in pool.condemned_rows:
                    matrix[row, :] = pool.reference_conductance[row, :]
                    changed = True
                for col in pool.condemned_cols:
                    matrix[:, col] = pool.reference_conductance[:, col]
                    changed = True
            else:
                for row, col in pool.remapping:
                    matrix[row, col] = pool.reference_conductance[row, col]
                    changed = True

            pending = self._pending_one_shot_pins.pop(name, None)
            if pending is not None:
                rows, cols, values = pending
                matrix[rows, cols] = values
                changed = True

            if changed:
                handle.accessor.write(matrix)

    def inject_faults(self) -> int:
        total = 0
        for name, handle in self.handles.items():
            total += inject_faults(handle, self.monitors[name], self.config.fault)
            self._update_stuck_pins(name, handle)
        self.total_faults_injected += total
        return total

    def _update_stuck_pins(self, name: str, handle) -> None:
        """Snapshot the just-written conductance value of every currently-
        stuck cell, so _reapply_pins() can re-clamp it after every future
        synchronize(). Called after every inject_faults() call in every
        mode (not just fault-aware retraining's one-time injection) -
        incremental and idempotent, since a stuck cell's value doesn't change
        between injection and this capture."""
        monitor = self.monitors[name]
        if not monitor.stuck_mask.any():
            return
        current = handle.accessor.read()
        pinned = self._stuck_pin_values.setdefault(name, torch.zeros_like(current))
        pinned[monitor.stuck_mask] = current[monitor.stuck_mask]

    def check_health(self) -> dict[str, MitigationOutcome]:
        outcomes = {}
        mitigation_config = self.config.mitigation
        self._health_check_count += 1

        for name, handle in self.handles.items():
            monitor = self.monitors[name]
            monitor.update_health_metrics()

            # Perf fix, verified safe (see docs/planning/project-setup-plan.md):
            # every mitigation action function (apply_soft_mitigation, apply_remap,
            # apply_layer_reset's positions=[] path) already no-ops on an empty
            # position list, so nothing below this check can have any observable
            # effect when a layer has zero verified faults - regardless of what
            # get_critical_devices()'s heuristic (health score / failure
            # probability / extreme conductance) flags. That heuristic set can be
            # enormous even fault-free: probed directly on a widened fc1 layer
            # (768x3072 = 2.36M cells), 261,889-337,546 cells (11-14%) were
            # flagged "critical" with zero faults injected anywhere, and
            # everything below (the stuck_positions filter, get_health_summary,
            # apply_runtime_mitigation) used to run in full against that list
            # every health_check_interval batches regardless - confirmed via
            # cProfile as check_health's dominant cost, growing with model size.
            if not monitor.stuck_mask.any() and not monitor.drift_mask.any():
                continue
            # Verified this actually matters, not just theoretically: on the
            # widened SimpleCNN at fault density 0.0, clean per-batch timing
            # (no cProfile overhead) went from 601.6ms/batch to 181.7ms/batch
            # after this guard - a 3.3x speedup, all from avoiding this exact
            # dead-end path every health_check_interval batches.

            critical = monitor.get_critical_devices(
                detection_method=mitigation_config.detection_method,
                checksum_z_threshold=mitigation_config.checksum_z_threshold,
            )
            if not critical:
                continue

            summary = monitor.get_health_summary()

            layer_name = self._find_layer_name(handle)
            reset_fn = None
            if layer_name is not None and self._mapping_routine is not None:
                # Precision fix found this session (see docs/planning/
                # "Catching Up" plan): `critical` is get_critical_devices()'s
                # broad heuristic set (health score / failure probability /
                # extreme conductance), which can include cells that are
                # simply legitimately-trained near an extreme value, not
                # actually faulty. layer_reset should only ever touch cells
                # confirmed permanently, physically dead - filter to
                # stuck_mask here, same fix applied to remapping in
                # dispatch.py.
                stuck_positions = [(r, c) for r, c in critical if monitor.stuck_mask[r, c]]
                reset_fn = lambda ln=layer_name, positions=stuck_positions: (
                    self._attempt_layer_reset(ln, positions)
                )
            elif layer_name is not None:
                logger.debug(
                    "Layer reset unavailable for simulator=%s (no mapping_routine_for); "
                    "skipping reset for %s",
                    self.config.simulator,
                    layer_name,
                )

            outcome = apply_runtime_mitigation(
                handle,
                self.pools[name],
                monitor,
                critical,
                summary["avg_health"],
                mitigation_config,
                reset_fn=reset_fn,
            )
            outcomes[name] = outcome
            self.stats["soft_mitigations"] += outcome.soft_mitigated
            self.stats["remappings"] += outcome.remapped
            self.stats["layer_resets"] += int(outcome.layer_reset)

            self._pin_soft_mitigation(name, handle, outcome)

        return outcomes

    def _attempt_layer_reset(self, layer_name: str, positions: list[tuple[int, int]]) -> bool:
        """Gated per mitigation.layer_reset_cooldown_checks (see
        MitigationConfig's docstring for the real problem this closes):
        returns False without resetting anything if this layer was reset
        too recently, measured in check_health() calls, not batches - so it
        composes with health_check_interval instead of a separate knob.

        positions (the calling handle's critical devices) targets the reset
        at only those cells - see apply_layer_reset's docstring for why a
        whole-layer reset is a real, measured design flaw, not just a
        theoretical one."""
        last = self._last_reset_check.get(layer_name)
        cooldown = self.config.mitigation.layer_reset_cooldown_checks
        if last is not None and self._health_check_count - last < cooldown:
            return False

        ok = apply_layer_reset(
            self._layer_handles[layer_name][0].layer,
            self._layer_handles[layer_name],
            self._original_weights[layer_name],
            self._mapping_routine,
            self.device_params.get("r_on", 100.0),
            self.device_params.get("r_off", 10000.0),
            scheme=self.config.crossbar.scheme,
            positions=positions,
        )
        if ok:
            self._last_reset_check[layer_name] = self._health_check_count
        return ok

    def _pin_soft_mitigation(self, name: str, handle, outcome: MitigationOutcome) -> None:
        """Pin any soft-mitigated positions through exactly the next
        synchronize() - see its docstring - then release so ordinary
        training resumes shaping those cells from the corrected baseline."""
        if not outcome.soft_mitigated_positions:
            return
        rows = torch.tensor([r for r, _c in outcome.soft_mitigated_positions])
        cols = torch.tensor([c for _r, c in outcome.soft_mitigated_positions])
        values = handle.accessor.read()[rows, cols].clone()
        self._pending_one_shot_pins[name] = (rows, cols, values)

    def get_health_report(self) -> dict:
        summaries = {name: m.get_health_summary() for name, m in self.monitors.items()}
        avg_health = (
            sum(s["avg_health"] for s in summaries.values()) / len(summaries)
            if summaries
            else 100.0
        )
        return {
            "total_operations": self.total_operations,
            "total_faults_injected": self.total_faults_injected,
            **self.stats,
            "avg_health": avg_health,
            "layer_health": summaries,
        }

    def _find_layer_name(self, handle) -> str | None:
        for layer_name, handles in self._layer_handles.items():
            if handle in handles:
                return layer_name
        return None
