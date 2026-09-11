"""Orchestrator: wires devices -> crossbar patching -> health monitoring ->
fault injection -> mitigation dispatch into one fault-tolerant model.

Builds exactly one CrossbarHandle, one HealthMonitor, and one RedundancyPool
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
from neurofault.crossbar.self_healing import RedundancyPool
from neurofault.devices.registry import build_device
from neurofault.faults.injection import inject_faults
from neurofault.mitigation.dispatch import MitigationOutcome, apply_runtime_mitigation
from neurofault.mitigation.reset import apply_layer_reset

logger = logging.getLogger(__name__)


def _backend_module(simulator: str):
    """Lazily import the crossbar backend module for the given simulator
    name, so system.py never depends on a backend that isn't in use."""
    if simulator == "memtorch":
        from neurofault.crossbar.backends import memtorch_backend

        return memtorch_backend
    if simulator == "xbtorch":
        from neurofault.crossbar.backends import xbtorch_backend

        return xbtorch_backend
    if simulator == "aihwkit":
        from neurofault.crossbar.backends import aihwkit_backend

        return aihwkit_backend
    raise ValueError(f"Unknown simulator backend: {simulator!r}")


class FaultTolerantNeuromorphic:
    def __init__(self, base_model: torch.nn.Module, config: ExperimentConfig):
        self.config = config
        backend = _backend_module(config.simulator)
        self.device_cls, self.device_params = build_device(config.device, config.simulator)

        self.model = backend.patch_layers(
            base_model, config.crossbar, self.device_cls, self.device_params
        )

        self.handles = backend.build_handles(self.model, self.device_params, config.crossbar)

        # Not every backend has an equivalent to memtorch's naive_map for
        # re-deriving conductances from weights - layer_reset degrades to
        # "unavailable" (never a crash) for backends that don't define this.
        mapping_routine_for = getattr(backend, "mapping_routine_for", None)
        self._mapping_routine = (
            mapping_routine_for(config.crossbar.scheme) if mapping_routine_for else None
        )
        self.monitors = {name: HealthMonitor(h) for name, h in self.handles.items()}
        self.pools = {
            name: RedundancyPool(h, config.crossbar.redundancy_factor)
            for name, h in self.handles.items()
        }

        # Original weights per patched layer, captured once, for layer_reset.
        self._original_weights: dict[str, torch.Tensor] = {}
        self._layer_handles: dict[str, list] = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "crossbars") and hasattr(module, "weight"):
                self._original_weights[name] = module.weight.data.clone()
                self._layer_handles[name] = [
                    h for hname, h in self.handles.items() if h.layer is module
                ]

        self.total_operations = 0
        self.total_faults_injected = 0
        self.stats = {"soft_mitigations": 0, "remappings": 0, "layer_resets": 0}

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

    def inject_faults(self) -> int:
        total = 0
        for name, handle in self.handles.items():
            total += inject_faults(handle, self.monitors[name], self.config.fault)
        self.total_faults_injected += total
        return total

    def check_health(self) -> dict[str, MitigationOutcome]:
        outcomes = {}
        mitigation_config = self.config.mitigation

        for name, handle in self.handles.items():
            monitor = self.monitors[name]
            monitor.update_health_metrics()
            critical = monitor.get_critical_devices()
            if not critical:
                continue

            summary = monitor.get_health_summary()

            layer_name = self._find_layer_name(handle)
            reset_fn = None
            if layer_name is not None and self._mapping_routine is not None:
                reset_fn = lambda ln=layer_name, h=handle: apply_layer_reset(
                    h.layer,
                    self._layer_handles[ln],
                    self._original_weights[ln],
                    self._mapping_routine,
                    self.device_params.get("r_on", 100.0),
                    self.device_params.get("r_off", 10000.0),
                    scheme=self.config.crossbar.scheme,
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

        return outcomes

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
