"""XBTorch backend: gives each patched Linear/Conv2d layer its own dedicated
XBTorch accelerator (SimpleFixedPoint), used for conductance-state tracking
(fault injection, health scoring, mitigation) via the same ConductanceAccessor
interface every other backend implements.

Deliberate scope decision (see docs/planning/project-setup-plan.md Phase 2):
XBTorch's own patch_model-equivalent, `xbtorch_model()`, requires a SINGLE
accelerator shared across every layer of a model, with each layer's weights
placed at scattered indices via a configurable mapping scheme (default:
random) - a genuinely different architecture than our per-layer
CrossbarHandle abstraction, and a bigger redesign than a backend swap.
Giving each layer its own dedicated accelerator keeps every other neurofault
module (health_monitor.py, mitigation/*.py) unchanged.

Honest cost of this choice, stated plainly rather than glossed over: this
backend does NOT call `xbtorch_model()`/`xbtorch.initialize()`, so it does
not exercise XBTorch's shared-chip resource-contention modeling, and the
live forward pass does NOT route through the accelerator's own
quantization/noise/drift math - our crossbar state correctly drives fault
injection, health scoring, and mitigation bookkeeping, but the network still
computes with the layer's own `.weight` tensor. Wiring the forward pass
through the accelerator's simulated read path is real follow-up work, not
claimed here. Because this backend never calls `xbtorch_model()`, the model
does NOT need XBTorch's `.model`-as-nn.Sequential wrapper - plain
models/cnn.py::SimpleCNN works with this backend as-is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from neurofault.config import CrossbarConfig
from neurofault.crossbar.array import CrossbarHandle

logger = logging.getLogger(__name__)

_LAYER_TYPES = {
    "Linear": torch.nn.Linear,
    "Conv2d": torch.nn.Conv2d,
}


@dataclass
class XBTorchAccessor:
    accelerator: object  # a dedicated xbtorch.deployment.SimpleFixedPoint instance

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.accelerator._chip.shape)

    def read(self) -> torch.Tensor:
        rows, cols = self.shape
        return self.accelerator.read_chip(0, rows, 0, cols)

    def write(self, matrix: torch.Tensor) -> None:
        # No public region/cell write API is exposed by XBTorch accelerators
        # (only map_weights_to_array, which re-derives conductances from a
        # software weight tensor via its own encoding scheme). Writing
        # conductance values directly requires the accelerator's own chip
        # tensor - a real, accessible instance attribute, just not part of
        # its documented public interface (confirmed by direct inspection).
        rows, cols = matrix.shape
        self.accelerator._chip[:rows, :cols] = matrix


def patch_layers(
    model: torch.nn.Module,
    crossbar_config: CrossbarConfig,
    device_cls: type,
    device_params: dict,
) -> torch.nn.Module:
    """No-op: this backend does not call xbtorch_model() (see module
    docstring). build_handles() does the actual per-layer accelerator setup."""
    return model


def build_handles(
    patched_model: torch.nn.Module, device_params: dict, crossbar_config: CrossbarConfig
) -> dict[str, CrossbarHandle]:
    from xbtorch.deployment import SimpleFixedPoint

    patch_targets = tuple(_LAYER_TYPES[t] for t in crossbar_config.patch_layer_types)
    r_on = device_params.get("r_on", 100.0)
    r_off = device_params.get("r_off", 10000.0)
    handles: dict[str, CrossbarHandle] = {}

    for name, module in patched_model.named_modules():
        if not isinstance(module, patch_targets) or not hasattr(module, "weight"):
            continue

        rows = module.weight.shape[0]
        cols = module.weight.data.reshape(rows, -1).shape[1]

        accelerator = SimpleFixedPoint(
            g_min=1.0 / r_off,
            g_max=1.0 / r_on,
            xb_size=(rows, cols),
            device="cpu",
        )
        accelerator.initialize_chip()

        handle_name = f"{name}_crossbar_0"
        handles[handle_name] = CrossbarHandle(
            name=handle_name,
            accessor=XBTorchAccessor(accelerator),
            layer=module,
            device_params=device_params,
        )

    logger.info("Built %d XBTorch-backed crossbar handle(s)", len(handles))
    return handles
