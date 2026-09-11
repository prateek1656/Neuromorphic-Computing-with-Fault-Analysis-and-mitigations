"""Thin wrapper around memtorch.mn.Module.patch_model.

memtorch itself is imported lazily, inside patch_layers(), so this module
(and CrossbarHandle) stays importable in environments without memtorch
installed - unit tests use a FakeCrossbar in place of a real one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neurofault.config import CrossbarConfig

_LAYER_TYPES = {
    "Linear": torch.nn.Linear,
    "Conv2d": torch.nn.Conv2d,
}


@dataclass
class CrossbarHandle:
    """The single canonical reference to one memristive crossbar.

    Every module that needs to read or write this crossbar's conductances
    (health monitoring, fault injection, mitigation) takes this same object -
    never a second, independently-constructed Crossbar. That invariant is
    what prevents the disconnected-shadow-crossbar bug the original codebase
    had (see docs/planning/project-setup-plan.md, bug #1 and #3).
    """

    name: str
    crossbar: Any  # memtorch.bh.crossbar.Crossbar, or a FakeCrossbar in tests
    layer: torch.nn.Module
    device_params: dict


def patch_layers(
    model: torch.nn.Module,
    crossbar_config: CrossbarConfig,
    device_cls: type,
    device_params: dict,
) -> torch.nn.Module:
    """Convert model's layers into memristive layers via MemTorch's patch_model.

    Defaults to patching BOTH Linear and Conv2d layers (crossbar_config.patch_layer_types),
    fixing the original codebase's gap where only Linear ever became memristive and the
    CNN's Conv2d layers never experienced any simulated faults.
    """
    from memtorch.bh.crossbar import Scheme
    from memtorch.map.Input import naive_scale
    from memtorch.map.Parameter import naive_map
    from memtorch.mn.Module import patch_model

    patch_targets = [_LAYER_TYPES[t] for t in crossbar_config.patch_layer_types]
    scheme = (
        Scheme.DoubleColumn if crossbar_config.scheme == "DoubleColumn" else Scheme.SingleColumn
    )

    return patch_model(
        model=model,
        memristor_model=device_cls,
        memristor_model_params=device_params,
        module_parameters_to_patch=patch_targets,
        mapping_routine=naive_map,
        transistor=True,
        scheme=scheme,
        tile_shape=crossbar_config.tile_shape,
        max_input_voltage=crossbar_config.max_input_voltage,
        scaling_routine=naive_scale,
        ADC_resolution=crossbar_config.adc_resolution,
        ADC_overflow_rate=crossbar_config.adc_overflow_rate,
        quant_method=crossbar_config.quant_method,
    )


def build_handles(patched_model: torch.nn.Module, device_params: dict) -> dict[str, CrossbarHandle]:
    """Build one CrossbarHandle per memristive crossbar found in the patched model."""
    handles: dict[str, CrossbarHandle] = {}
    for name, module in patched_model.named_modules():
        if hasattr(module, "crossbars"):
            for i, crossbar in enumerate(module.crossbars):
                handle_name = f"{name}_crossbar_{i}"
                handles[handle_name] = CrossbarHandle(
                    name=handle_name,
                    crossbar=crossbar,
                    layer=module,
                    device_params=device_params,
                )
    return handles
