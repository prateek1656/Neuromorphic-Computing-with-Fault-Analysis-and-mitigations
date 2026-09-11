"""MemTorch backend: patches a model via memtorch.mn.Module.patch_model and
wraps each resulting crossbar's .conductance_matrix as a ConductanceAccessor.

memtorch is imported lazily inside these functions, never at module scope,
so this file stays importable even when memtorch isn't installed - only
calling patch_layers()/build_handles() actually requires it.

NOTE: memtorch itself is confirmed unmaintained since 2022-04-13 and its
current PyPI packaging fails to install in two independent, unconditional
ways (CUDA build requires an NVIDIA driver at build-metadata time; the CPU
variant pins the now-deprecated `sklearn` package name). See
docs/planning/project-setup-plan.md Phase 2 for the full validation. This
backend is kept for compatibility/reference but XBTorch and AIHWKit are the
backends actually expected to run.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neurofault.config import CrossbarConfig
from neurofault.crossbar.array import CrossbarHandle

_LAYER_TYPES = {
    "Linear": torch.nn.Linear,
    "Conv2d": torch.nn.Conv2d,
}


@dataclass
class MemTorchAccessor:
    crossbar: object  # memtorch.bh.crossbar.Crossbar

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.crossbar.conductance_matrix.shape)

    def read(self) -> torch.Tensor:
        return self.crossbar.conductance_matrix

    def write(self, matrix: torch.Tensor) -> None:
        self.crossbar.write_conductance_matrix(matrix)


def patch_layers(
    model: torch.nn.Module,
    crossbar_config: CrossbarConfig,
    device_cls: type,
    device_params: dict,
) -> torch.nn.Module:
    """Convert model's layers into memristive layers via MemTorch's patch_model.

    Defaults to patching BOTH Linear and Conv2d layers (crossbar_config.patch_layer_types),
    fixing the original codebase's gap where only Linear ever became memristive.
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


def mapping_routine_for(scheme: str):
    """The weight->conductance mapping callable layer_reset needs to
    re-derive conductances after restoring original weights. Returns
    memtorch's naive_map, which apply_layer_reset calls as
    mapping_routine(weight, r_on, r_off, scheme)."""
    from memtorch.map.Parameter import naive_map

    return naive_map


def build_handles(
    patched_model: torch.nn.Module,
    device_params: dict,
    crossbar_config: CrossbarConfig | None = None,
) -> dict[str, CrossbarHandle]:
    """Build one CrossbarHandle per memristive crossbar found in the patched model.

    crossbar_config is unused here (patch_layers already selected which
    layer types got crossbars) - accepted only so every backend's
    build_handles() shares one call signature in system.py.
    """
    handles: dict[str, CrossbarHandle] = {}
    for name, module in patched_model.named_modules():
        if hasattr(module, "crossbars"):
            for i, crossbar in enumerate(module.crossbars):
                handle_name = f"{name}_crossbar_{i}"
                handles[handle_name] = CrossbarHandle(
                    name=handle_name,
                    accessor=MemTorchAccessor(crossbar),
                    layer=module,
                    device_params=device_params,
                )
    return handles
