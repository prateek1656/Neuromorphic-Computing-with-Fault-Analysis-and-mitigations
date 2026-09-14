"""CrossSim backend (github.com/sandialabs/cross-sim): replaces each patched
Linear/Conv2d layer with a real CrossSim analog equivalent whose forward
pass computes through the same 2D matrix this module's ConductanceAccessor
reads and writes.

Writing a fault into the accessor via `write()` measurably changes the
model's actual forward-pass output, because CrossSim's analog layers
genuinely compute through their programmed matrix rather than the layer's
original `.weight` tensor - the property XBTorch's backend was dropped for
lacking (see docs/planning/project-setup-plan.md §9).

CrossSim's own nonideality models (read_noise/programming_error/drift_error)
are deliberately left at their library default of `enable=False` - fault
injection for this backend, like every other backend, is entirely
neurofault's own custom path (faults/injection.py), never CrossSim's built-in
simulation.

crossbar_config.tile_shape IS wired, via params.core.rows_max/cols_max - the
real, primary multi-core partitioning controls (params.core.mapping.weights.
*_partition_priority, mentioned in an earlier version of this docstring, is
only a tie-breaking heuristic for uneven splits, not the size limit itself).
**These two field names are inverted relative to the axis they actually
constrain** - verified with an asymmetric test: on a Linear(20, 16) layer
(weight shape (16, 20)), setting cols_max=4 (not rows_max) produced 4 cores of
shape (4, 20), i.e. cols_max constrains the row dimension (nrow/out_features)
and rows_max constrains the column dimension (ncol/in_features). _build_params()
below cross-wires tile_shape=(tile_rows, tile_cols) accordingly - do not
"fix" this to look intuitive, it would silently misconfigure the partition.
get_matrix()/set_matrix() already transparently reassemble across cores
(verified: logical shape stays the full layer size, a fault write on one core
still changes forward output) - build_handles() below needs no changes for
this, one handle per layer stays correct regardless of internal tiling.

ADC/DAC resolution (crossbar_config.adc_resolution/dac_resolution) IS wired,
via CrossSim's QuantizerADC/QuantizerDAC models (real, bit-accurate
quantization - verified this session) with ADCRangeLimits.MAX for the ADC's
quantization range, which auto-derives from the actual signal rather than
needing manual calibration data.

Unit-space caveat, stated plainly rather than glossed over: CrossSim's
get_matrix()/core.min/core.max operate in the layer's own weight-space
(verified: matches the original nn.Linear weight's init range exactly, e.g.
~[-0.22, 0.22]), not literal Ohms/Siemens, even though DeviceParameters.Rmin/
Rmax (real Ohms) drive CrossSim's own internal error simulation. Because that
range is signed (unlike a physical conductance range), build_handles() below
derives this handle's r_on/r_off from the layer's weight *magnitude*
(max(|min|, |max|)) rather than its raw signed bounds - deriving r_off
directly from a negative w_min would flip HealthMonitor's g_min/g_max
ordering for an all-negative-weight layer, since 1/x is order-reversing for
negative x. Using signed [-magnitude, +magnitude] as [g_min, g_max] instead
keeps the existing g_min=1/r_off, g_max=1/r_on formula correct with zero
changes to health_monitor.py/faults/injection.py, and maps naturally onto
"stuck-at" faults as pinning to a large-magnitude value of either sign.

crossbar_config.circuit_topology (§5.2 - 1T1R/0T1R/1S1R) IS wired, via
params.xbar.array.parasitics (row/column interconnect wire resistance, plus
separate per-terminal series resistance - real ohms, verified via
ParasiticParameters' own docstring) and params.xbar.array.Icol_max (per-column
current compliance, normalized to one device's max current). None (default)
leaves parasitics disabled and Icol_max uncapped - today's fully-idealized
behavior, unchanged, so no existing experiment config's verified numbers
shift just because this field now exists. Honest simplification, stated
plainly: this models each topology's aggregate resistive/current effect, not
a true nonlinear selector I-V (e.g. OTS threshold switching).

crossbar_config.devices_per_synapse (§5.2 - multi-device/bit-slicing) IS
wired, via CrossSim's own native CoreStyle.BITSLICED (params.core.style) and
params.core.bit_sliced.num_slices - a real, existing CrossSim feature, not
custom-built here. Verified directly: leaving params.core.weight_bits at its
library default of 0 while setting BITSLICED produces a silent
divide-by-zero RuntimeWarning inside CrossSim's own bitsliced_core.py -
_build_params() below always sets weight_bits alongside num_slices (default
8, matching this project's adc_resolution/dac_resolution convention) to
avoid that. get_matrix()/set_matrix() and synchronize() all verified to work
transparently through a bit-sliced core exactly as they do for a flat one -
no accessor changes needed.

Training gap, found and fixed in a later session: CrossSim's analog layers
track an "ideal" weight tensor separately from the analog core's programmed
matrix (the latter is what get_matrix()/forward() actually use) - standard
torch optimizers update parameters in-place, which is exactly the case
CrossSim's own docs warn does NOT auto-propagate to the analog core.
Verified directly: after optimizer.step(), .weight changes but get_matrix()
does not, until simulator.algorithms.dnn.torch.convert.synchronize(model) is
called. synchronize() below wraps that - system.py calls it after every
optimizer.step() so training actually reaches the analog compute path,
not just the shadow ideal-weight tensor.
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
class CrossSimAccessor:
    layer: object  # a CrossSim AnalogLinear/AnalogConv2d instance

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.layer.get_matrix().shape)

    def read(self) -> torch.Tensor:
        return self.layer.get_matrix()

    def write(self, matrix: torch.Tensor) -> None:
        self.layer.core.set_matrix(matrix)


def _build_params(device_params: dict, crossbar_config: CrossbarConfig):
    from simulator import CrossSimParameters
    from simulator.parameters.xbar_parameters import ADCRangeLimits

    params = CrossSimParameters()
    params.simulation.useGPU = crossbar_config.use_gpu
    params.xbar.device.Rmin = device_params.get("r_on", 100.0)
    params.xbar.device.Rmax = device_params.get("r_off", 10000.0)

    # ADC/DAC resolution: real, working quantization (verified this session),
    # not CrossSim's IdealADC/IdealDAC default (infinite precision). MAX range
    # auto-derives the quantization range from the actual signal - no manual
    # calibration data needed.
    params.xbar.adc.mvm.bits = crossbar_config.adc_resolution
    params.xbar.adc.mvm.model = "QuantizerADC"
    params.xbar.adc.mvm.adc_range_option = ADCRangeLimits.MAX
    params.xbar.dac.mvm.bits = crossbar_config.dac_resolution
    params.xbar.dac.mvm.model = "QuantizerDAC"

    if crossbar_config.tile_shape is not None:
        # See module docstring: cols_max/rows_max are inverted relative to
        # the axis they actually constrain - this cross-wire is deliberate,
        # verified, not a bug.
        tile_rows, tile_cols = crossbar_config.tile_shape
        params.core.cols_max = tile_rows
        params.core.rows_max = tile_cols

    if crossbar_config.circuit_topology is not None:
        _apply_circuit_topology(params, crossbar_config.circuit_topology)

    if crossbar_config.devices_per_synapse > 1:
        from simulator.parameters.core_parameters import CoreStyle

        params.core.style = CoreStyle.BITSLICED
        params.core.bit_sliced.num_slices = crossbar_config.devices_per_synapse
        # Real footgun, verified directly: weight_bits left at its library
        # default of 0 alongside BITSLICED produces a silent divide-by-zero
        # RuntimeWarning inside CrossSim's own bitsliced_core.py. Always set
        # both together - 8 matches this project's adc_resolution/
        # dac_resolution convention, not an arbitrary choice.
        if params.core.weight_bits == 0:
            params.core.weight_bits = 8

    return params


# Representative interconnect/access-device resistances (real ohms, but
# modeling choices in the same sense as FaultConfig.weibull_shape's default -
# not a specific device measurement) grounding the three circuit_topology
# values below. See module docstring for the honest caveat that this models
# aggregate resistive/current effect, not a true nonlinear selector I-V.
_WIRE_RESISTANCE_OHMS = 5.0
_TRANSISTOR_ON_RESISTANCE_OHMS = 1.0  # 1T1R: small relative to a memristor's R_off
_SELECTOR_ON_RESISTANCE_OHMS = 20.0  # 1S1R: real, but higher than a transistor's


def _apply_circuit_topology(params, circuit_topology: str) -> None:
    parasitics = params.xbar.array.parasitics
    parasitics.enable = True
    parasitics.Rp_row = _WIRE_RESISTANCE_OHMS
    parasitics.Rp_col = _WIRE_RESISTANCE_OHMS

    if circuit_topology == "1T1R":
        parasitics.Rp_row_terminal = _TRANSISTOR_ON_RESISTANCE_OHMS
        parasitics.Rp_col_terminal = _TRANSISTOR_ON_RESISTANCE_OHMS
        params.xbar.array.Icol_max = 1.0  # capped at one device's max current -
        #   the transistor's actual physical role (current-limiting).
    elif circuit_topology == "0T1R":
        parasitics.Rp_row_terminal = 0.0
        parasitics.Rp_col_terminal = 0.0
        params.xbar.array.Icol_max = 0.0  # uncapped - nothing limits current
    elif circuit_topology == "1S1R":
        parasitics.Rp_row_terminal = _SELECTOR_ON_RESISTANCE_OHMS
        parasitics.Rp_col_terminal = _SELECTOR_ON_RESISTANCE_OHMS
        params.xbar.array.Icol_max = 2.0  # moderately capped - looser than 1T1R
    else:
        raise ValueError(
            f"Unknown crossbar_config.circuit_topology: {circuit_topology!r} "
            "(expected one of '1T1R', '0T1R', '1S1R')"
        )


def patch_layers(
    model: torch.nn.Module,
    crossbar_config: CrossbarConfig,
    device_cls: type,
    device_params: dict,
) -> torch.nn.Module:
    """Replaces only the configured layer types (crossbar_config.patch_layer_types)
    with CrossSim analog equivalents - unlike simulator.algorithms.dnn.torch.
    convert.from_torch()'s own blanket conversion, which also covers
    Conv1d/Conv3d/RNN variants this project doesn't use. device_cls is
    unused: CrossSim has no separate swappable "device class" the way
    MemTorch/AIHWKit do (see devices/registry.py::_build_crosssim_device),
    just the DeviceParameters fields already folded into device_params.
    """
    from simulator.algorithms.dnn.torch.conv import AnalogConv2d
    from simulator.algorithms.dnn.torch.linear import AnalogLinear

    conversion_map = {
        torch.nn.Linear: AnalogLinear,
        torch.nn.Conv2d: AnalogConv2d,
    }
    patch_targets = tuple(_LAYER_TYPES[t] for t in crossbar_config.patch_layer_types)
    params = _build_params(device_params, crossbar_config)

    def _convert_children(module: torch.nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, patch_targets) and type(child) in conversion_map:
                analog_cls = conversion_map[type(child)]
                setattr(module, name, analog_cls.from_torch(child, params, 0))
            else:
                _convert_children(child)

    _convert_children(model)
    return model


def build_handles(
    patched_model: torch.nn.Module, device_params: dict, crossbar_config: CrossbarConfig
) -> dict[str, CrossbarHandle]:
    from simulator.algorithms.dnn.torch.conv import AnalogConv2d
    from simulator.algorithms.dnn.torch.linear import AnalogLinear

    analog_types = (AnalogLinear, AnalogConv2d)
    handles: dict[str, CrossbarHandle] = {}

    for name, module in patched_model.named_modules():
        if not isinstance(module, analog_types):
            continue

        # See module docstring: derive this handle's r_on/r_off from the
        # layer's actual weight-space magnitude, not device_params' literal
        # Ohms, so HealthMonitor/faults/injection.py normalize against the
        # range this accessor's read()/write() actually operate in.
        w_min, w_max = float(module.core.min), float(module.core.max)
        magnitude = max(abs(w_min), abs(w_max), 1e-12)
        handle_params = dict(device_params)
        handle_params["r_on"] = 1.0 / magnitude
        handle_params["r_off"] = -1.0 / magnitude

        handle_name = f"{name}_crossbar_0"
        handles[handle_name] = CrossbarHandle(
            name=handle_name,
            accessor=CrossSimAccessor(module),
            layer=module,
            device_params=handle_params,
        )

    logger.info("Built %d CrossSim-backed crossbar handle(s)", len(handles))
    return handles


def mapping_routine_for(scheme: str):
    """Returns the weight->conductance mapping `layer_reset` needs. CrossSim's
    set_matrix() takes real weight-space values directly - no separate
    weight->conductance derivation needed. `scheme` is accepted only so every
    backend's mapping_routine_for() shares one call signature; CrossSim has
    no DoubleColumn-style split to honor.
    """

    def _map(weight: torch.Tensor, r_on: float, r_off: float, scheme: str) -> torch.Tensor:
        rows = weight.shape[0]
        return weight.reshape(rows, -1)

    return _map


def synchronize(model: torch.nn.Module) -> None:
    """Pushes each analog layer's optimizer-updated ideal weight into its
    analog core - see module docstring for why this is necessary (standard
    torch optimizers update parameters in-place, which CrossSim's own docs
    say does not auto-propagate to the analog core). Call after every
    optimizer.step() during training."""
    from simulator.algorithms.dnn.torch.convert import synchronize as _crosssim_synchronize

    _crosssim_synchronize(model)
