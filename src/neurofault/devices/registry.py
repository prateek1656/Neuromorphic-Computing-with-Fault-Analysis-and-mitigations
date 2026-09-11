"""Maps a device-technology name to its simulator-specific device class and
default parameters, dispatched per simulation backend (memtorch | xbtorch |
aihwkit). Replaces the VTEAM-only block that was duplicated verbatim in two
places in the original codebase - device AND backend choice are now config
values, not hardcoded if-statements.

Each backend library is imported lazily inside its own build_*_device()
function, so this module stays importable without any of them installed.
"""

from __future__ import annotations

from neurofault.config import DeviceConfig

_VTEAM_DEFAULTS = {
    "d": 3e-9,
    "k_on": -10,
    "k_off": 5e-4,
    "alpha_on": 3,
    "alpha_off": 1,
    "v_on": -0.2,
    "v_off": 0.02,
    "x_on": 0,
    "x_off": 3e-9,
}


def build_device(config: DeviceConfig, simulator: str = "memtorch") -> tuple[type, dict]:
    """Returns (device class, params dict) for the given device config and backend."""
    if simulator == "memtorch":
        return _build_memtorch_device(config)
    if simulator == "xbtorch":
        return _build_xbtorch_device(config)
    if simulator == "aihwkit":
        return _build_aihwkit_device(config)
    raise ValueError(f"Unknown simulator backend: {simulator!r}")


def _build_memtorch_device(config: DeviceConfig) -> tuple[type, dict]:
    params = {
        "r_on": config.r_on,
        "r_off": config.r_off,
        "time_series_resolution": config.time_series_resolution,
    }

    if config.name == "vteam":
        from memtorch.bh.memristor.VTEAM import VTEAM

        params.update(_VTEAM_DEFAULTS)
        params.update(config.extra_params)
        return VTEAM, params

    if config.name == "linear_ion_drift":
        from memtorch.bh.memristor.LinearIonDrift import LinearIonDrift

        params.update(config.extra_params)
        return LinearIonDrift, params

    if config.name == "data_driven":
        from memtorch.bh.memristor.Data_Driven import Data_Driven

        params.update(config.extra_params)
        return Data_Driven, params

    if config.name == "stanford_pku":
        from memtorch.bh.memristor.Stanford_PKU import Stanford_PKU

        params.update(config.extra_params)
        return Stanford_PKU, params

    raise ValueError(f"Unknown memtorch device name: {config.name!r}")


def _build_xbtorch_device(config: DeviceConfig) -> tuple[type, dict]:
    """XBTorch presets (xbtorch.devices.presets): physically-motivated
    (Analytical*) and empirically-fitted (Tabular*, including real FeFET
    measurement data) device models - a genuinely different taxonomy from
    memtorch's, not just a renamed VTEAM.

    NOTE: crossbar/backends/xbtorch_backend.py's current per-layer-accelerator
    design (see its module docstring) does not use these presets - it drives
    SimpleFixedPoint directly from DeviceConfig.r_on/r_off, since these
    presets are meant for use via xbtorch.initialize(device_type=...), part
    of the shared-chip global-patching flow that backend deliberately
    bypasses. Kept here for the more faithful shared-chip backend that
    remains a real follow-up, not implemented yet.
    """
    name_map = {
        "analytical_ideal": "AnalyticalIdeal",
        "analytical_real": "AnalyticalReal",
        "tabular_compact_fefet": "TabularCompactFeFETKriging",
        "tabular_experimental_fefet": "TabularExperimentalFemFETKriging",
    }
    params = {"r_on": config.r_on, "r_off": config.r_off, **config.extra_params}

    # xbtorch_backend's current per-layer-accelerator design doesn't consume
    # the returned class (see its module docstring) - only r_on/r_off matter
    # today. Resolve a real preset class opportunistically when the name
    # matches one, but don't hard-fail on an unrecognized/default name (e.g.
    # DeviceConfig's default "vteam", meaningful for memtorch, meaningless
    # here) - that would force every xbtorch config to redundantly name an
    # unused class just to get past this check.
    if config.name in name_map:
        from xbtorch.devices import presets

        return getattr(presets, name_map[config.name]), params
    return None, params


def _build_aihwkit_device(config: DeviceConfig) -> tuple[type, dict]:
    """AIHWKit presets (aihwkit.simulator.presets): calibrated against real
    IBM hardware measurements (1M-device PCM array; ReRAM/HfOx arrays) -
    the richest real-silicon-grounded device statistics of the three backends,
    but with no persistent fault model of its own (noise/drift only) - fault
    injection for this backend is entirely neurofault's own custom path."""
    from aihwkit.simulator.presets import devices as aihwkit_devices

    name_map = {
        "pcm": "PCMPresetDevice",
        "reram_es": "ReRamESPresetDevice",
        "reram_sb": "ReRamSBPresetDevice",
        "ecram": "EcRamPresetDevice",
    }
    if config.name not in name_map:
        raise ValueError(
            f"Unknown aihwkit device name: {config.name!r} (expected one of {list(name_map)})"
        )

    preset_cls = getattr(aihwkit_devices, name_map[config.name])
    return preset_cls, dict(config.extra_params)
