"""Maps a device-technology name to its memtorch memristor class and default
parameters. Replaces the VTEAM-only block that was duplicated verbatim in
two places in the original codebase (fault_tolerant_neuromorphic.py and
self_healing_crossbar.py) - device choice is now a config value, not a
hardcoded if-statement.

memtorch is imported lazily inside build_device() so this module stays
importable without memtorch installed.
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


def build_device(config: DeviceConfig) -> tuple[type, dict]:
    """Returns (memtorch memristor class, params dict) for the given device config."""
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

    raise ValueError(f"Unknown device name: {config.name!r}")
