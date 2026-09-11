"""Typed experiment configuration. Everything downstream takes one of these
dataclasses instead of raw dicts/kwargs, so required fields (like the
enable_* mitigation flags) can't be silently omitted or ignored."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DeviceConfig:
    name: str = "vteam"  # key into devices/registry.py
    r_on: float = 100.0
    r_off: float = 10000.0
    time_series_resolution: float = 1e-10
    extra_params: dict = field(default_factory=dict)  # device-specific overrides


@dataclass
class CrossbarConfig:
    scheme: str = "DoubleColumn"  # "DoubleColumn" | "SingleColumn"
    tile_shape: tuple[int, int] | None = None
    patch_layer_types: list[str] = field(default_factory=lambda: ["Linear", "Conv2d"])
    max_input_voltage: float = 0.3
    adc_resolution: int = 8
    adc_overflow_rate: float = 0.0
    quant_method: str = "linear"
    redundancy_factor: float = 0.15
    # Documents the Phase 1 design decision explicitly per-run: remapping restores
    # a faulty cell's own reference conductance in place rather than modeling
    # physical spare row/column hardware. See docs/planning/project-setup-plan.md.
    redundancy_scheme: str = "idealized_per_cell"


@dataclass
class FaultConfig:
    density: float = 0.1
    distribution: str = "random"  # "random" | "clustered" | "gradient"
    injection_interval: int = 8  # batches between re-injections
    backend: str = "auto"  # "auto" | "native" | "custom"


@dataclass
class MitigationConfig:
    enable_soft_mitigation: bool = False
    enable_remapping: bool = False
    enable_layer_reset: bool = False
    enable_predictive: bool = True
    force_remapping: bool = False
    health_threshold: float = 75.0
    health_check_interval: int = 10  # batches between check_health() calls


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    dataset: str = "cifar10"  # "cifar10" | "mnist"
    train_samples: int = 2000
    test_samples: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 15
    num_batches: int = 128
    eval_interval: int = 3
    device: DeviceConfig = field(default_factory=DeviceConfig)
    crossbar: CrossbarConfig = field(default_factory=CrossbarConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    output_dir: str = "results"


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file, converting nested dicts
    (device/crossbar/fault/mitigation) into their typed dataclasses so
    downstream code always has real fields to check, never raw dict keys."""
    raw = yaml.safe_load(Path(path).read_text())

    if "tile_shape" in raw.get("crossbar", {}) and raw["crossbar"]["tile_shape"] is not None:
        raw["crossbar"]["tile_shape"] = tuple(raw["crossbar"]["tile_shape"])

    return ExperimentConfig(
        **{k: v for k, v in raw.items() if k not in ("device", "crossbar", "fault", "mitigation")},
        device=DeviceConfig(**raw.get("device", {})),
        crossbar=CrossbarConfig(**raw.get("crossbar", {})),
        fault=FaultConfig(**raw.get("fault", {})),
        mitigation=MitigationConfig(**raw.get("mitigation", {})),
    )
