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

    def __post_init__(self):
        if not (self.r_on > 0 and self.r_off > 0):
            raise ValueError(
                f"DeviceConfig.r_on/r_off must be positive (real resistances), "
                f"got r_on={self.r_on!r}, r_off={self.r_off!r}"
            )
        if self.r_on >= self.r_off:
            raise ValueError(
                f"DeviceConfig.r_on must be less than r_off (on-resistance < "
                f"off-resistance, real device physics), got r_on={self.r_on!r}, "
                f"r_off={self.r_off!r}"
            )
        if self.time_series_resolution <= 0:
            raise ValueError(
                f"DeviceConfig.time_series_resolution must be positive, "
                f"got {self.time_series_resolution!r}"
            )


@dataclass
class CrossbarConfig:
    scheme: str = "DoubleColumn"  # "DoubleColumn" | "SingleColumn"
    tile_shape: tuple[int, int] | None = None
    patch_layer_types: list[str] = field(default_factory=lambda: ["Linear", "Conv2d"])
    max_input_voltage: float = 0.3
    adc_resolution: int = 8  # ADC bits - quantizes the read-out current/output
    dac_resolution: int = 8  # DAC bits - quantizes the input voltages
    redundancy_factor: float = 0.15
    # "idealized_per_cell": remapping restores a faulty cell's own reference
    #   conductance in place - no physical spare-hardware analogue (Phase 1
    #   design decision, see docs/planning/project-setup-plan.md).
    # "row_col_granularity": repairs whole rows/columns at once (word-line/
    #   bit-line resolution, matching how real spare hardware is organized) -
    #   see mitigation/remap.py and crossbar/self_healing.py.
    redundancy_scheme: str = "idealized_per_cell"
    # row_col_granularity only: fraction of a row's/column's own cells that
    # must be critical before that row/column is condemned and restored -
    # condemning on a single critical cell would waste an entire row/column's
    # worth of scarce spare capacity on one bad cell.
    redundancy_condemn_threshold: float = 0.5

    # CrossSim-only (crosssim_backend.py's _build_params): sets
    # params.simulation.useGPU, a real, existing CrossSim mechanism (see
    # simulator/parameters/simulation_parameters.py and backend/backend.py's
    # ComputeBackend singleton, which transparently swaps its whole array
    # module between numpy and cupy) - not something this project built.
    # Default False so every existing config/result on this CPU-only Mac is
    # unaffected. Requires the "gpu" extra (cupy) installed and actual CUDA
    # hardware - verify on the real rented box before trusting a long run,
    # per this project's own verify-before-committing-compute norm; this
    # session's Mac has no CUDA to test the path against.
    use_gpu: bool = False

    # Circuit-level access topology (see docs/planning/project-setup-plan.md §5.2).
    # None (default) preserves today's fully-idealized behavior (no parasitics/
    # series resistance/current compliance modeled) - every existing experiment
    # config's already-verified numbers stay unchanged unless this is set
    # explicitly. "1T1R" (transistor per cell - near-ideal isolation) | "0T1R"
    # (bare crossbar - worst-case sneak path) | "1S1R" (selector device in
    # series) - see crosssim_backend.py/aihwkit_backend.py for the real,
    # physically-meaningful knobs (parasitic wire/terminal resistance, current
    # compliance, series resistance, IR drop) each value maps to.
    circuit_topology: str | None = None

    # Devices combined per synapse (see §5.2's "multi-device-per-synapse /
    # bit-slicing" goal). 1 (default) = today's single-device-per-synapse
    # behavior, unchanged. >1: CrossSim uses its native bit-sliced core
    # (core.style=BITSLICED); AIHWKit wraps the device in a VectorUnitCell of
    # that many physical devices. Orthogonal to `scheme` (DoubleColumn's
    # signed-weight column split), same as tile_shape/redundancy_scheme are.
    devices_per_synapse: int = 1

    def __post_init__(self):
        if self.adc_resolution < 1 or self.dac_resolution < 1:
            raise ValueError(
                f"CrossbarConfig.adc_resolution/dac_resolution must be >= 1 bit, got "
                f"adc_resolution={self.adc_resolution!r}, dac_resolution={self.dac_resolution!r}"
            )
        if not (0.0 <= self.redundancy_factor <= 1.0):
            raise ValueError(
                f"CrossbarConfig.redundancy_factor must be in [0, 1] (a fraction of "
                f"cells/rows/cols), got {self.redundancy_factor!r}"
            )
        if not (0.0 <= self.redundancy_condemn_threshold <= 1.0):
            raise ValueError(
                "CrossbarConfig.redundancy_condemn_threshold must be in [0, 1] (a "
                f"fraction of a row's/column's cells), got {self.redundancy_condemn_threshold!r}"
            )
        if self.devices_per_synapse < 1:
            raise ValueError(
                f"CrossbarConfig.devices_per_synapse must be >= 1, got {self.devices_per_synapse!r}"
            )
        if self.tile_shape is not None and (self.tile_shape[0] <= 0 or self.tile_shape[1] <= 0):
            raise ValueError(
                f"CrossbarConfig.tile_shape dimensions must be positive, got {self.tile_shape!r}"
            )
        if self.circuit_topology is not None and self.circuit_topology not in (
            "1T1R",
            "0T1R",
            "1S1R",
        ):
            raise ValueError(
                f"Unknown crossbar_config.circuit_topology: {self.circuit_topology!r} "
                "(expected None, '1T1R', '0T1R', or '1S1R')"
            )
        if self.redundancy_scheme not in ("idealized_per_cell", "row_col_granularity"):
            raise ValueError(
                f"Unknown crossbar_config.redundancy_scheme: {self.redundancy_scheme!r} "
                "(expected 'idealized_per_cell' or 'row_col_granularity')"
            )


@dataclass
class FaultConfig:
    density: float = 0.1
    distribution: str = "random"  # "random" | "clustered" | "gradient" - spatial pattern,
    #   only used by failure_model="density"; orthogonal to failure_model (see below).
    injection_interval: int = 8  # batches between re-injections

    # failure_model picks WHICH cells fail and HOW MANY, each injection call:
    # "density": today's behavior - a flat density * total_cells picked fresh
    #   each call via `distribution`, no memory of prior calls. Hard stuck-at
    #   (pinned to g_min/g_max) - permanent, never recoverable.
    # "weibull" | "exponential": real time-to-failure models - each cell's
    #   own accumulated simulated cycles (HealthMonitor.cycles) determines its
    #   own conditional failure probability this interval. See
    #   faults/reliability.py and faults/injection.py::_statistical_stuck_at.
    #   Also hard stuck-at, just a different arrival-time distribution.
    # "drift": a real, distinct failure mode found missing this session (see
    #   docs/planning/ "Catching Up" plan) - soft_mitigation was built to
    #   correct "drift/variability, not a hardware-stuck-at device" (see its
    #   own docstring) but this project never actually injected drift, only
    #   stuck-at, so that strategy was never tested against the failure mode
    #   it targets. Cells shift gradually toward one conductance rail
    #   (drift_direction) by drift_magnitude*g_range each injection call,
    #   accumulating over repeated calls (clamped to [g_min, g_max]) - never
    #   marked stuck_mask, since the cell stays writable/functional, just
    #   imprecise. See drift_magnitude/drift_direction below.
    failure_model: str = "density"
    weibull_shape: float = 2.0  # dimensionless shape (k) - 2.0 is the classic
    #   wear-out region (increasing hazard rate), a modeling choice not a
    #   device measurement, so it's the one statistical parameter with a
    #   real default.
    weibull_scale: float | None = None  # characteristic life in simulated
    #   cycles (lambda) - REQUIRED when failure_model="weibull"; no invented
    #   default, must be a real value for the device being modeled.
    exponential_rate: float | None = None  # REQUIRED when
    #   failure_model="exponential"; no invented default.
    # Arrhenius is a modifier on weibull/exponential, not a fourth model -
    # None (default) means no temperature acceleration at all.
    arrhenius_activation_energy_ev: float | None = None
    temperature_kelvin: float = 300.0  # ~27C, room temperature
    reference_temperature_kelvin: float = 300.0  # the temperature
    #   weibull_scale/exponential_rate were characterized at; acceleration
    #   factor is exactly 1.0 when temperature_kelvin == this.

    # failure_model="drift" only. No single canonical value exists in the
    # literature the way stuck-at density figures do (checked this session -
    # retention/endurance papers describe the mechanism, not a portable
    # per-cycle magnitude) - this is a documented modeling judgment call, not
    # a lookup. 0.05 (5% of g_range per injection event) is a starting point
    # chosen so repeated injections (this project's own moderate-regime
    # calibration uses ~3 events) compound into a meaningfully degraded but
    # not instantly rail-clamped cell.
    drift_magnitude: float = 0.05
    # "toward_hrs": literature on oxide RRAM retention loss most commonly
    #   describes LRS filaments as the less stable state, spontaneously
    #   losing conductance (drifting toward HRS/g_min) via oxygen-vacancy
    #   diffusion - a real but not universal-across-all-technologies
    #   direction, used as the default with this caveat stated plainly.
    # "toward_lrs": the opposite bias, for devices/technologies where that's
    #   the better-supported direction - a config choice, not a default.
    # "random_per_cell": each affected cell's direction is decided once (at
    #   its first drift event) and reused for every subsequent event -
    #   matches real drift being monotonic per-device, not a fault model
    #   that flips direction call to call.
    drift_direction: str = "toward_hrs"

    def __post_init__(self):
        if self.failure_model == "weibull" and self.weibull_scale is None:
            raise ValueError(
                "FaultConfig.failure_model='weibull' requires weibull_scale to be set "
                "explicitly - no invented default for a device's characteristic life."
            )
        if self.failure_model == "exponential" and self.exponential_rate is None:
            raise ValueError(
                "FaultConfig.failure_model='exponential' requires exponential_rate to be "
                "set explicitly - no invented default for a device's failure rate."
            )
        if self.failure_model not in ("density", "weibull", "exponential", "drift"):
            raise ValueError(
                f"Unknown FaultConfig.failure_model: {self.failure_model!r} "
                "(expected 'density', 'weibull', 'exponential', or 'drift')"
            )
        if self.drift_direction not in ("toward_hrs", "toward_lrs", "random_per_cell"):
            raise ValueError(
                f"Unknown FaultConfig.drift_direction: {self.drift_direction!r} "
                "(expected 'toward_hrs', 'toward_lrs', or 'random_per_cell')"
            )
        if not (0.0 <= self.drift_magnitude <= 1.0):
            raise ValueError(
                f"FaultConfig.drift_magnitude must be in [0, 1], got {self.drift_magnitude!r}"
            )
        if self.distribution not in ("random", "clustered", "gradient"):
            raise ValueError(
                f"Unknown FaultConfig.distribution: {self.distribution!r} "
                "(expected 'random', 'clustered', or 'gradient')"
            )
        if not (0.0 <= self.density <= 1.0):
            raise ValueError(f"FaultConfig.density must be in [0, 1], got {self.density!r}")
        if self.injection_interval < 1:
            raise ValueError(
                f"FaultConfig.injection_interval must be >= 1, got {self.injection_interval!r}"
            )
        if self.weibull_shape <= 0:
            raise ValueError(
                f"FaultConfig.weibull_shape must be positive, got {self.weibull_shape!r}"
            )
        if self.weibull_scale is not None and self.weibull_scale <= 0:
            raise ValueError(
                f"FaultConfig.weibull_scale must be positive, got {self.weibull_scale!r}"
            )
        if self.exponential_rate is not None and self.exponential_rate <= 0:
            raise ValueError(
                f"FaultConfig.exponential_rate must be positive, got {self.exponential_rate!r}"
            )
        if self.temperature_kelvin <= 0 or self.reference_temperature_kelvin <= 0:
            raise ValueError(
                "FaultConfig.temperature_kelvin/reference_temperature_kelvin must be "
                f"positive (real Kelvin), got temperature_kelvin={self.temperature_kelvin!r}, "
                f"reference_temperature_kelvin={self.reference_temperature_kelvin!r}"
            )


@dataclass
class MitigationConfig:
    enable_soft_mitigation: bool = False
    enable_remapping: bool = False
    enable_layer_reset: bool = False
    enable_predictive: bool = True
    force_remapping: bool = False
    health_threshold: float = 75.0
    health_check_interval: int = 10  # batches between check_health() calls
    # "full_diff" (default): read every cell, compare to reference - always
    #   correct, models the expensive naive approach real hardware avoids.
    # "checksum": cheap row/column-sum first pass (see health_monitor.py);
    #   falls back to full_diff automatically once a mismatch is found, so
    #   fault localization is identical whenever a fault actually exists -
    #   only the no-fault case gets cheaper (in modeled cell-touch terms).
    detection_method: str = "full_diff"
    # checksum only: real detection-method upgrade made this session (see
    # docs/planning/ "Catching Up" plan) - replaces a fixed-fraction-of-
    # g_range threshold with a variance-based adaptive one (V-ABFT's
    # approach, 2026: derive the flagging threshold from the OBSERVED spread
    # of row/column checksum deviations across the whole crossbar this call,
    # rather than assuming a fixed magnitude always means "fault"). A
    # row/column is flagged when its deviation exceeds
    # mean(deviation) + checksum_z_threshold * std(deviation) - the classic
    # z-score outlier convention (3.0 = "3-sigma"). Honest limitation, same
    # as the fixed-threshold version it replaces: self-referential to the
    # current population, so it degrades once faults are a majority rather
    # than a minority of cells (the population's own mean/std become
    # dominated by the faults) - see health_monitor.py.
    checksum_z_threshold: float = 3.0
    # Real design flaw found and fixed this session (see
    # docs/planning/project-setup-plan.md's mitigation-persistence section):
    # layer_reset wipes an ENTIRE layer's weights back to their random
    # pre-training initialization - unlike soft_mitigation (a small nudge)
    # and remapping (capacity-limited to redundancy_factor), it had no
    # throttle at all, so once avg_health crossed health_threshold it fired
    # on essentially every subsequent health check for the rest of training
    # (measured: 222 times over 1257 batches in a real run), repeatedly
    # discarding legitimate learning - actively worse than no mitigation at
    # all. This gates it to at most once every N check_health() CALLS
    # (project-wide, not per-batch, so it composes with health_check_interval
    # rather than introducing an unrelated batch-count knob) per layer -
    # "last resort" should mean rare, not near-constant.
    layer_reset_cooldown_checks: int = 10

    def __post_init__(self):
        if not (0.0 <= self.health_threshold <= 100.0):
            raise ValueError(
                f"MitigationConfig.health_threshold must be in [0, 100], got {self.health_threshold!r}"
            )
        if self.health_check_interval < 1:
            raise ValueError(
                f"MitigationConfig.health_check_interval must be >= 1, got {self.health_check_interval!r}"
            )
        if self.detection_method not in ("full_diff", "checksum"):
            raise ValueError(
                f"Unknown MitigationConfig.detection_method: {self.detection_method!r} "
                "(expected 'full_diff' or 'checksum')"
            )
        if self.checksum_z_threshold < 0:
            raise ValueError(
                "MitigationConfig.checksum_z_threshold must be >= 0, got "
                f"{self.checksum_z_threshold!r}"
            )
        if self.layer_reset_cooldown_checks < 0:
            raise ValueError(
                "MitigationConfig.layer_reset_cooldown_checks must be >= 0, got "
                f"{self.layer_reset_cooldown_checks!r}"
            )


@dataclass
class RetrainingConfig:
    # Fault-aware retraining baseline (see docs/planning/project-setup-plan.md
    # §6): instead of runtime mitigation reacting to faults during inference,
    # this bakes a fixed, known defect map into the crossbar once before
    # training and never changes it again - gradient descent itself learns
    # weights for the surviving cells that compensate for the permanently
    # stuck ones. When enabled, ExperimentConfig.fault still controls the
    # defect map's density/distribution/failure_model (same generation
    # mechanism, just triggered once instead of periodically), and
    # mitigation.enable_* should be set to False since retraining itself is
    # the compensation mechanism here.
    enable_fault_aware: bool = False


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42
    simulator: str = "crosssim"  # "crosssim" | "aihwkit" - which crossbar simulation
    #   backend patches the model.
    dataset: str = "cifar10"  # "cifar10" | "mnist"
    train_samples: int = 2000
    test_samples: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.001
    # Real gap found this session (see docs/planning/ "Catching Up" plan):
    # no L2 regularization at all, on a model well below realistic accuracy
    # on a severely-limited-data regime - a real lever for generalization,
    # not architecture. 0.0 (off) preserves prior behavior exactly.
    weight_decay: float = 0.0
    # Cosine-anneals the optimizer's LR from learning_rate down to ~0 over
    # the full run (Loshchilov & Hutter, SGDR) - same session's finding:
    # a flat LR the whole run plateaus early once augmentation is in the
    # mix (augmentation needs more effective training time to pay off, and
    # a decaying LR is standard practice for actually using that time to
    # converge rather than oscillate). False preserves prior behavior.
    lr_cosine_schedule: bool = False
    epochs: int = 15
    num_batches: int = 128
    eval_interval: int = 3
    device: DeviceConfig = field(default_factory=DeviceConfig)
    crossbar: CrossbarConfig = field(default_factory=CrossbarConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    output_dir: str = "results"

    def __post_init__(self):
        if self.simulator not in ("crosssim", "aihwkit"):
            raise ValueError(
                f"Unknown ExperimentConfig.simulator: {self.simulator!r} "
                "(expected 'crosssim' or 'aihwkit')"
            )
        if self.dataset not in ("cifar10", "mnist"):
            raise ValueError(
                f"Unknown ExperimentConfig.dataset: {self.dataset!r} (expected 'cifar10' or 'mnist')"
            )
        if self.train_samples < 1 or self.test_samples < 1:
            raise ValueError(
                f"ExperimentConfig.train_samples/test_samples must be >= 1, got "
                f"train_samples={self.train_samples!r}, test_samples={self.test_samples!r}"
            )
        if self.batch_size < 1:
            raise ValueError(f"ExperimentConfig.batch_size must be >= 1, got {self.batch_size!r}")
        if self.epochs < 1:
            raise ValueError(f"ExperimentConfig.epochs must be >= 1, got {self.epochs!r}")
        if self.num_batches < 1:
            raise ValueError(f"ExperimentConfig.num_batches must be >= 1, got {self.num_batches!r}")
        if self.eval_interval < 1:
            raise ValueError(
                f"ExperimentConfig.eval_interval must be >= 1, got {self.eval_interval!r}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"ExperimentConfig.learning_rate must be positive, got {self.learning_rate!r}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"ExperimentConfig.weight_decay must be >= 0, got {self.weight_decay!r}"
            )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file, converting nested dicts
    (device/crossbar/fault/mitigation) into their typed dataclasses so
    downstream code always has real fields to check, never raw dict keys."""
    raw = yaml.safe_load(Path(path).read_text())

    if "tile_shape" in raw.get("crossbar", {}) and raw["crossbar"]["tile_shape"] is not None:
        raw["crossbar"]["tile_shape"] = tuple(raw["crossbar"]["tile_shape"])

    return ExperimentConfig(
        **{
            k: v
            for k, v in raw.items()
            if k not in ("device", "crossbar", "fault", "mitigation", "retraining")
        },
        device=DeviceConfig(**raw.get("device", {})),
        crossbar=CrossbarConfig(**raw.get("crossbar", {})),
        fault=FaultConfig(**raw.get("fault", {})),
        mitigation=MitigationConfig(**raw.get("mitigation", {})),
        retraining=RetrainingConfig(**raw.get("retraining", {})),
    )
