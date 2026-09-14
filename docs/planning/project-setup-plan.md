# Project Setup Plan

Status: **planning only — nothing in this document is implemented yet.**
This is the agreed plan for restructuring the repo before any new code is written.

## 1. Goals

- A modular, testable, later-packagable Python project (not one 1000+ line script per concern).
- Reproducible environment that can be spun up on a rented cloud GPU box in minutes.
- Support multiple memristor device types and crossbar configurations, not just one hardcoded VTEAM setup.
- Match how the fault-tolerance research community actually models devices, injects faults, and reports results — so results are comparable to published work, not just internally consistent.
- Fix the known correctness bugs (remapping not touching the compute path, unconditional layer-reset fallthrough, disconnected shadow crossbar, unreachable soft-mitigation path) as part of the restructure, not after.

## 2. Directory structure

```
neuromorphic-fault-tolerance/
├── pyproject.toml              # uv-managed deps, build config, entry points
├── uv.lock                     # committed lockfile — reproducible on any rented box
├── README.md / LICENSE / CITATION.cff / .gitignore
├── Dockerfile                  # provider-agnostic: same image runs on RunPod/Vast/Lambda/local GPU
├── .github/workflows/ci.yml    # lint + typecheck + CPU smoke tests on every push
├── src/
│   └── neurofault/             # the installable package
│       ├── devices/            # memristor device models (VTEAM + others, see §5)
│       ├── crossbar/           # crossbar array, self-healing/redundancy, health monitor
│       ├── faults/             # injection + fault models (stuck-at, drift, distributions)
│       ├── mitigation/         # soft.py / remap.py / reset.py — each gated independently,
│       │                       #   no silent fallthrough between strategies
│       ├── models/             # neural network architectures (SimpleCNN, etc.)
│       ├── system.py           # FaultTolerantNeuromorphic orchestration
│       └── config.py           # typed experiment config (dataclass), replaces raw dicts
├── experiments/
│   ├── configs/*.yaml          # one file per experiment arm and per crossbar configuration
│   └── run.py                  # single CLI entrypoint: uv run experiments/run.py --config configs/combined.yaml
├── tests/
│   ├── unit/                   # per-module, fast, CPU-only
│   └── integration/            # tiny end-to-end smoke run
├── results/                    # gitignored — real run outputs, never committed
├── notebooks/                  # exploration only, not source of truth
└── docs/
    ├── planning/               # this file and future planning docs
    ├── dissertation.pdf
    └── device-configs.md       # reference table of supported crossbar configurations (see §5)
```

`main.py`'s current responsibilities (multiprocessing, plotting, report-writing, config
definitions, orchestration all in one file) get split across `experiments/run.py` and the
`src/neurofault/` modules so each piece is independently testable.

## 3. Dependencies and tooling

- **uv** for environment + dependency management (chosen over pip/Poetry): single lockfile
  (`uv.lock`), fast installs — matters when billed by the hour on rented compute.
- `torch` + `torchvision` — pinned exact versions; CUDA wheel for cloud, CPU wheel for local dev/tests.
- `memtorch` and `xbtorch` were both evaluated and **dropped entirely** — see §9 for the
  reasoning behind each. `crosssim` and `aihwkit` are the two supported crossbar simulation
  backends going forward.
- `numpy`, `pandas`, `matplotlib` — pinned.
- `pyyaml` for experiment/device configs.
- Dev group: `pytest`, `ruff` (lint + format), `mypy` (light touch — public function signatures only).

## 4. Compute / GPU plan

Cloud GPU provider is **not decided yet** — deliberately left open until we're ready to actually
rent hours. What gets built regardless of provider:

- One `Dockerfile` (CUDA base image + `uv sync`) that runs identically on RunPod, Vast.ai,
  Lambda, or a local GPU — no provider-specific setup script to maintain.
- **Vectorize before renting anything.** Several hot paths today (`SelfHealingCrossbar`'s
  read/write matrix, `_detect_runtime_faults_by_conductance`) loop over individual `(row, col)`
  cells in plain Python. That is an interpreter bottleneck, not a tensor-math bottleneck — a GPU
  will not speed it up until these are rewritten as batched tensor ops. Sequence: vectorize first
  (fast enough to iterate on CPU) → rent GPU hours only for full-scale sweeps once the code is
  actually GPU-bound.
- Given the workload (small CNN, CIFAR-10-scale dataset), a single mid-range GPU is sufficient
  once vectorized — no need to plan for multi-GPU or datacenter-class cards.

**Decided (2026-09-12): no cloud GPU needed — runs locally on the M4 Mac.** Real timing, not
estimated: `no_mitigation.yaml` (CrossSim, the same 15-epoch/128-batch/2000-train-sample shape as
every other real config) measured 31.9s for 1 epoch and 95.6s for 3 epochs — linear, no
meaningful fixed setup cost, ~900% CPU (numpy/CrossSim already parallelizes across the M4's 10
cores). `no_mitigation_aihwkit.yaml` (AIHWKit backend) measured 25.7s/epoch at ~183% CPU (mostly
single-threaded torch-on-CPU). At these rates a full 15-epoch CrossSim arm is ~8 minutes and a
full sweep of all ~12 real configs at one seed is comfortably under 2 hours — even a 5-seed sweep
for statistical significance is an overnight-on-a-laptop job, not a rental. CrossSim's GPU path is
CUDA-only (via CuPy) so it couldn't use this Mac's GPU regardless; AIHWKit's `TorchInferenceRPUConfig`
tile is plain PyTorch so it likely could target MPS, but there's no need to find out — CPU is
already fast enough for this workload's scale. Revisit only if a future config axis (much larger
crossbars, many more seeds, or a `circuit_topology` parasitics-solver sweep — see §5.2's noted
120s+ single-layer cost) actually makes CPU the bottleneck.

**Environment fix required on this Mac, done this session:** importing AIHWKit aborts the process
(`Fatal Python error: Aborted`, SIGABRT) unless `KMP_DUPLICATE_LIB_OK=TRUE` is set before torch or
scikit-learn load — both bundle their own `libomp.dylib`
(`.venv/lib/python3.12/site-packages/{torch/lib,sklearn/.dylibs}/libomp.dylib`), and AIHWKit's
import chain pulls in scikit-learn alongside the already-imported torch, so macOS's dynamic linker
aborts on the second `libomp` init. This is a well-known, common conflict for this exact pair of
libraries, not a project-specific bug — the fix is one process-env flag, not a dependency change.
Verified this doesn't silently corrupt numerics for this project's single-process CPU-only use (a
torch-vs-numpy matmul cross-check matched exactly with the flag set). Fixed at the two real
entrypoints so no one has to remember to export it by hand: root `conftest.py` (pytest) and the
top of `experiments/run.py`, both setting `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`
before the `torch` import.

## 5. Memristor device and crossbar configurations

Today the codebase hardcodes exactly one device model (VTEAM) and one crossbar configuration
(single-column-per-weight via `Scheme.DoubleColumn`, fixed `r_on=100Ω` / `r_off`, fixed
per-cell "redundancy" that doesn't reflect how real spare resources are organized). That's not
representative of the field, and it means every result is really a claim about one specific,
somewhat arbitrary configuration rather than the general strategy. The restructure should turn
"which device / which crossbar layout" into a config axis, not a hardcoded assumption.

### 5.1 Device technologies to support

| Technology | Real endurance range | Real retention behavior | Notes |
|---|---|---|---|
| Oxide-based RRAM (VTEAM-modeled, e.g. HfOx/TaOx) | 10^6–10^9 cycles | drift-prone, moderate | what the codebase currently models exclusively |
| Phase-Change Memory (PCM) | 10^6–10^8 cycles | resistance drift over time (well-documented in literature) | different fault signature — gradual drift is dominant, not stuck-at |
| Electrochemical Metallization (ECM/CBRAM) | device-dependent, filament-based | can be more prone to abrupt filament rupture/formation faults | different stuck-at mechanism than oxide RRAM |
| Ferroelectric (FeFET/FTJ) | 10^10–10^12 cycles | strong retention, different failure modes (polarization fatigue) | much higher endurance — changes which fault rates are even realistic |
| Magnetic (STT-MRAM/spintronic) | 10^12+ cycles | very high retention | least fault-prone of the group; useful as a "best case" comparison point |

Each of these should be a pluggable device model under `src/neurofault/devices/`, with
parameters sourced from published device papers (already partially cited in the dissertation's
lit review) rather than picked arbitrarily. The endurance ranges above match what's actually
reported in the literature, not invented numbers.

**Ferroelectric and Magnetic — done.** `src/neurofault/devices/presets.py` (new) adds
`FeFETPresetDevice(LinearStepDevice)` and `STTMRAMPresetDevice(IdealizedPresetDevice)`,
registered under `devices/registry.py`'s aihwkit `name_map` as `"fefet"`/`"sttmram"` — neither
preset exists in AIHWKit itself (confirmed via `dir(aihwkit.simulator.presets.devices)`), so
these are neurofault's own, following the exact pattern AIHWKit's own presets use (a
`@dataclass` subclass of a base pulsed-device model, parameters fit to one cited paper).
`FeFETPresetDevice` cites Jerry, Chen, Zhang, et al. (incl. Ni), "Ferroelectric FET analog
synapse for acceleration of deep neural network training", IEDM 2017 (IEEE Xplore 8268338): a
real 5-bit/32-state FeFET synapse, 45x (Gmax/Gmin) dynamic range, symmetric potentiation/
depression — `LinearStepDevice` (not `ExpStepDevice`, which PCM/ReRAM use for strongly
nonlinear switching) is the right base for that near-linear, near-symmetric behavior.
`STTMRAMPresetDevice` cites the domain-wall MTJ synapse literature (Sengupta et al., "Proposal
for an All-Spin Artificial Neural Network...", arXiv:1510.00459; comparative arXiv:1910.12919)
— basing it on AIHWKit's own `IdealizedPresetDevice` (near-perfect symmetry, minimal d2d
asymmetry) is a deliberate, stated choice matching this table's own characterization of
STT-MRAM as the least fault-prone, "best case" technology, not a cop-out. Verified end-to-end:
both construct under `TorchInferenceRPUConfig`, a fault write changes the forward pass, and a
real experiment run (`device.name: fefet`, aihwkit backend) trains without error.
**Real, structural limitation on the CrossSim backend, not closed by this work**: CrossSim has
one generic `Rmin`/`Rmax` device model, no swappable per-technology class (confirmed directly)
— `experiments/configs/device_fefet_crosssim.yaml`/`device_sttmram_crosssim.yaml` set
`device.r_on`/`r_off` to the same papers' real dynamic-range values so these technologies are
still exercisable at the conductance-range level on that backend, but CrossSim cannot model
either technology's distinct switching dynamics. Honest sourcing note: unlike FeFET's explicit
45x figure, the domain-wall literature checked didn't give one canonical Gmax/Gmin ratio, so
`device_sttmram_crosssim.yaml` keeps this project's generic default range rather than invent a
number with no cited source.

### 5.2 Circuit-level configurations

- **1T1R** (transistor + resistor per cell) vs **0T1R** (bare crossbar) vs **1S1R**
  (selector + resistor) — these change sneak-path behavior and are standard configuration
  choices in real crossbar designs; should be selectable, not assumed away.
  **Done, CrossSim-only.** `CrossbarConfig.circuit_topology: str | None = None` (default
  preserves today's fully-idealized behavior — no existing experiment's verified numbers
  shift). `crosssim_backend.py::_build_params()` maps each value onto real, already-existing
  CrossSim knobs: `params.xbar.array.parasitics` (row/column wire resistance, real ohms, plus
  separate per-terminal series resistance) and `params.xbar.array.Icol_max` (per-column current
  compliance) — `1T1R` caps `Icol_max` and uses a small terminal resistance (the transistor's
  actual role), `0T1R` leaves both uncapped/zero (worst-case sneak path), `1S1R` uses a
  moderate terminal resistance and compliance (the selector's on-resistance). Honest
  simplification, stated in code: this models each topology's aggregate resistive/current
  effect, not a true nonlinear selector I-V (e.g. OTS threshold switching).
  **Real finding, not supported on AIHWKit**: AIHWKit does expose the equivalent real fields
  (`forward.r_series`, `forward.ir_drop`/`ir_drop_g_ratio`), but `TorchInferenceTile` — this
  project's only working AIHWKit tile (the default `SingleRPUConfig` tile is broken, §9) —
  explicitly refuses both: constructing a layer with either set raises AIHWKit's own
  `ConfigError("IR drop not supported in torch tile")` /
  `ConfigError("Voltage offset or R-series not supported in torch tile")`, confirmed by direct
  construction attempts. `aihwkit_backend.py::_build_rpu_config()` raises a clear `ValueError`
  if `circuit_topology` is set on that backend rather than silently ignoring it. Verified with
  unit tests on real (not mocked) CrossSim/AIHWKit analog layers - construction, and a direct
  fault write still changing the forward pass, for all three values.
  **Real performance finding, discovered while trying to verify this at full-experiment
  scale**: CrossSim's parasitics/IR-drop solver is dramatically more expensive at real crossbar
  sizes than the idealized (default) path - timed directly: a single forward pass through
  `SimpleCNN`'s `fc1` layer (512×2048, ~1M cells) took **0.002s with `circuit_topology=None`**
  but **did not complete within 120s with `circuit_topology="0T1R"` set** (same layer, same
  input). This is an inherent cost of enabling a real iterative circuit solve at this scale,
  not a bug in the wiring (already verified correct at unit-test scale, small `Linear(6,4)`
  layers, where it's instant) - but it's real, useful information for Phase 6: any real-scale
  run that sets `circuit_topology` will need to budget for this, likely via a much smaller
  crossbar/`tile_shape`, fewer patched layers, or accepting a much longer run time.
- **Weight mapping scheme**: single device per synapse (differential encoding across two
  columns, what `Scheme.DoubleColumn` already does) vs multi-device-per-synapse / bit-slicing
  for higher precision — the dissertation already names the multi-device approach as a goal;
  the code doesn't implement it yet.
  **Done.** `CrossbarConfig.devices_per_synapse: int = 1` (1 = unchanged default).
  **CrossSim**: uses its own native bit-sliced core (`params.core.style = CoreStyle.BITSLICED`,
  `params.core.bit_sliced.num_slices`) — a real, existing CrossSim feature, not custom-built.
  Real footgun found and avoided: leaving `params.core.weight_bits` at its library default of 0
  alongside `BITSLICED` produces a silent divide-by-zero `RuntimeWarning` inside CrossSim's own
  `bitsliced_core.py` — `_build_params()` always sets `weight_bits` (default 8) alongside
  `num_slices`. **AIHWKit**: wraps the resolved device class in
  `VectorUnitCell(unit_cell_devices=[device_cls() for _ in range(n)])` instead of assigning it
  directly — verified to construct under `TorchInferenceRPUConfig`, and its default
  `update_policy=ALL` already lets one `optimizer.step()` reach every constituent device with no
  extra synchronize() logic (confirmed with a real multi-step Adam probe; an initial looser
  `torch.allclose` check gave a false negative, re-verified with an explicit weight-diff before
  trusting it — not a repeat of the CrossSim training-sync bug, §9). Verified end-to-end with
  unit + integration tests and a real experiment run (`devices_per_synapse=4`).
- **Crossbar / tile size**: **done.** `CrossbarConfig.tile_shape` is wired into both working
  backends' real multi-core/multi-tile partitioning (not new math — both already supported it
  internally, just needed the right field found and connected). **CrossSim**:
  `params.core.rows_max`/`cols_max` are the actual primary controls (the previously-assumed
  `*_partition_priority` fields are only a tie-breaking heuristic for uneven splits) — and
  **these two field names are inverted relative to the axis they actually constrain**, verified
  with an asymmetric test (`cols_max=4` on a `Linear(20,16)` layer produced 4 cores of shape
  `(4,20)` — i.e. `cols_max` constrains the row/out_features dimension, `rows_max` the
  column/in_features dimension). `crosssim_backend.py::_build_params()` cross-wires this
  deliberately — do not "fix" it to look intuitive. **AIHWKit**:
  `rpu_config.mapping.max_output_size`/`max_input_size` are intuitively named (verified) and
  **already default to 512/512**, not unlimited — this backend has been silently tiling large
  layers all along. Neither backend's accessor needed any changes: CrossSim's `get_matrix()`/
  `set_matrix()` already transparently reassemble across cores, and AIHWKit's `build_handles()`
  already iterated `analog_tiles()` per-tile.
- **ADC/DAC resolution**: **done.** `CrossbarConfig.adc_resolution`/`dac_resolution` are real,
  independently-tunable config values, wired into both working backends (not just declared —
  actually consumed). CrossSim: `params.xbar.adc.mvm`/`dac.mvm` set to `QuantizerADC`/
  `QuantizerDAC` with the configured `bits`, `adc_range_option=ADCRangeLimits.MAX` (auto-derives
  the quantization range from the actual signal, no manual calibration data needed). AIHWKit:
  `TorchInferenceRPUConfig().forward.inp_res`/`.out_res` set via `res = 1/(2**bits - 2)` —
  verified this is the same formula AIHWKit's own defaults already used (its un-configured
  default is effectively a 7-bit DAC / 9-bit ADC, meaning this backend was silently already
  running non-ideal ADC/DAC before this was made an explicit, intentional, configurable value).
  See `crossbar/backends/{crosssim,aihwkit}_backend.py`'s module docstrings.
- **Redundancy granularity**: **done, as a selectable second scheme (per-cell stays default).**
  `CrossbarConfig.redundancy_scheme: "row_col_granularity"` repairs whole rows/columns at once
  via `RowColumnRedundancyPool` (`crossbar/self_healing.py`) — separate spare-row and
  spare-column capacity (`redundancy_factor` reinterpreted per-axis: fraction of rows / fraction
  of columns, not fraction of cells), gated by a new `redundancy_condemn_threshold` (fraction of
  a row's/column's own cells that must be critical before it's condemned - avoids wasting an
  entire row's capacity on one bad cell). Backend-agnostic: no changes to either backend, only
  `self_healing.py`/`mitigation/remap.py` (which dispatches on pool type so
  `mitigation/dispatch.py` needs no changes at all) /`system.py`. Verified with a real run:
  `RowColumnRedundancyPool` constructed with correct row/col capacities, and a real condemnation
  ("Condemned 0 row(s), 4 column(s)...") occurred under fault injection. Explicitly **not**
  physical spare-hardware (padding crossbar dimensions beyond the logical weight size) - that
  remains a bigger, deferred, cross-backend change; this is the granularity real designs repair
  at, using the existing logical crossbar.

### 5.3 What this buys us

Running the same mitigation strategies across multiple device technologies and crossbar
configurations turns this from "one strategy tested on one arbitrary setup" into an actual
sensitivity study — which configurations does the multi-tiered approach help most, and where
does it not matter. That's a stronger, more defensible result than a single-configuration
number, and it directly addresses the "does this generalize" question a reviewer will ask.

## 6. Alignment with research/industry standards

To make results comparable to published work rather than only internally consistent:

- **Fault models**: **done.** `FaultConfig.failure_model` (`"density"` default | `"weibull"` |
  `"exponential"`) adds real time-to-failure modeling in `faults/reliability.py` (pure math,
  independently tested) and `faults/injection.py::_statistical_stuck_at` (vectorized, no
  per-cell Python loop) — each cell's own accumulated simulated cycles
  (`HealthMonitor.cycles`/`stuck_mask`) determines its own conditional failure probability
  this interval, with an optional Arrhenius temperature-acceleration modifier
  (`arrhenius_activation_energy_ev`) on either model, per the standard JEDEC-style
  accelerated-life formula. Verified two exact invariants (not approximations): the Weibull
  CDF at `n=scale` is `1-e^-1≈0.6321` for **any** shape parameter, and the Arrhenius
  acceleration factor is exactly `1.0` when `temperature_kelvin == reference_temperature_kelvin`
  for any activation energy. Verified end-to-end with a real run: injected fault counts grew
  monotonically with cycle count (415→1261→2058→...→6152 over 8 cycles), matching the
  Weibull wear-out hazard curve, not flat or random. Deliberately kept orthogonal to the
  existing spatial `distribution` (random/clustered/gradient) option rather than combined with
  it (e.g. clustering as a hot-spot proxy feeding per-region temperature) - a real, named,
  deferred extension. `weibull_scale`/`exponential_rate` have no invented default - selecting
  a statistical model without setting them is a load-time `ValueError`, not a silent
  placeholder number, matching this doc's own standard against reporting fabricated-looking
  numbers as real.
- **Detection methodology**: **done.** `MitigationConfig.detection_method` (`"full_diff"`
  default | `"checksum"`) adds row/column checksum-based detection in `health_monitor.py` -
  a cheap first pass (reusing the pristine matrix already captured in `g_history[0]` at
  construction as the reference checksum, no new snapshot) that falls back to the exact same
  full-diff scan whenever a row/column checksum mismatches, so fault localization is identical
  whenever a fault actually exists. Honesty note recorded here since it matters: this
  simulation doesn't model real crossbar peripheral-readout circuit cost, so "cheaper" is
  reported as **modeled cell-touches** (`HealthMonitor.total_detection_cost`/
  `last_detection_cost`, surfaced via `get_health_summary()` and `experiments/run.py`'s
  `metrics.json` `detection_cost` field) rather than a fabricated wall-clock number - verified
  with a real run: cost was `rows+cols` per layer before any fault existed, then correctly
  jumped to the full `rows*cols` cost once faults were injected and checksums caught the
  mismatch. Checksums have a real, known blind spot, tested explicitly rather than glossed
  over: a "conjugate" 2×2 fault pattern (each row's and each column's sum contribution
  cancels exactly) is entirely invisible to checksum detection while `full_diff` still finds
  every cell - see `tests/unit/test_health_monitor.py`. Wiring checksum mismatches directly
  into `RowColumnRedundancyPool` (both are row/column-shaped signals) is a real, natural
  follow-up, not done here.
- **Benchmarks/datasets**: **done** (CIFAR-10 default, MNIST added). Turned out MNIST support
  already existed in `data.py` (grayscale→3-channel, resized to 32×32, normalized to match
  `SimpleCNN`'s input) - just untested and unexercised. Verified it hands-on, added test
  coverage (`tests/unit/test_data.py`) and an example experiment config
  (`experiments/configs/no_mitigation_mnist.yaml`) so it's a real, checked capability now, not
  merely present in source.
- **Baselines to report against**: unmitigated-faulted and ideal/no-fault, plus **fault-aware
  retraining — done**. Mechanism: `system.py` gains `_capture_fixed_defects()` (snapshot the
  conductance value every currently-stuck cell was set to, right after a one-time
  `inject_faults()` call) and `reapply_stuck_faults()` (re-clamp exactly those cells back to
  their fixed values, called after every `system.synchronize()` during training — necessary
  because `synchronize()` pushes the *entire* optimizer-updated weight into the analog core,
  which would otherwise silently "un-stick" a permanently-failed device; real hardware can't
  reprogram one either). `experiments/run.py` branches on the new
  `RetrainingConfig.enable_fault_aware` (nested as `ExperimentConfig.retraining`): when set, it
  injects once before training instead of periodically, and calls `reapply_stuck_faults()`
  right after `synchronize()` each step; `experiments/configs/fault_aware_retraining.yaml` mirrors
  `no_mitigation.yaml`'s fault density/distribution with all `mitigation.enable_*: false`, for a
  fair three-way comparison. This only became buildable on top of the training-sync fix above —
  training around a "known defect map" is meaningless if training doesn't reach the crossbar
  state at all. Verified: `tests/unit/test_system_retraining.py` (the re-clamp mechanism in
  isolation, `FakeCrossbar`-based) and a new CrossSim integration test
  (`test_fault_aware_retraining_pins_stuck_cells_while_others_keep_learning`) asserting all
  three load-bearing claims together — stuck-cell count never grows after the one-time
  injection, stuck cells' values stay pinned (within CrossSim's own ~1e-8 get_matrix()/
  set_matrix() float round-trip noise, unrelated to its disabled nonideality models) across
  repeated `synchronize()` calls, and non-stuck cells' values do change (real learning still
  happens around the fixed defects). Also ran all three configs end-to-end at a small smoke
  scale (3 epochs, 20 batches, 320 train samples) confirming the mechanics and metrics.json
  schema are comparable: `baseline_no_fault` 21.09% acc / 0 faults,
  `no_mitigation` 24.22% acc / 458,680 cumulative faults (periodic re-injection) / health
  degrading to 68.10%, `fault_aware_retraining` 22.66% acc / 114,670 faults (fixed after one
  injection) / health stable at 90.95%. At this tiny smoke scale no strategy has a real edge —
  a **real-scale run (Phase 6/8) is required before this comparison means anything for the
  dissertation**; what's confirmed now is that the mechanism is real and the three arms are
  honestly comparable once that run happens.
- **Reproducibility**: fixed random seeds per experiment config, versioned YAML configs checked
  into `experiments/configs/`, and every reported number traceable to a specific config + seed +
  git commit — standard practice for any result intended to be cited.
- **Reporting discipline**: only report metrics that are actually instrumented (accuracy,
  health, loss, redundancy utilization, detection/mitigation counts). Do not report estimated
  hardware costs (silicon area, GOPS, memory bandwidth) as measured results unless we build
  real instrumentation or a cited cost model for them — this was the core problem with the
  current dissertation draft's Chapter 6 and must not be repeated in the rebuilt version.
- **Citation hygiene**: dedupe the existing 138-entry reference list, fix the "Refrences"
  header typo, and keep it as the single source `CITATION.cff` and the README both point to.

## 7. Coding practices

- Dataclass/typed configs instead of raw dicts.
- `logging` instead of the ~150 scattered `print()` calls.
- Every mitigation strategy gated independently on its own `enable_*` flag, no fallthrough.
- Fault detection and mitigation operate on one canonical crossbar object per layer (kills the
  disconnected shadow-crossbar bug).
- Minimal docstrings only at public module/class boundaries — not narrating every line.
- Tests written against the *fixed* logic, encoding the actual invariants (e.g. "soft mitigation
  only touches drift-capable devices, never stuck-at ones") as assertions, not comments.

## 8. Sequencing

1. Confirm this plan.
2. Scaffold the structure + `pyproject.toml` + `uv.lock`, migrating logic while fixing known bugs.
3. Vectorize crossbar hot loops.
4. Add the device/crossbar configuration matrix (§5) as pluggable, config-driven components.
5. Local CPU smoke run at small scale to prove correctness.
6. Pick the cloud GPU provider and rent hours for a real-scale run.
7. Regenerate results from that real run, replacing `plot_creation.py`'s synthetic curves.
8. Rewrite the dissertation's results chapter from the real numbers, including honest
   comparison against fault-aware retraining as a baseline.

## 9. Open decisions (deferred, not blocking)

- Cloud GPU provider (RunPod / Vast.ai / Lambda / other) — decide when ready to rent.
- **Real bug, found and fixed: CrossSim training was a no-op for every analog-patched layer,
  for this entire session's CrossSim-backed runs, until this fix.** Verified directly:
  after `optimizer.step()`, an analog layer's `.weight` (the gradient-tracked shadow tensor)
  changes, but `get_matrix()` - what `forward()` actually computes through - does not, because
  standard torch optimizers update parameters in-place, which CrossSim's own docs state does
  not auto-propagate to the analog core. Every prior "loss decreased" observation in this
  session's CrossSim runs reflects only whatever *unpatched* layers existed (e.g. BatchNorm,
  or Conv layers in configs that only patched Linear) - the analog-patched layers themselves
  were frozen at whatever they were initialized/fault-injected to. **Fix**: `crosssim_backend.py`
  gains a `synchronize(model)` function wrapping CrossSim's own
  `simulator.algorithms.dnn.torch.convert.synchronize`; `system.py` gains a
  `FaultTolerantNeuromorphic.synchronize()` method dispatching to the active backend's
  `synchronize()` (a no-op for AIHWKit - verified separately that AIHWKit's tiles already
  reflect optimizer updates with no extra call, since it's purpose-built for hardware-aware
  training); `experiments/run.py`'s training loop calls it after every `optimizer.step()`.
  Verified fixed with a real, larger, fault-free training run: loss dropped `2.37→~1.4-1.8`,
  accuracy climbed `~7%→14-19%` over 3 epochs - genuine learning, not flat/random. This was
  found while investigating the fault-aware-retraining baseline (§6) and fixed as its own
  prerequisite, since retraining-around-known-faults is meaningless if training doesn't reach
  the crossbar state at all.
- **AIHWKit backend — added, working, via its pure-PyTorch tile.** aihwkit 1.1.0's default
  tile (`SingleRPUConfig`, backed by its compiled RPU C++ extension) is broken: every
  `AnalogLinear`/`AnalogConv2d` construction (any device preset, with or without bias) raises
  `RuntimeError: Invalid weights dimensions: expected [10,20] tensor` inside aihwkit's own
  `reset_parameters()`. Root-caused further in a follow-up session: calling the lowest-level
  tile's `set_weights()` directly (bypassing `from_digital()`/`reset_parameters()` entirely —
  constructing `AnalogTile(10, 20, rpu_config, bias=False)` and calling `.set_weights()`
  myself) reproduces the identical failure, confirming the bug lives in the compiled binding
  itself — there is no lower-level workaround. The actual fix: aihwkit ships a second, pure-
  PyTorch tile — `TorchInferenceRPUConfig` (`tile_class=TorchInferenceTile`) — that never
  touches the broken binding. Verified hands-on: `from_digital()` works for both Linear and
  Conv2d, a real forward pass computes through the tile (a written fault measurably changes
  output), and the existing device-preset classes (`PCMPresetDevice`, etc.) work unchanged
  when assigned as `rpu_config.device = preset_cls()` **after** construction (not as a
  constructor kwarg). See `crossbar/backends/aihwkit_backend.py`'s module docstring for the
  full reasoning, including the same weight-space unit-mismatch handling CrossSim needed.
- **CrossSim backend — added, working, real fault-to-accuracy forward pass.** XBTorch (the
  backend this replaced as default, see below) never read back from the crossbar's
  conductance state in its forward pass, so injected faults didn't affect model accuracy,
  only health/mitigation bookkeeping. Evaluated CrossSim (Sandia National Labs, BSD-3-Clause, pinned to the `v3.2.1`
  tag) as a second backend and verified hands-on: pure Python/NumPy/SciPy, no C++ extension to
  build (installs in seconds, unlike MemTorch/AIHWKit's packaging pain); `from_torch()`-style
  conversion of `Linear`/`Conv2d` layers into real analog equivalents whose forward pass
  genuinely computes through the programmed matrix — confirmed a fault write changes the
  model's actual output. See `crossbar/backends/crosssim_backend.py`'s module docstring for the
  unit-space handling (`get_matrix()`/`core.min`/`core.max` are weight-space, not literal
  Ohms/Siemens) and the deferred `tile_shape` → CrossSim partition-priority translation.
- **XBTorch — forward-pass wiring investigated and abandoned, then the backend removed
  entirely.** XBTorch's own module docstring admitted its forward pass never read back from
  the crossbar's conductance state (CrossSim and AIHWKit now both have this for their own
  backends, see above). A first pass found `SimpleFixedPoint` (what this backend used)
  has no compute/matmul method at all, and the documented `xbtorch.initialize()` +
  `xbtorch_model()` path sets global singleton state and requires a `.model`-as-`nn.Sequential`
  container our models don't have.

  A follow-up session read the installed package's actual source (not just its docstrings)
  and found the real differential-crossbar analog forward pass *does* exist —
  `xbtorch/patches/decorators.py::xbtorch_layer`, a class decorator implementing genuine
  Gpos/Gneg crossbar reads with DAC/ADC quantization — but it is **never applied anywhere**
  in the shipped package (not to `xbtorch.nn.Linear`/`Conv2d`, not by `xbtorch_model()` or
  `replace_all_layers_stateless`/`stateful`, all of which produce plain undecorated layers
  with no analog simulation at all). Applying the decorator manually to a local subclass and
  wiring a dedicated per-layer `SimpleFixedPoint` looked structurally promising (its
  differential-encoding writes land in the same `_chip` tensor this project's accessor
  already manipulates directly). But hands-on testing of that path immediately hit a real,
  confirmed bug in `SimpleFixedPoint` itself: constructing one with `xb_size=(40, 80)`
  produces a chip tensor of the correct shape `torch.Size([40, 80])`, but the object's own
  `.rows`/`.columns` attributes come back **transposed** (`rows=80, columns=40`) — so
  `xbtorch.deployment.mapping.map_random` (which bounds its random placement using
  `accelerator.rows`/`.columns`) computes indices that are out-of-bounds against the chip's
  *actual* first dimension, and `map_weights_to_array` crashes with a shape-mismatch error
  slicing `_chip`. This is on top of the forward pass's voltage-scaling formula
  (`gamma = torch.unique(sw_weight)[-1]`) reading as designed for WAGE-ternary-quantized
  weights, not this project's ordinary float-trained weights — a second, still-open fidelity
  question that was never reached because the rows/columns bug blocks it first.

  Conclusion: this isn't a wiring exercise on working code, it's assembling correctness out
  of a genuinely broken, seemingly never-exercised-by-any-real-user code path (three
  independent problems found: unwired dead code, a real dimension-swap bug, and an
  unverified quantization assumption). Not pursued further. With CrossSim and AIHWKit both
  working and equally easy to install, XBTorch's bookkeeping-only mode had no remaining
  unique value, so the backend was removed entirely: `crossbar/backends/xbtorch_backend.py`
  deleted, its tests deleted, `crosssim` became the new default `simulator`, and the six
  original experiment configs were repointed at it.
- Whether per-cell remapping is kept at all as an idealized upper-bound comparison, or dropped
  entirely in favor of row/column-granularity redundancy only.

## 10. Remaining work (honest audit, current as of this session)

Everything marked **done** above really is done and verified. This section exists because
"apart from fault-aware retraining, is everything else done?" deserved a real audit, not a
reflexive yes - it isn't quite everything else. What's actually still open, in one place:

- ~~Fault-aware retraining baseline (§6)~~ - **done**, see §6 above for the mechanism,
  verification, and honest caveat that a real-scale run is still required before the
  three-way comparison numbers mean anything for the dissertation.
- ~~Device technology breadth (§5.1)~~ - **done** for AIHWKit (all 6 named technologies now
  have real presets: pcm/reram_es/reram_sb/ecram plus the new fefet/sttmram). CrossSim keeps a
  real, structural limitation (one generic device model, no per-technology switching dynamics)
  - see §5.1 above for the honest caveat and the example configs that work around it at the
  conductance-range level only.
- ~~Circuit-level configuration (§5.2)~~ - **done**: `circuit_topology` (CrossSim-only - not
  supported on AIHWKit's working tile, a real verified constraint, see §5.2 above) and
  `devices_per_synapse` (both backends). All four §5.2 items are now addressed.
- **Repo scaffolding (§2)** - `README.md`, `LICENSE`, `CITATION.cff`, `Dockerfile`,
  `docs/device-configs.md` are all still missing from the target directory structure this
  plan named at the start. None block running experiments; all block a clean handoff/rental.
- **Citation hygiene (§6)** - dissertation-document work (dedupe the 138-entry reference
  list, fix the "Refrences" header typo), entirely untouched - unrelated to the codebase,
  needs doing directly in the dissertation draft.
- ~~Cloud GPU provider decision / local-Mac feasibility (§4)~~ - **done**: no GPU rental needed,
  runs locally on the M4 Mac (real per-epoch timing in §4). Also fixed this session: AIHWKit
  import aborting the process on this Mac (duplicate `libomp.dylib`, §4).
- **Phases 6-8 (§8) remaining** - the real-scale run itself (all ~12 configs, multiple seeds),
  regenerating results from it (replacing `plot_creation.py`'s synthetic curves), and rewriting
  the dissertation's results chapter from real numbers. Downstream of the code work above being
  finished, not started.

## 11. Deep correctness/validity test suite (`tests/invariants/`)

The existing `tests/unit/`+`tests/integration/` suites (100 tests before this addition) mostly
check "does this construct, does a write reach the forward pass, does training reach the
analog core" - real, but shallow. This new suite specifically targets silent logical bugs that
wouldn't crash anything but would quietly produce misleading numbers, only surfacing as "why
does this result look weird" after a long, expensive real-scale run.

**Not hypothetical - three real, verified bugs were found and fixed while building this
suite, just by asking "does this behave the way its own docstrings claim?":**

1. **Stuck cells silently self-healed.** `HealthMonitor.update_health_metrics()` recomputed
   `stress`/`health_scores` every call with no `stuck_mask` awareness - a permanently-stuck
   cell's stress decayed 1%/call toward a small steady state regardless of being dead forever.
   Verified directly: a cell injected with `health_scores=0` reached `74.73` after 200 calls.
   **Fixed**: the function now snapshots stuck-position values before its (unchanged)
   computation and restores them after - stuck cells are now provably inert to it.
2. **Soft mitigation "healed" stuck-at faults it physically cannot touch.**
   `dispatch.py::apply_runtime_mitigation()` passed the raw, unfiltered `critical_devices` list
   into `apply_soft_mitigation()` - stuck cells trivially satisfy the extreme-conductance check,
   so they were "soft-mitigated" too, despite `soft.py`'s own docstring saying this models drift
   correction, not un-sticking a hardware-stuck-at device. Verified directly: 64/73 critical
   devices at fault density 0.2 were stuck, and all 64 got mitigated anyway. **Fixed**:
   `dispatch.py` now filters `critical_devices` against `monitor.stuck_mask` before the soft
   path.
3. **Unknown/typo'd config string values silently fell back to a default, no error.**
   `build_redundancy_pool()`'s `if/else` treated any unrecognized `redundancy_scheme` as
   `idealized_per_cell`; `get_critical_devices()`'s `detection_method` had the identical
   pattern. Verified directly: `redundancy_scheme="row_col_granularityy"` (one extra letter)
   silently ran as the wrong scheme. **Fixed**: both now raise `ValueError` on an unrecognized
   value, and - going further - every config dataclass (`DeviceConfig`, `CrossbarConfig`,
   `FaultConfig`, `MitigationConfig`, `ExperimentConfig`) gained a systematic `__post_init__`
   sweep rejecting out-of-physical-range values and unknown enum strings at construction time
   (fail-fast), with the point-of-use checks kept as defense-in-depth for configs mutated after
   construction (dataclasses don't re-run `__post_init__` on attribute assignment, and this
   codebase does that in a few places, e.g. `config.fault.density = 0.5` in some tests).

**Suite structure** (`tests/invariants/`, alongside `tests/unit/`/`tests/integration/`): 80 new
tests across 6 files - `test_known_bug_regressions.py` (the 3 bugs above), `test_fault_statistics.py`
(marked `statistical` - goodness-of-fit checks via `scipy.stats.kstest`/`binomtest` that the
density/Weibull/exponential/Arrhenius models actually produce the distributions their own math
says they should, not just "some cells become stuck"), `test_physical_invariants.py`
(property-based via `hypothesis` - bounds, no NaN/Inf, redundancy capacity never negative/
double-consumed, fuzzed across many operation sequences), `test_config_validation.py` (the
systematic sweep above, 48 tests), `test_reproducibility.py` (same seed -> bit-identical
results; different seed -> different results; dataset shuffle order reproducible),
`test_documented_claims.py` (checksum blind spot generalized to many patterns via `hypothesis`,
detection-cost formula exactness, mitigation dispatch's health-threshold boundaries tested
exactly at the boundary, not just comfortably inside each branch), and `test_end_to_end_canary.py`
(marked `slow` - a real, small, fixed-seed training run: loss never NaN/Inf, loss trends
downward, final accuracy clears random chance, and a golden/canary variant with a generous
checked-in expected-range tripwire for silent numerical regressions).

Two new dev-only dependencies (`scipy`, `hypothesis` - already resolved from CrossSim's own
transitive deps in scipy's case, `pyproject.toml`'s `dependency-groups.dev`). Two new pytest
markers: `statistical` (a tiny, documented false-failure rate at a fixed significance
threshold, not a flaky retry) and `slow` (opt-in, not part of the fast default loop).

**Policy going forward**: `uv run pytest tests/ -m "not slow"` for normal iteration (fast);
`uv run pytest tests/ -q` (the full suite, including `slow`) once before trusting any real
experiment run - the actual "catch fishy shit before it burns real experiment time" checkpoint
this suite exists for. 180/180 tests passing (100 prior + 80 new), lint/format clean.

## 12. Post-experiment result validation (`neurofault.validation`, `experiments/validate_results.py`)

§11 checks the simulation's own logic before/during a run. This is the second, independent line
of defense: given a *finished* run's actual output artifacts on disk, is this something safe to
report, or does it show signs of corruption or a broken pipeline - a partially-written file from
a killed process, a config/mitigation combination that only misbehaves at real scale, a future
regression reintroducing one of §11's three bugs but only visible in a real run's *recorded
numbers*, or an experiment that quietly didn't learn anything.

**Design**: `src/neurofault/validation/results_validator.py` - pure functions over already-parsed
`metrics`/`config` dicts (no file I/O, so tests feed synthetic data directly), returning
`CheckResult(name, level, message)` at `PASS`/`WARN`/`FAIL`. Structural checks (missing keys,
mismatched list lengths, non-monotonic `batch`, `config.json` name mismatch) gate everything
else - a broken schema can't be meaningfully checked further. Then: numeric sanity (NaN/Inf,
`accuracy`/`health` in `[0,100]`, non-negative `loss`, and - the key insight - every cumulative
counter `system.py` maintains (`faults`, `soft_mitigations`, `remappings`, `layer_resets`,
`detection_cost`) must be non-negative and monotonically non-decreasing, since they only ever
accumulate; a decrease is impossible under correct code, real corruption if seen). Config-
consistency checks (FAIL) are an independent, outside-in re-verification of guarantees §11
already tests at the code level, checked instead against what a real run's artifact actually
recorded: `enable_soft_mitigation=False` (etc.) implies the corresponding metric stayed zero all
run; a zero-density baseline's `faults` stayed exactly zero; fault-aware retraining's `faults`
never grew past its first non-zero value. Qualitative/outlier checks (WARN, not FAIL - legitimate
configs can produce weak results at small scale) flag no-evidence-of-learning, non-decreasing
loss, a perfectly flat metric (the signature §9's CrossSim training-sync bug would have left),
and implausible single-step jumps.

**Marker/report**: `experiments/validate_results.py --results-dir results/<name>` (or `--all`)
writes `results/<name>/validation_report.json` and exits non-zero on `FAIL`.
`experiments/run.py::main()` calls this automatically right after writing
`metrics.json`/`config.json` - the natural moment results "come back" needs no extra manual
step for a fresh run; the CLI stays available to audit any existing/older results directory.

**Real bug found wiring this in, fixed before landing**: the glue originally lived in
`experiments/validate_results.py` and `experiments/run.py` tried `from
experiments.validate_results import validate_results_dir` - this raised `ModuleNotFoundError: No
module named 'experiments'` under this project's own established bare-script invocation style
(`uv run experiments/run.py ...`), since `experiments` is only reliably importable as a package
via its registered console script, not as a bare path. Fixed by moving the shared load/validate/
report/print logic into `src/neurofault/validation/report.py` (the properly-installed
`neurofault` package, always importable regardless of invocation style) - both
`experiments/run.py` and `experiments/validate_results.py` import from there now.

**Verification**: `tests/post_experiment/test_results_validator.py` (27 tests) constructs
deliberately-corrupted synthetic metrics/config dicts for every check above and confirms each
one actually flips the verdict (not just that a clean case passes) - plus runs the real
validator against this session's own already-generated `results/{no_mitigation,
baseline_no_fault, fault_aware_retraining}/` directories and confirms no FAIL-level check fires
on real data. Also hand-verified: the CLI run against a real `metrics.json` with a NaN injected
correctly reports FAIL and exits 1; the automatic `experiments/run.py` integration verified with
a real small-scale run (correctly WARNed "no clear evidence of learning" at that tiny scale,
neither a false PASS nor an incorrect FAIL). 207/207 tests passing project-wide (180 prior + 27
new), lint/format clean.

## 13. Mitigation strategies and fault persistence were completely non-functional (found and
fixed 2026-09-12)

Running the first real track-1 comparison (`baseline_no_fault`, `no_mitigation`,
`soft_mitigation_only`, `remapping_only`, `layer_reset_only`, `combined_all_strategies`,
`fault_aware_retraining` - all CrossSim, seed=42, full epoch counts) surfaced something a smaller
smoke test never would: `soft_mitigation_only`, `remapping_only`, and `layer_reset_only` produced
**bit-identical** loss/accuracy/health trajectories despite soft-mitigation and remapping actually
firing (34M and 71.7M cell-touches logged) and `layer_reset_only` firing zero times. This made the
whole track's central question - which mitigation strategy works best - unanswerable. Three real
bugs, found by direct probe (not code-reading alone), sharing one root cause:

- **Mitigation writes didn't survive `synchronize()`.** `mitigation/soft.py`/`remap.py` write
  corrections only via `handle.accessor.write()` (conductance). `system.py`'s training loop calls
  `system.synchronize()` after every `optimizer.step()` (every batch), which re-derives
  conductance from `.weight` - oblivious to the correction just made. Verified: a 122-cell remap
  was overwritten by exactly one real training step + `synchronize()`.
- **`layer_reset` was dead code on both backends.** `system.py` populated the bookkeeping
  `reset_fn` construction needs only for modules where `hasattr(module, "crossbars")` - an
  attribute no patched layer class in either backend (`AnalogLinear`/`AnalogConv2d` for CrossSim,
  `TorchSimulatorTile` for AIHWKit) ever actually defines. Grepped the whole `src/` tree to
  confirm: checked, never set, anywhere. `reset_fn` was `None` unconditionally, on both backends,
  regardless of config or health.
- **Stuck-at faults self-healed within one batch, in every mode except `fault_aware_retraining`.**
  Verified: injected 86 stuck cells, ran one real `optimizer.step()` + `synchronize()`, and all
  86/86 reverted from g_min/g_max to arbitrary trained values. `fault_aware_retraining` already
  had a narrow fix for exactly this (`reapply_stuck_faults()`), but nothing else called it - the
  other 8 fault-injecting configs' reported "health" degradation was real bookkeeping, decoupled
  from what the crossbar's actual conductance was doing.

**Fix**: generalized `fault_aware_retraining`'s existing pin-and-reapply pattern to every mode and
every persistent-state source, entirely at the conductance/accessor level (backend-agnostic, no
weight-space inversion needed). Two categories: *permanent* pins (stuck-at faults, updated
incrementally on every `inject_faults()` call now instead of once; remap/condemned-row-or-column
positions, already tracked permanently by `RedundancyPool`/`RowColumnRedundancyPool` - no new
state needed there) and a *one-shot* pin (soft-mitigation's corrective nudge - models drift
correction, not a permanent structural change, so it survives exactly the one `synchronize()` that
would otherwise erase it, then releases so ordinary training resumes shaping the cell). Both
reapplied inside `system.py::synchronize()` itself now, so every mode gets this automatically -
`experiments/run.py` no longer needs a `fault_aware_retraining`-specific branch for it. Separately,
the `.crossbars` check was replaced with correct membership against `self.handles` (which already
names exactly the patched layers). Re-verified all three directly against the real CrossSim
backend after the fix: remap positions 122/122 survive `synchronize()`; stuck cells 86/86 survive;
`layer_reset` fires 150/150 times in a real `layer_reset_only` run (was 0). One real, separate,
already-documented backend limitation surfaced by the `.crossbars` fix, not caused by it: AIHWKit's
`AnalogLinear`/`AnalogConv2d` wrapper modules define a `.weight` attribute that's `None` (real
weights live inside the tile), so `layer_reset` still can't construct a resettable weight there -
correctly and safely excluded now (`getattr(module, "weight", None) is not None`), not a crash.

**Why 207 existing tests missed all three, and what changes going forward**: every existing test
touching these mechanisms tested the *writer function in isolation* (`test_mitigation_remap.py`,
`test_mitigation_soft.py`, `test_mitigation_reset.py`) or the *one already-working mode*
(`test_system_retraining.py`, only ever exercised for `fault_aware_retraining`) - never the
orchestration that's supposed to invoke a mitigation, and never whether its effect survives the
next step of a real training loop. `test_mitigation_reset.py` in particular called
`apply_layer_reset()` with a hand-built closure standing in for exactly the seam
(`system.py::check_health()`'s `reset_fn` construction) where the bug lived - bypassing the one
place that could have caught it. **New rule for this project**: any mitigation/fault-persistence
feature gated by a config flag needs, beyond its isolated unit test, one test that builds the real
`FaultTolerantNeuromorphic` (a fake backend is fine - real `synchronize()` *behavior* is what
matters, so the fake must actually overwrite conductance from `.weight` like a real backend does,
not no-op), drives it through `check_health()` and a real `optimizer.step()` + `system.synchronize()`,
and asserts the claimed effect is still present afterward. A test that only calls the writer
function directly, or fakes out the exact seam it's supposed to check, doesn't satisfy this.
`tests/invariants/test_mitigation_persistence.py` (9 tests) and the rewritten
`tests/unit/test_system_retraining.py` follow this pattern now; `test_system_retraining.py`'s
assertions structurally cover every mode going forward (not just retraining), since the mechanism
itself is no longer mode-specific.

**Corrected re-run, regime 1 (original parameters)**: `no_mitigation`, `soft_mitigation_only`,
`remapping_only`, `layer_reset_only`, and `combined_all_strategies` re-run at full scale (same
seed=42, unchanged `density=0.1`/`injection_interval=8`) with the fix in place. Result: every
crossbar reaches **100.00% permanently stuck** within the first few epochs (measured directly:
`no_mitigation`'s 118 injection cycles over 945 batches saturates every layer, e.g. 1,048,570/
1,048,576 cells in `fc1`), and every arm's final accuracy collapses to chance level (7.5-10.8%,
validator WARN "no clear evidence of learning") - because no mitigation strategy can help once
redundancy/correction capacity is this thoroughly overwhelmed (a stuck cell can't be
reprogrammed by soft-mitigation or layer_reset by physical definition, and remapping's own 15%
capacity is a small fraction of what would be needed). `fault_aware_retraining` (density=0.1,
injects once, never reinjects) still learns fine (46.4% final accuracy, health 82.8%) precisely
because it avoids this dynamic entirely.

This is not a new problem this session introduced: `injection_interval=8`/`density=0.1`/these
exact epoch counts are the *same* parameters `Implementation.ipynb`'s own `main()` used for its
four arms - the original notebook's own reported results were very likely produced under a fault
model that also never truly persisted (or, per §9's MemTorch finding, never even computed
through the corrupted conductance in the first place).

**Regime 2 (recalibrated, the one to use for the mitigation comparison)**: rather than silently
keep or silently retune this, surfaced to the user, who delegated the specific recalibration
choice. Kept `density=0.1` (preserves the literature-matched per-event severity) and recalibrated
`injection_interval` per arm (315/420/210 batches depending on epoch count) so each arm gets
exactly 3 total injection cycles regardless of how many epochs it runs for - landing at a
consistent `1-0.9³ ≈ 27%` final stuck fraction (verified directly by probe: 26.9-27.7% across all
five arms), deliberately close to `remapping_only`'s own 15% redundancy capacity so its
capacity-exhaustion dynamics are actually exercised rather than either trivially sufficient or
instantly overwhelmed. New configs: `experiments/configs/{no_mitigation,soft_mitigation_only,
remapping_only,layer_reset_only,combined_all_strategies}_moderate.yaml` - the original configs are
kept unchanged (still faithfully mirror the notebook's parameters, now correctly understood to
saturate under real persistence).

Results, all five `_moderate` arms, same seed=42, all validator **PASS** (no WARN - every arm
shows real, non-chance learning this time):

| Arm | Final accuracy | Max accuracy | Final health | Mitigation actions (final cumulative) |
|---|---|---|---|---|
| `no_mitigation_moderate` | 41.70% | 47.50% | 67.08% | - |
| `soft_mitigation_only_moderate` | **45.30%** | 45.80% | 65.74% | 25.9M cell-touches |
| `remapping_only_moderate` | 40.10% | 47.90% | 65.66% | 171,942 cells remapped (≈ at its 15% capacity) |
| `layer_reset_only_moderate` | **26.20%** | 48.80% | 66.06% | 222 full-layer resets |
| `combined_all_strategies_moderate` (10 epochs, less training time) | 26.30% | 43.00% | 68.65% | 9.6M soft + 171,943 remap + 100 reset |

Soft-mitigation gives a real, modest improvement over no mitigation (+3.6pp final accuracy) -
consistent with its design as a non-destructive, surgical drift correction. Remapping is roughly
on par with no mitigation once its capacity saturates (171,942 cells remapped against a ~157K-cell
15%-of-`fc1` budget - right at capacity, as calibrated). **`layer_reset` is measurably *worse*
than no mitigation** (-15.5pp): `apply_layer_reset` restores the *entire* layer's weights back to
their random pre-training initialization (`_original_weights`, captured once at construction),
and with `health_threshold=75.0` against a trajectory that spends most of the run below that
threshold, it fires 222 times over 1257 batches (~every 5.7 batches) - far more often than a
"last resort for catastrophic failure" mitigation should, repeatedly discarding legitimate learned
weights along with whatever fault correction it provides. This is a real, honestly-reported
result of the current threshold/design, not a code bug (the dispatch logic firing it exactly when
health crosses the configured thresholds was independently verified in §11/`test_mitigation_
persistence.py`).

## 14. `layer_reset` cooldown fix and the final multi-seed comparison (2026-09-13)

Follow-up to §13's `layer_reset` finding, done in the same session per explicit user direction
("fix and shit and then re run, and get us whats best of result or real numbers we deserve").

**Fix**: added `MitigationConfig.layer_reset_cooldown_checks` (default 10) - a layer can now be
reset at most once every N `check_health()` calls (project-wide, composing with
`health_check_interval` rather than a new unrelated batch-count knob), gated in a new
`system.py::_attempt_layer_reset()` that tracks `_last_reset_check` per layer name. Regression
test added (`test_layer_reset_is_throttled_by_cooldown_not_constant`) - note `dispatch.py`
appends `"layer_reset"` to `actions_taken` whenever `reset_fn` is *called*, regardless of whether
the cooldown blocked it, so the test (and any future one) must check `outcome.layer_reset` (the
actual boolean result), not `actions_taken`, to tell a throttled attempt from a real reset. Also
added `--seed` to `experiments/run.py`'s CLI (didn't exist before - needed for the sweep below;
results land in a seed-suffixed directory, `results/<name>_seed<N>/`, so a sweep doesn't overwrite
itself).

**Single-seed verification of the cooldown fix** (seed=42, `layer_reset_only_moderate`): firing
dropped from 222 to 26 as designed, but final accuracy got *worse* (26.20% -> 9.70%) - because an
infrequent-but-still-total layer wipe can land unluckily late in training with no batches left to
recover, and a single seed can't distinguish "the fix made things worse" from "this run got
unlucky." This is exactly why a multi-seed sweep was needed before drawing any conclusion, not a
sign the cooldown fix itself was wrong.

**Multi-seed sweep**: all 5 `_moderate` arms x 3 seeds (42, 123, 7), 15 full-scale runs. Final
accuracy/health, mean across seeds with range:

| Arm | Mean final acc | Range | Mean final health | Verdict (all 3 seeds) |
|---|---|---|---|---|
| `no_mitigation` | 41.57% | 39.60-43.40% | 67.07% | PASS |
| **`soft_mitigation`** | **44.13%** | 42.20-45.30% | 65.72% | PASS |
| `remapping` | 41.90% | 40.10-44.40% | 65.71% | PASS |
| **`layer_reset`** | **11.80%** | 9.70-14.70% | 65.69% | **WARN** (all 3: "no clear evidence of learning") |
| `combined_all_strategies` | 33.07% | 31.70-34.00% | 68.51% | PASS |

This is now a *robust* finding, not single-seed noise: `no_mitigation`'s range (39.60-43.40%) and
`layer_reset`'s range (9.70-14.70%) don't even overlap across any of the 3 seeds. **The cooldown
fix (222 -> 26 firings) was necessary but not sufficient** - even 26 full-layer wipes to random
initialization, spread across the run, chronically prevents the network from ever accumulating
enough uninterrupted training to converge. `combined_all_strategies` (which also has
`enable_layer_reset=True`, firing 10 times consistently across seeds) is dragged below
`no_mitigation` (33.07% vs 41.57%) for the same reason, despite its soft-mitigation and remapping
components individually being neutral-to-positive - one destructive component can outweigh two
good ones. `soft_mitigation` shows a small, seed-consistent improvement over `no_mitigation`
(+2.56pp mean, non-destructive drift correction). `remapping` is a statistical wash (+0.33pp,
well within both arms' spread) - consistent with its capacity (~171,930 cells, matching its 15%
budget) being fully consumed under this fault burden without providing a net accuracy edge over
doing nothing.

**Honest conclusion for the dissertation's mitigation-strategy comparison**: soft-mitigation is
the only one of the three runtime strategies that clearly, robustly helps under this fault regime;
remapping is neutral once its capacity saturates; layer_reset as currently designed (whole-layer
wipe to random init) is actively harmful regardless of how rarely it's allowed to fire, because
each firing's *severity* - not just its frequency - is the problem. A cooldown throttle was the
right fix for "fires too often" (a real bug); it cannot fix "resets too much when it does fire"
(a design limitation, not a bug - the dispatch/threshold logic behaves exactly as configured).
**Left as an explicit, well-scoped decision for a future session, not resolved here**: whether to
redesign `layer_reset`'s granularity (e.g. reset only the fault-affected sub-region of a layer,
using `HealthMonitor.stuck_mask`/`get_critical_devices()` to scope the reset, instead of
`apply_layer_reset`'s current whole-layer `layer.weight.data.copy_(original_weight)`) - a
methodology change to what "layer_reset" means as a technique, not a bug fix, and therefore a
decision for the user rather than something to redesign unprompted while unsupervised.

**Verification**: full suite 214/214 passing (1 new cooldown regression test), `ruff
check`/`format --check` clean, all 15 sweep runs' `validation_report.json` read directly (not
assumed) - 12/15 PASS, 3/15 WARN (`layer_reset_only_moderate`, all 3 seeds, "no clear evidence of
learning" - an honest, correct WARN given the real numbers, not a validator bug).

**Follow-up, same day: the targeted-reset redesign, done per explicit user request.** Implemented
the granularity redesign flagged above rather than leaving it open: `apply_layer_reset()` gained a
`positions=` parameter (`src/neurofault/mitigation/reset.py`) - when given, restores only those
specific crossbar cells (and the corresponding elements of `layer.weight.data`, via an in-place
indexed assignment on a `reshape()` view - verified directly that this writes through to the
underlying parameter, not a copy) to their pre-training values, leaving every other cell's trained
value untouched. `system.py::check_health()` now passes each handle's own `critical` devices list
through to `_attempt_layer_reset()` as the target positions, instead of a full-layer wipe. 2 new
regression tests (`test_layer_reset_only_touches_critical_cells_not_the_whole_layer`,
verifying non-critical cells survive a reset that only touches deliberately-marked-critical ones).

Re-verified with the full 3-seed rerun this fix specifically needed (a single seed was exactly
what produced a misleading read for the cooldown fix earlier - not repeating that mistake here):

| | Mean final acc | Range | Verdict |
|---|---|---|---|
| `layer_reset`, whole-layer (previous) | 11.80% | 9.70-14.70% | WARN, all 3 seeds |
| **`layer_reset`, targeted (this fix)** | **19.80%** | 16.40-24.00% | **PASS, all 3 seeds** |
| `combined_all_strategies`, whole-layer (previous) | 33.07% | 31.70-34.00% | PASS |
| **`combined_all_strategies`, targeted (this fix)** | **37.93%** | 34.80-40.30% | PASS |
| `no_mitigation` (unchanged, for comparison) | 41.57% | 39.60-43.40% | PASS |

A real, substantial improvement (+8pp mean, and the validator no longer flags "no clear evidence
of learning" on any seed) - but **`layer_reset` is still net-negative versus doing nothing**,
even redesigned. Root cause, checked directly rather than assumed: `get_critical_devices()`'s
criteria (health score, failure probability, extreme conductance - not just `stuck_mask`) flag a
much larger fraction of cells "critical" than the ~27% permanently-stuck target as training
progresses and more cells drift/degrade, so a "targeted" reset late in a run still touches roughly
half the layer (measured directly: `fc1`'s targeted resets ranged 494,360-573,206 cells out of
1,048,576, i.e. ~47-55%, in the last several resets of the seed=42 run) - not the small, surgical
correction the fix was aiming for, just a smaller sledgehammer than before. Fixing this further
(e.g. a stricter/separate criticality threshold specifically for reset eligibility, distinct from
the general health-monitoring threshold shared with soft-mitigation/remap detection) is a real,
identifiable next step, deliberately not chased further in this session - the requested fix
(reset the fault-affected region, not the whole layer) is implemented, verified, and a genuine
improvement; tightening exactly how "fault-affected" is scoped is a further, separate design
question for a future session.

## 15. "Other chips" campaign: FeFET, STT-MRAM, AIHWKit, MNIST (2026-09-13)

Extended the fixed pipeline + moderate fault regime beyond the core CIFAR-10/CrossSim/default-
device track, per explicit user request, budgeted to a ~6-hour autonomous window (full plan and
scope tradeoffs at the time: a saved planning artifact, not reproduced here). 10 new device-tech
configs (`experiments/configs/device_{fefet,sttmram}_{no_mitigation,soft_mitigation,remapping,
layer_reset,combined}_moderate.yaml`) plus 2 backend/dataset baselines
(`no_mitigation_aihwkit_moderate.yaml`, `no_mitigation_mnist_moderate.yaml`), all recalibrated to
the same ~27%-final-stuck-fraction moderate regime as §13's core track.

**Real, honest finding, not a bug**: STT-MRAM's literature-sourced `r_on=100`/`r_off=10000` (no
canonical domain-wall MTJ ratio exists in the checked literature - see
`device_sttmram_crosssim.yaml`'s own sourcing note) are numerically identical to
`DeviceConfig`'s own defaults (`src/neurofault/config.py`), and CrossSim has no per-technology
switching-dynamics model - only `r_on`/`r_off` matter (§5.1, already documented before this
session). So on CrossSim, "STT-MRAM" and "the core track's default device" are the literal same
configuration - confirmed by `device_sttmram_*_moderate_seed42`'s results being bit-identical to
the corresponding core-track arms already in hand. Caught mid-campaign (should have been checked
before launching the STT-MRAM runs, not after - a real process miss, noted so it doesn't repeat):
the 4 STT-MRAM Tier-3 reruns (extra seeds for `no_mitigation`/`soft_mitigation`) were killed
within seconds of starting once this was confirmed, rather than let ~35-40 minutes of genuinely
redundant compute run to completion. STT-MRAM's numbers below are therefore seed=42 only,
explicitly understood to equal the core track's seed=42 (not independent data) - **`device_
sttmram_*` should be read as "the same experiment, differently labeled" until CrossSim gains
actual per-technology device dynamics, not as real STT-MRAM-specific evidence.**

**FeFET results** (genuinely distinct: `r_off=4500`, a 45x dynamic range vs. the default's 100x):

| Arm | seed=42 | 3-seed mean (no_mitigation/soft_mitigation only) | Range |
|---|---|---|---|
| no_mitigation | 40.60% | 42.87% | 40.60-46.10% |
| soft_mitigation | 42.60% | 41.60% | 40.80-42.60% |
| remapping | 41.30% | - | - |
| layer_reset (targeted) | 13.30% (WARN) | - | - |
| combined | 39.50% | - | - |

Mostly consistent with the core track (`remapping` roughly neutral, `layer_reset` still behind
- and even worse here on this one seed, though within the already-known 16.40-24.00% variance
range for this arm, not investigated as a separate anomaly). One genuine divergence worth flagging
rather than smoothing over: FeFET's 3-seed means show `soft_mitigation` (41.60%) *not* clearly
beating `no_mitigation` (42.87%), unlike every other device/backend tested this session where it
did. The ranges overlap substantially (3 seeds isn't enough to call this conclusively either way),
but it's a real, honestly-reported open question - plausibly FeFET's narrower 45x conductance
range interacts differently with soft-mitigation's fixed `blend=0.3`-toward-`g_mid` correction
than the wider-range devices tested elsewhere - not chased further this session.

**AIHWKit** (`no_mitigation_aihwkit_moderate`, seed=42): 54.60% final accuracy, PASS - notably
higher than CrossSim's ~41-42% no_mitigation baseline. Single run, single backend comparison point
only - AIHWKit's real analog-tile fault/noise model differs fundamentally from CrossSim's more
idealized one, so this isn't an apples-to-apples "AIHWKit mitigates faults better" claim, just
confirmation the fixed pipeline and moderate regime run cleanly on this backend too.

**MNIST** (`no_mitigation_mnist_moderate`, seed=42): 86.70% final accuracy (max 91.2%), **WARN**
(`outlier.accuracy_jump`: "accuracy jumped by more than 50 points in one step, max jump=72.20").
Checked directly, not assumed: this is genuine, expected MNIST behavior (loss 2.33 -> 0.26, a
much easier task than CIFAR-10 that a small CNN converges on within a handful of gradient steps),
not a data-corruption signature - the validator's outlier check is correctly doing its job
(flagging something legitimately unusual for a human to glance at), not incorrectly flagging a
bug. Exactly the "legitimate configs can produce weak/unusual results, WARN not FAIL" case §12's
own design anticipated.

**Not covered this session** (explicitly, for a future one): AIHWKit's and MNIST's full 5-arm
mitigation comparison (only their `no_mitigation` baseline was run); more than 3 seeds anywhere;
`remapping`/`layer_reset`/`combined` multi-seed confidence intervals on FeFET (only `no_mitigation`/
`soft_mitigation` got the extra seeds, per the budgeted plan's own prioritization); CrossSim
gaining actual per-technology device switching dynamics (the structural reason STT-MRAM and the
default device are indistinguishable here).

**Verification**: every new config validated to load before the campaign launched; every result
above read directly from its `validation_report.json` and `metrics.json` (not assumed); full
`uv run pytest tests/ -q` (215/215) and `ruff check`/`format --check` clean, re-run after the full
campaign, not just after the code changes earlier in the day.

## 16. Phase 1a: base-model realism investigation - every training-recipe and capacity lever tried, honest ceiling found (2026-09-13)

The "Catching Up" plan's §1a flagged our fault-free baseline (43.30% on CIFAR-10, from
`train_samples=2000`) as far below realistic hardware accuracy (IBM's fabricated analog chip:
92.81%, cited in the plan). Per explicit user instruction ("throw more things in there that can
absolutely increase accuracy... whatever we can do absolutely"), every reasonable
architecture-independent and architecture-dependent lever was tried in sequence, each verified
empirically before moving to the next rather than assumed to help - full iteration log, using
`experiments/configs/baseline_no_fault_scaled_check.yaml` (density=0.0, fault-free, isolates
model capacity from fault-tolerance) throughout:

| Change | final acc | max acc |
|---|---|---|
| Original (2000 samples, no aug, 15 epochs, old arch) | 43.30% | 47.30% |
| +8x data (16000 samples), no aug, 10 epochs, old arch | 49.50% | 52.55% |
| + augmentation (RandomCrop+Flip), same 10 epochs, old arch | 49.75% | 51.90% |
| + weight_decay=1e-4 + cosine LR schedule + 30 epochs, old arch | 49.25% | 52.65% |
| + widened arch (32/64/128->48/96/192 channels, fc1 512->768, ~2.25x cells) | **51.70%** | **55.55%** |

Two real findings along the way, not just numbers:
- `num_batches` (not `train_samples`) was the actual binding constraint on how much data a run
  saw per epoch - raising `train_samples` 8x alone would have silently used the same amount of
  data per epoch as before, capped by the old `num_batches=128`. Fixed by raising `num_batches` to
  600 (16000/32=500 batches/epoch).
- Augmentation showed literally no benefit at the same epoch count (49.75% vs. 49.50%) - it trades
  early convergence speed for later generalization, so testing it at a fixed short epoch count
  made it look useless. Confirmed correct once epochs were also raised together with weight_decay
  and a cosine schedule.

**Honest conclusion**: every architecture-independent lever (8x data, augmentation, weight_decay,
cosine LR, 3x epochs) plateaued at ~49-53%, essentially flat regardless of which of those were
combined - none of them individually or together closed the gap. Only a genuine capacity increase
(widening the CNN) produced a real, distinct improvement (+2.45pp final, +2.9pp max over the best
training-recipe-only result). This means the bottleneck was **model capacity**, not training
procedure - a small custom CNN (even well-regularized, well-scheduled, well-augmented) cannot
reach IBM's reported hardware accuracy on this dataset; IBM's number comes from a substantially
larger/deeper model trained on the full 50,000-image dataset, not a training-recipe difference.

**What was not tried, and why, rather than silently left out**: the full 50,000-sample dataset
(this session's checks used 16,000, 32% of it, to keep each check's turnaround under CPU-only
constraints) and a deeper/differently-structured architecture (this session only widened the
existing 3-conv-layer design, did not add depth or residual connections). Both are real remaining
levers, explicitly deferred rather than exhausted, because they represent a materially larger time
investment (full-dataset epochs take ~3x longer each; a deeper architecture is a design change,
not a config toggle) that crosses from "verify within an autonomous window" into "a scope decision
worth checking in on" - consistent with this project's standing plan-before-large-effort norm.
**51.70%/55.55% is the honest ceiling reached this session with the current architecture family on
32% of the dataset - a real, verified, but not final number.**

Full test suite (`uv run pytest tests/ -q`, 220/220) and lint (`ruff check`/`format --check`)
re-verified clean after the widened-architecture change, before this section was written.

## 17. Killing wasted health-check compute, real architectural depth, and the GPU decision (2026-09-14)

51.70%/55.55% (§16) wasn't good enough - correctly called out as "basically a coin flip." Rather
than tune further, this pushed on the two real remaining levers: fix a genuine performance bug
first (free, CPU-only), then add real depth, then make an honest, measured GPU decision instead of
a guessed one. Full plan: a saved planning artifact (not reproduced here), approved before any of
this work started.

**Step 1 - a real, verified performance bug, not a hypothetical one.** Profiling `check_health()`
(`system.py:215`, cProfile on a 20-batch smoke run) showed it costing 0.585s/call even on a tiny
config - and a direct probe confirmed why: on the widened, completely fault-free architecture,
`get_critical_devices()`'s heuristic (health score / failure probability / extreme conductance)
flagged **261,889-337,546 of `fc1_crossbar_0`'s 2.36M cells (11-14%) "critical" with zero faults
injected anywhere.** `system.py` and `dispatch.py` then each ran a Python list-comprehension over
that list, indexing `monitor.stuck_mask[row, col]` one tuple at a time - pure waste, since every
mitigation action function (`apply_soft_mitigation`, `apply_remap`, `apply_layer_reset`'s
`positions=[]` path, `reset.py:86-87`) already no-ops on an empty position list, confirmed by
reading each one directly. Fix: skip the entire per-layer dispatch pipeline in `check_health()`
when `not monitor.stuck_mask.any() and not monitor.drift_mask.any()` - behavior-preserving (full
suite stayed 220/220 green after), performance-only. **Verified, not assumed: clean isolated
per-batch timing went from 601.6ms/batch to 181.7ms/batch - a 3.3x speedup** - and the *full test
suite's own* wall time dropped 231.95s -> 144.72s -> 68.83s across this session's edits, since
other tests exercise the same path. Directly validates this project's own pre-existing philosophy
(§4: "vectorize before renting anything... a GPU will not speed up... an interpreter bottleneck").

**Step 2 - real depth, not more width.** `SimpleCNN` (`src/neurofault/models/cnn.py`) went from 3
conv layers (each widened in §16) to 6 - a VGG-style 2-conv-layers-per-stage structure
(`conv1a/1b`, `conv2a/2b`, `conv3a/3b`), same final 4x4/3072-feature size so this only adds
capacity, not a head redesign. Verified in order, not assumed to work: a 3-batch smoke run
confirmed crossbar-patching handles all 8 resulting crossbars cleanly; a real 5-epoch/16k-sample
trial confirmed it actually learns (loss noisy-but-bounded 1.3-2.2, accuracy climbing to
43.65%/44.85% - not a fair ceiling comparison to §16's numbers since the cosine schedule was
compressed to match the 5-epoch override, but that wasn't the point of this check).

**Step 3 - the GPU decision, made from a real measurement, not a guess.** Timed one complete real
epoch at the actual full target scale (50,000 samples, deeper architecture, real `eval_interval`
overhead included) directly via `experiments/run.py`, not extrapolated from a partial sample:
**1727s (28.8 min) for one epoch.** Extrapolated to the ~30 epochs this session's convergence
behavior has consistently needed: **~14.4 hours** - over the plan's own pre-agreed "stay on CPU
under ~8-10 hours" threshold. Note on measurement conditions: this session's Mac was under real,
externally-caused load (uptime 26 days, load average 20-38, ~69MB free RAM at points) from
unrelated long-running processes - this added noise to shorter timing probes but the real
full-epoch measurement used for this decision is a direct, complete, wall-clock timing of the
actual target workload, not a noisy extrapolation.

**GPU readiness work done, verified as far as locally possible:**
- `CrossbarConfig.use_gpu: bool = False` (`config.py`) wired into `crosssim_backend.py`'s
  `_build_params()` as `params.simulation.useGPU` - a real, pre-existing CrossSim mechanism
  (`simulator/parameters/simulation_parameters.py`, `backend/backend.py`'s `ComputeBackend`
  singleton, which transparently swaps CrossSim's whole array module between numpy and cupy), not
  something built from scratch. Default `False`: zero behavior change on this CPU-only Mac.
- New `gpu` extra in `pyproject.toml` (`cupy-cuda12x>=13.0`, resolved into `uv.lock`) - only
  needed on an actual CUDA box.
- New provider-agnostic `Dockerfile` (CUDA 12.4.1 runtime base, `uv`-managed Python 3.12 rather
  than apt's - verified directly that Ubuntu 22.04's default repos don't carry a `python3.12`
  package at all, caught by actually trying the build, not assumed to work).
- **Verified twice via a real `docker build --platform linux/amd64`** (using a local colima VM,
  since this Mac has no Docker Desktop/daemon by default): every dependency resolves and installs
  cleanly - torch pulls in the full NVIDIA CUDA runtime stack (cudnn, cublas, nvjitlink, nccl,
  etc.) automatically for this platform+extras combination, alongside `cupy-cuda12x` and CrossSim,
  with no conflicts. The build's *final* step (writing torch/cudnn's several-GB of shared libraries
  into the image layer) hit local colima VM disk-partition limits (its docker storage mount capped
  at 20G regardless of the `--disk` flag) - confirmed this is a local virtualization quirk, not a
  Dockerfile defect, and not chased further since real cloud GPU instances ship far more disk than
  this local test VM by default. **Not fully verified: actual GPU-accelerated execution** - this
  session has no CUDA hardware, so the `useGPU=True` path's real speedup and correctness must be
  smoke-tested on the actual rented box (a few batches, not the full run) before trusting a long
  training run there, per this project's own verify-before-committing-compute norm.

**Recommendation given to the user**: rent one mid-range CUDA GPU (RTX 3090/4090-class - this
workload is a moderate CNN plus CrossSim's array-simulation overhead, not a large transformer, so
no need for A100/H100-class hardware, matching §4's existing "no need to plan for multi-GPU or
datacenter-class cards" call) from RunPod or Vast.ai, smoke-test the Docker image there first, then
run the real full-scale (50,000-sample, deeper-architecture) baseline. Actually creating and
funding a cloud account is the one part of this only the user can do.

Full test suite (220/220) and `ruff check`/`format --check` re-verified clean after all of
Step 1-3's code changes, before this section was written.
