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
- `memtorch` — pinned to a specific git commit/tag, not an unpinned `pip install`, so the
  environment is actually reproducible (this is currently broken in the repo).
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

### 5.2 Circuit-level configurations

- **1T1R** (transistor + resistor per cell) vs **0T1R** (bare crossbar) vs **1S1R**
  (selector + resistor) — these change sneak-path behavior and are standard configuration
  choices in real crossbar designs; should be selectable, not assumed away.
- **Weight mapping scheme**: single device per synapse (differential encoding across two
  columns, what `Scheme.DoubleColumn` already does) vs multi-device-per-synapse / bit-slicing
  for higher precision — the dissertation already names the multi-device approach as a goal;
  the code doesn't implement it yet.
- **Crossbar / tile size**: realistic fabricated-array sizes (commonly 64×64, 128×128, up to
  1024×1024 with tiling) rather than one implicit size baked into a specific experiment.
  `tile_shape` already exists as a parameter in MemTorch — it should be exposed as a first-class
  experiment axis, not left at its default.
- **ADC/DAC resolution**: currently hardcoded at 8-bit. Real designs commonly explore 4–8 bit
  resolution (this tradeoff is a recurring theme in the crossbar accelerator literature, e.g.
  ISAAC/PRIME-style architectures) — should be a config parameter.
- **Redundancy granularity**: the current per-cell remapping has no physical hardware analogue.
  Real designs repair at **row/column (word-line/bit-line) granularity** via the peripheral
  decoder — that should become the default, physically-grounded redundancy scheme, with
  per-cell remapping kept (if at all) as an explicitly-labeled idealized upper bound, never
  presented as equivalent to hardware redundancy.

### 5.3 What this buys us

Running the same mitigation strategies across multiple device technologies and crossbar
configurations turns this from "one strategy tested on one arbitrary setup" into an actual
sensitivity study — which configurations does the multi-tiered approach help most, and where
does it not matter. That's a stronger, more defensible result than a single-configuration
number, and it directly addresses the "does this generalize" question a reviewer will ask.

## 6. Alignment with research/industry standards

To make results comparable to published work rather than only internally consistent:

- **Fault models**: keep using the statistical injection models the dissertation already names
  (Weibull for programming-cycle-count failures, exponential for voltage stress, Arrhenius for
  temperature effects) but actually implement them — today fault injection is a flat
  density-based random/clustered/gradient pattern with no temporal statistical model behind it.
- **Detection methodology**: the current approach (read back every cell, diff against a stored
  reference) is the expensive, naive version of what the field does. Real work favors
  algorithm-based fault tolerance (checksum vectors through the matrix multiply, O(n) overhead)
  or signature-based detection (power/current changepoint analysis). At minimum, implement one
  checksum-based detector as an option and report its overhead honestly against the full-diff
  approach, rather than defaulting to the expensive method without acknowledging the tradeoff.
- **Benchmarks/datasets**: keep CIFAR-10 (already used, and it's the standard mid-complexity
  benchmark in this literature) as the default; consider adding MNIST as a lighter/faster
  sanity-check benchmark for quick iteration during development, matching common practice of
  reporting on more than one dataset scale.
- **Baselines to report against**: unmitigated-faulted and ideal/no-fault are already present;
  add a comparison point against **fault-aware retraining** (the dominant, most empirically
  successful approach in the literature — near-full accuracy recovery at ~20% defect rates via
  retraining with a known defect map). Even if our runtime, no-retraining approach doesn't beat
  it, reporting that comparison honestly is what the field expects and is the actual bar a
  reviewer will hold this work to.
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
- Whether per-cell remapping is kept at all as an idealized upper-bound comparison, or dropped
  entirely in favor of row/column-granularity redundancy only.
