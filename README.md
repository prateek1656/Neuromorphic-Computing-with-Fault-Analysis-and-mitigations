# neurofault

Fault-tolerant memristive crossbar neuromorphic computing simulation. This is the code behind
the M.Tech dissertation *"Design and Analysis of Fault-Tolerant Memristive Crossbar for
Neuromorphic Computing"* (Jawaharlal Nehru University, May 2025) — rebuilt from an
originally-broken, never-fully-executed codebase into something that actually runs end-to-end
and produces real, reproducible numbers. See [Citation](#citation) and
`docs/planning/project-setup-plan.md` for the full design/verification log, including the
original bugs found and fixed along the way.

## What it does

A `FaultTolerantNeuromorphic` system (`src/neurofault/system.py`) wraps a CNN's weights in
simulated memristive crossbar arrays, injects faults into those crossbars during training
(stuck-at, drift, and runtime conductance faults — `src/neurofault/faults/`), and evaluates how
well different runtime mitigation strategies preserve accuracy as faults accumulate:

| Strategy | What it does | Config flag |
|---|---|---|
| `no_mitigation` | Baseline — faults persist, nothing corrects them | (all disabled) |
| `soft_mitigation` | Small in-place nudge toward the reference weight for flagged cells | `enable_soft_mitigation` |
| `remapping` | Reroutes flagged weights to spare redundant cells (capacity-limited) | `enable_remapping` |
| `layer_reset` | Targeted reset of only the fault-affected cells/weights (last resort, cooldown-throttled) | `enable_layer_reset` |
| `combined_all_strategies` | All three above, dispatched together | all three |
| `fault_aware_retraining` | Bakes a known defect map in *before* training instead of reacting at runtime | `RetrainingConfig` |

A `HealthMonitor` (`src/neurofault/crossbar/health_monitor.py`) periodically checks crossbar
health (`full_diff` or the cheaper adaptive `checksum` method) and drives which cells get flagged
for mitigation.

### Devices and backends

- **Backends**: [CrossSim](https://github.com/sandialabs/cross-sim) (pure Python/NumPy, no build
  step) and [AIHWKit](https://github.com/IBM/aihwkit) (IBM's analog hardware toolkit) —
  `src/neurofault/crossbar/backends/`. Both run fine CPU-only; CrossSim also supports a CUDA/CuPy
  GPU path (`gpu` extra).
- **Device technologies**: VTEAM (default), FeFET, and STT-MRAM
  (`src/neurofault/devices/presets.py`, `registry.py`) — FeFET and STT-MRAM presets are custom
  (AIHWKit doesn't ship them itself).
- **Datasets**: CIFAR-10 (default) and MNIST (`src/neurofault/data.py`).

## Setup

This project uses **uv** exclusively for Python and dependency management — never a system/conda
Python or bare `pip`. `uv sync` creates and manages an isolated `.venv/` in the repo root
(pinned to Python 3.12 via `.python-version`); it does not touch any system or conda
interpreter, and works correctly regardless of whether a conda env happens to be active in your
shell.

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it.
2. Sync dependencies into `.venv/`:

   ```bash
   uv sync --extra crosssim --extra aihwkit
   ```

   Add `--extra gpu` on a CUDA box that needs CrossSim's GPU (CuPy) backend — not installable
   and not needed on a CPU-only machine (see `pyproject.toml`'s `gpu` extra).

3. Run everything through `uv run` (this activates `.venv/` for the command, no manual
   `source .venv/bin/activate` needed and no risk of it silently falling through to a system
   or conda interpreter):

   ```bash
   uv run pytest tests/ -v
   uv run ruff check src experiments tests
   uv run ruff format --check src experiments tests
   uv run experiments/run.py --config experiments/configs/no_mitigation.yaml
   ```

Do not `pip install` anything directly (system, conda, or otherwise) — every dependency this
project needs is declared in `pyproject.toml` / locked in `uv.lock`. If a new dependency is
needed, add it with `uv add <package>` (or under the right extra in `pyproject.toml`) and commit
the resulting `uv.lock` change, rather than installing it ad hoc.

### Docker

`Dockerfile` builds a CUDA image the same way (`uv sync --extra crosssim --extra gpu`) for
full-scale runs on rented GPU compute — see its comments for what's intentionally excluded
(AIHWKit) and why.

## Running experiments

```bash
uv run experiments/run.py --config experiments/configs/no_mitigation.yaml
# quick smoke run, overriding scale:
uv run experiments/run.py --config experiments/configs/no_mitigation.yaml \
    --epochs 1 --num-batches 2 --train-samples 64 --test-samples 32
```

`experiments/configs/` holds one YAML per arm — the core comparison
(`no_mitigation`/`soft_mitigation_only`/`remapping_only`/`layer_reset_only`/
`combined_all_strategies`), a `_moderate` fault-regime variant of each (see the plan doc §13 for
why the original fault density saturates every crossbar), per-device (`device_fefet_*`,
`device_sttmram_*`) and per-backend (`*_crosssim`, `*_aihwkit`) variants, and
`fault_aware_retraining`. Each run writes to `results/<name>/` (gitignored — regenerate rather
than expect it in a fresh clone) and is auto-validated at the end; check any run's
`validation_report.json` (or run `uv run experiments/validate_results.py --results-dir
results/<name>` / `--all` standalone) before trusting its numbers.

## Testing

Two independent layers, both required to trust a change:

- `tests/unit/`, `tests/integration/` — standard correctness tests against real backend calls
  (not mocks at the seam being tested — see `docs/planning/project-setup-plan.md`'s "why tests
  missed it" note on why that distinction mattered here).
- `tests/invariants/` — deep statistical/property-based correctness suite (goodness-of-fit tests,
  `hypothesis` fuzzing, a documented "known bug regressions" suite, real end-to-end training
  canaries). Marked `slow`/`statistical` where relevant — see `pyproject.toml`'s pytest markers.
- `tests/post_experiment/` + `neurofault.validation` — the results-directory sanity gate
  described above.

```bash
uv run pytest tests/ -v
```

## Repo layout

- `src/neurofault/` — package: crossbar simulation, fault injection, mitigation strategies,
  device configs.
- `experiments/` — run configs (`experiments/configs/*.yaml`), the `run.py` CLI entrypoint, and
  `validate_results.py`.
- `tests/unit/`, `tests/integration/`, `tests/invariants/`, `tests/post_experiment/` — see above.
- `docs/planning/project-setup-plan.md` — the running design/verification log; read this first
  for any "why" question not answered by code comments.
- `Implementation.ipynb` — the original working notebook `experiments/run.py`'s training loop,
  results schema, and plotting were ported from; kept as the proven reference it was verified
  against.
- `dissertation-5-merged.pdf` — the dissertation itself.

## License

MIT — see `LICENSE`.

## Citation

See `CITATION.cff` for the full software and preferred (thesis) citation metadata.
