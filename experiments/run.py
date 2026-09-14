"""CLI entrypoint for running one experiment arm.

Ports the training loop, results-dict schema, and plotting pattern from
Implementation.ipynb cells 5-6 - the only code in this repo proven to have
actually run against real data - rather than the never-executed main.py/
experiment.py. plot_creation.py's synthetic curve generator is not ported at
all: it's the confirmed source of the dissertation's fabricated numbers.

Usage:
    uv run experiments/run.py --config experiments/configs/no_mitigation.yaml
    uv run experiments/run.py --config experiments/configs/no_mitigation.yaml \\
        --epochs 1 --num-batches 2 --train-samples 64 --test-samples 32
"""

from __future__ import annotations

import os

# Must precede the torch/sklearn imports below: both bundle their own
# libomp.dylib, and macOS aborts the process on the second dlopen unless this
# is set (aihwkit's import chain pulls in sklearn). See
# docs/planning/project-setup-plan.md §4.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import optim

from neurofault.config import ExperimentConfig, load_config
from neurofault.data import load_datasets
from neurofault.models.cnn import SimpleCNN
from neurofault.system import FaultTolerantNeuromorphic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_experiment(config: ExperimentConfig) -> dict:
    torch.manual_seed(config.seed)

    trainloader, testloader = load_datasets(config)
    base_model = SimpleCNN()
    system = FaultTolerantNeuromorphic(base_model, config)

    optimizer = optim.Adam(
        system.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.lr_cosine_schedule
        else None
    )
    criterion = torch.nn.CrossEntropyLoss()

    metrics = {
        "epoch": [],
        "batch": [],
        "loss": [],
        "accuracy": [],
        "health": [],
        "faults": [],
        "soft_mitigations": [],
        "remappings": [],
        "layer_resets": [],
        "detection_cost": [],  # modeled cell-touches (see health_monitor.py) - the
        #   honest number for comparing mitigation.detection_method="checksum" against
        #   "full_diff" (rows*cols per layer per check, always, for full_diff).
    }

    if config.retraining.enable_fault_aware:
        # Bake a fixed, known defect map in once, before any training - the
        # dominant literature baseline (see docs/planning/project-setup-plan.md
        # §6): gradient descent then learns weights that compensate for these
        # specific, unchanging faults, instead of reacting to faults at
        # runtime. No periodic re-injection below for this mode. Pinning the
        # stuck cells so they survive synchronize() is now automatic (see
        # system.py's inject_faults()/synchronize()) - no separate capture call.
        system.inject_faults()

    global_batch = 0
    for epoch in range(config.epochs):
        system.model.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx >= config.num_batches:
                break

            optimizer.zero_grad()
            outputs = system.forward(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            system.synchronize()  # pushes the update into the analog compute path, then
            #   re-pins stuck faults/mitigation corrections it would otherwise erase -
            #   without this, training silently never reaches backends that need it, and
            #   faults/mitigations silently self-heal within one batch (see
            #   system.py::synchronize()'s docstring - real bugs, not defensive code)

            if (
                not config.retraining.enable_fault_aware
                and global_batch % config.fault.injection_interval == 0
            ):
                system.inject_faults()

            if global_batch % config.eval_interval == 0:
                accuracy = _evaluate(system, testloader)
                report = system.get_health_report()
                metrics["epoch"].append(epoch)
                metrics["batch"].append(global_batch)
                metrics["loss"].append(loss.item())
                metrics["accuracy"].append(accuracy)
                metrics["health"].append(report["avg_health"])
                metrics["faults"].append(report["total_faults_injected"])
                metrics["soft_mitigations"].append(report["soft_mitigations"])
                metrics["remappings"].append(report["remappings"])
                metrics["layer_resets"].append(report["layer_resets"])
                metrics["detection_cost"].append(
                    sum(s["detection_cost"] for s in report["layer_health"].values())
                )
                logger.info(
                    "epoch=%d batch=%d loss=%.4f accuracy=%.2f%% health=%.2f%%",
                    epoch,
                    global_batch,
                    loss.item(),
                    accuracy,
                    report["avg_health"],
                )

            global_batch += 1

        if scheduler is not None:
            scheduler.step()

    return metrics


def _evaluate(system: FaultTolerantNeuromorphic, testloader) -> float:
    system.model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            outputs = system.model(inputs)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    system.model.train()
    return 100.0 * correct / total if total > 0 else 0.0


def plot_single_run(metrics: dict, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(metrics["batch"], metrics["accuracy"])
    axes[0, 0].set(title="Accuracy", xlabel="Batch", ylabel="Accuracy (%)")

    axes[0, 1].plot(metrics["batch"], metrics["health"])
    axes[0, 1].set(title="Health", xlabel="Batch", ylabel="Health (%)")

    axes[1, 0].plot(metrics["batch"], metrics["loss"])
    axes[1, 0].set(title="Loss", xlabel="Batch", ylabel="Loss")

    axes[1, 1].plot(metrics["batch"], metrics["faults"], label="Faults")
    axes[1, 1].plot(metrics["batch"], metrics["soft_mitigations"], label="Soft mitigations")
    axes[1, 1].plot(metrics["batch"], metrics["remappings"], label="Remappings")
    axes[1, 1].plot(metrics["batch"], metrics["layer_resets"], label="Layer resets")
    axes[1, 1].set(title="Faults and mitigations", xlabel="Batch", ylabel="Count")
    axes[1, 1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--test-samples", type=int)
    parser.add_argument("--simulator", choices=["crosssim", "aihwkit"])
    parser.add_argument("--dataset", choices=["cifar10", "mnist"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    overrides = {
        k: v
        for k, v in {
            "epochs": args.epochs,
            "num_batches": args.num_batches,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "simulator": args.simulator,
            "dataset": args.dataset,
            "seed": args.seed,
        }.items()
        if v is not None
    }
    if args.seed is not None:
        # Distinct results/<name>/ per seed - otherwise a seed sweep would
        # have every run overwrite the same directory.
        overrides["name"] = f"{config.name}_seed{args.seed}"
    if overrides:
        config = replace(config, **overrides)

    metrics = run_experiment(config)

    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    plot_single_run(metrics, output_dir / "plots.png", title=config.name)

    logger.info("Results written to %s", output_dir)

    # Post-experiment sanity gate (see docs/planning/project-setup-plan.md) -
    # the natural moment results "come back" is exactly when a run finishes,
    # so this needs no extra manual step. Imported from neurofault.validation
    # (an installed package), not experiments.validate_results - verified
    # directly that `experiments` itself isn't reliably importable under
    # this project's own bare-script invocation style (`uv run
    # experiments/run.py ...`), only under its registered console script.
    from neurofault.validation.report import validate_results_dir

    validate_results_dir(output_dir)


if __name__ == "__main__":
    main()
