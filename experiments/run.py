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

    optimizer = optim.Adam(system.model.parameters(), lr=config.learning_rate)
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
    }

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

            if config.fault.density > 0 and global_batch % config.fault.injection_interval == 0:
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
                logger.info(
                    "epoch=%d batch=%d loss=%.4f accuracy=%.2f%% health=%.2f%%",
                    epoch,
                    global_batch,
                    loss.item(),
                    accuracy,
                    report["avg_health"],
                )

            global_batch += 1

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
    args = parser.parse_args()

    config = load_config(args.config)
    overrides = {
        k: v
        for k, v in {
            "epochs": args.epochs,
            "num_batches": args.num_batches,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
        }.items()
        if v is not None
    }
    if overrides:
        config = replace(config, **overrides)

    metrics = run_experiment(config)

    output_dir = Path(config.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
    plot_single_run(metrics, output_dir / "plots.png", title=config.name)

    logger.info("Results written to %s", output_dir)


if __name__ == "__main__":
    main()
