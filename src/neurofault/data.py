"""Dataset loading, ported from Implementation.ipynb cell 5 - the only code
in this repo proven to have actually loaded and trained on real data -
parameterized via ExperimentConfig instead of the notebook's hardcoded
range(2000)/range(1000).
"""

from __future__ import annotations

import torch
import torchvision

from neurofault.config import ExperimentConfig


def load_datasets(
    config: ExperimentConfig,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if config.dataset == "cifar10":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_full = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_full = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

    elif config.dataset == "mnist":
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_full = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_full = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

    else:
        raise ValueError(f"Unknown dataset: {config.dataset!r}")

    train_subset = torch.utils.data.Subset(
        train_full, range(min(config.train_samples, len(train_full)))
    )
    test_subset = torch.utils.data.Subset(
        test_full, range(min(config.test_samples, len(test_full)))
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=config.batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        test_subset, batch_size=config.batch_size, shuffle=False
    )

    return trainloader, testloader
