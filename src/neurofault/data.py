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
        # Real gap found this session (see docs/planning/ "Catching Up"
        # plan): with no augmentation at all, the model plateaued by epoch
        # 2-3 regardless of epoch count or how much more data it was given -
        # it converges onto the exact same fixed set of images fast and
        # stops improving, since there's no variation left to learn from.
        # Standard CIFAR-10 recipe (random crop with padding + horizontal
        # flip) - train-only, never applied to the test set, which must stay
        # clean/deterministic for a fair, reproducible accuracy measurement.
        train_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_full = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )
        test_full = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=test_transform
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
