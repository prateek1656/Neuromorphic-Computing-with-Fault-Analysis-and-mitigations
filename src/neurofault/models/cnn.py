"""Plain, backend-unaware CNN for CIFAR-10. Which layers become memristive
is decided entirely by neurofault.crossbar.array.patch_layers - this module
knows nothing about crossbars, so it stays importable and unit-testable
without any particular backend installed at all.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Iteration log (see docs/planning/project-setup-plan.md):
        # 1) widening (32/64/128 -> 48/96/192 channels, fc1 512 -> 768) on top
        #    of every training-recipe lever (augmentation, weight_decay,
        #    cosine LR, 3x epochs) plateaued at ~52-56% fault-free CIFAR-10
        #    accuracy - real but not enough to approach realistic hardware
        #    accuracy (~90%+).
        # 2) This adds actual depth (VGG-style: 2 conv layers per stage,
        #    6 total instead of 3), not just width - a materially different
        #    lever, verified empirically before being adopted as the default,
        #    not assumed to help. Final spatial size (4x4) and fc1 input size
        #    are unchanged from the widened version, so this only adds
        #    capacity, it doesn't restructure the head.
        self.conv1a = nn.Conv2d(3, 48, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm2d(48)
        self.conv1b = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm2d(48)

        self.conv2a = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm2d(96)
        self.conv2b = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm2d(96)

        self.conv3a = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(192)
        self.conv3b = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(192)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(192 * 4 * 4, 768)
        self.bn4 = nn.BatchNorm1d(768)
        self.fc2 = nn.Linear(768, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1a(self.conv1a(x)))
        x = self.pool(torch.relu(self.bn1b(self.conv1b(x))))

        x = torch.relu(self.bn2a(self.conv2a(x)))
        x = self.pool(torch.relu(self.bn2b(self.conv2b(x))))

        x = torch.relu(self.bn3a(self.conv3a(x)))
        x = self.pool(torch.relu(self.bn3b(self.conv3b(x))))
        x = self.dropout1(x)

        # reshape(), not view(): AIHWKit's AnalogConv2d can return a
        # non-contiguous tensor, which view() rejects.
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout2(x)
        return self.fc2(x)
