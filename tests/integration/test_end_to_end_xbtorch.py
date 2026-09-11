"""End-to-end test through the real xbtorch pipeline.

Skips cleanly (not a failure) if xbtorch isn't installed - install via
`uv sync --extra xbtorch`. Unlike memtorch, xbtorch is confirmed to install
and run cleanly on CPU with no GPU driver required (see
docs/planning/project-setup-plan.md Phase 2).
"""

import pytest

pytest.importorskip("xbtorch", reason="xbtorch not installed (uv sync --extra xbtorch)")

import torch

from neurofault.config import ExperimentConfig
from neurofault.models.cnn import SimpleCNN
from neurofault.system import FaultTolerantNeuromorphic


def test_tiny_end_to_end_run_does_not_raise():
    config = ExperimentConfig(
        name="integration_smoke_xbtorch",
        simulator="xbtorch",
        epochs=1,
        num_batches=2,
        train_samples=16,
        test_samples=8,
        batch_size=4,
    )
    config.crossbar.patch_layer_types = ["Linear"]  # keep it minimal for speed

    system = FaultTolerantNeuromorphic(SimpleCNN(), config)
    assert len(system.handles) > 0  # actually built real crossbar handles

    x = torch.randn(4, 3, 32, 32)
    output = system.forward(x)

    assert output.shape == (4, 10)

    injected = system.inject_faults()
    assert injected > 0  # real faults actually landed on a real accelerator's chip

    system.check_health()  # must not raise
