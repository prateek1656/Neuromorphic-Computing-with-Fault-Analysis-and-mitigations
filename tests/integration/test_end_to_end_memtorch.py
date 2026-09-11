"""End-to-end test through the real memtorch pipeline.

Skips cleanly (not a failure) if memtorch isn't importable - its C++/CUDA
extension build requires an NVIDIA driver even for CPU-only usage in the
installed 1.1.6 build, which this sandbox doesn't have. See
docs/planning/project-setup-plan.md for the confirmed root cause.
"""

import pytest

pytest.importorskip("memtorch", reason="memtorch not installed/buildable in this environment")

import torch

from neurofault.config import ExperimentConfig
from neurofault.models.cnn import SimpleCNN
from neurofault.system import FaultTolerantNeuromorphic


def test_tiny_end_to_end_run_does_not_raise():
    config = ExperimentConfig(
        name="integration_smoke",
        epochs=1,
        num_batches=2,
        train_samples=16,
        test_samples=8,
        batch_size=4,
    )
    config.crossbar.patch_layer_types = ["Linear"]  # keep it minimal for speed

    system = FaultTolerantNeuromorphic(SimpleCNN(), config)
    x = torch.randn(4, 3, 32, 32)

    output = system.forward(x)

    assert output.shape == (4, 10)
    system.inject_faults()
    system.check_health()  # must not raise
