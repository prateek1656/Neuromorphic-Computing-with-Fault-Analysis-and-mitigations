"""A lightweight stand-in for memtorch.bh.crossbar.Crossbar, exposing only the
interface the rest of neurofault actually depends on. Lets crossbar/faults/
mitigation logic be unit-tested without a real memtorch install (whose C++/CUDA
build is fragile and GPU-driver-dependent even at build-metadata time - see
docs/planning/project-setup-plan.md).
"""

from __future__ import annotations

import torch

from neurofault.crossbar.array import CrossbarHandle


class FakeCrossbar:
    def __init__(self, shape: tuple[int, int], r_on: float = 100.0, r_off: float = 10000.0):
        self.shape = shape
        self.r_on = r_on
        self.r_off = r_off
        g_min, g_max = 1.0 / r_off, 1.0 / r_on
        self.conductance_matrix = torch.full(shape, (g_min + g_max) / 2)
        self.devices = None

    def write_conductance_matrix(self, matrix: torch.Tensor, transistor: bool = True, **kwargs):
        self.conductance_matrix = matrix.clone()

    def update(self, from_devices: bool = True, parallelize: bool = False):
        pass


def make_handle(shape: tuple[int, int] = (8, 8), **crossbar_kwargs) -> CrossbarHandle:
    crossbar = FakeCrossbar(shape, **crossbar_kwargs)
    layer = torch.nn.Linear(shape[1], shape[0])
    return CrossbarHandle(
        name="test_layer_crossbar_0",
        crossbar=crossbar,
        layer=layer,
        device_params={"r_on": crossbar.r_on, "r_off": crossbar.r_off},
    )
