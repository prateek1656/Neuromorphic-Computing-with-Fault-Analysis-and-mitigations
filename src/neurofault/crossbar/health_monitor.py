"""Health monitoring for a single memristive crossbar.

Ported from the original health_monitor.py, with get_critical_devices()
rewritten as vectorized tensor ops instead of a Python double for-loop over
every (row, col) cell - that loop is confirmed (via notebook investigation)
to have actually caused a real training run to be killed for being too slow.
"""

from __future__ import annotations

import logging

import torch

from neurofault.crossbar.array import CrossbarHandle

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Tracks per-device stress/health/stability for one crossbar, and predicts
    which devices are likely heading toward failure."""

    def __init__(self, handle: CrossbarHandle):
        self.handle = handle
        self.shape = tuple(handle.crossbar.conductance_matrix.shape)

        self.r_on = float(handle.device_params.get("r_on", 100.0))
        self.r_off = float(handle.device_params.get("r_off", 16000.0))
        self.g_min = 1.0 / self.r_off
        self.g_max = 1.0 / self.r_on
        self.g_range = self.g_max - self.g_min

        self.stress = torch.zeros(self.shape)
        self.health_scores = torch.ones(self.shape) * 100
        self.stability_index = torch.ones(self.shape)
        self.operation_counts = torch.zeros(self.shape)

        self.g_history: list[torch.Tensor] = [handle.crossbar.conductance_matrix.clone()]
        self.max_history = 20

        self.failure_probability = torch.zeros(self.shape)

        logger.info("Initialized health monitor for %dx%d crossbar", *self.shape)

    def update_health_metrics(
        self, voltage_applied: torch.Tensor | None = None, write_op: bool = False
    ):
        current_g = self.handle.crossbar.conductance_matrix
        self.operation_counts += 1

        g_normalized = torch.clamp((current_g - self.g_min) / self.g_range, 0, 1)
        deviation_from_mid = 2 * torch.abs(g_normalized - 0.5)

        if len(self.g_history) >= 3:
            prev_g1, prev_g2 = self.g_history[-1], self.g_history[-2]
            recent_change1 = torch.abs(current_g - prev_g1) / self.g_range
            recent_change2 = torch.abs(prev_g1 - prev_g2) / self.g_range
            change_acceleration = torch.abs(recent_change1 - recent_change2)

            self.stability_index = (
                self.stability_index * 0.9 + (1.0 - change_acceleration * 10) * 0.1
            )
            self.stability_index = torch.clamp(self.stability_index, 0.0, 1.0)

            unstable = recent_change1 > 0.1
            self.health_scores[unstable] -= 0.5

        op_stress_factor = 0.02 if write_op else 0.001
        voltage_factor = 1.0
        if voltage_applied is not None:
            voltage_factor = 1.0 + torch.abs(voltage_applied).mean().item()

        new_stress = op_stress_factor * voltage_factor * (0.5 + deviation_from_mid)
        new_stress = new_stress * (2.0 - self.stability_index)

        self.stress = torch.clamp(self.stress * 0.99 + new_stress, 0.0, 1.0)
        self.health_scores = torch.clamp(100 * (1.0 - self.stress), 0, 100)

        self.g_history.append(current_g.clone())
        if len(self.g_history) > self.max_history:
            self.g_history.pop(0)

    def predict_faults(self) -> torch.Tensor:
        g_normalized = torch.clamp(
            (self.handle.crossbar.conductance_matrix - self.g_min) / self.g_range, 0, 1
        )
        proximity_to_extreme = torch.clamp(
            torch.max(4 * g_normalized**2, 4 * (1 - g_normalized) ** 2), 0.0, 1.0
        )
        instability_factor = 1.0 - self.stability_index

        drift_factor = torch.zeros(self.shape)
        if len(self.g_history) >= 5:
            g_diffs = torch.stack(
                [(self.g_history[-i] - self.g_history[-i - 1]) / self.g_range for i in range(1, 5)]
            )
            drift_magnitude = torch.mean(torch.abs(g_diffs), dim=0)
            drift_consistency = torch.std(torch.sign(g_diffs), dim=0)
            drift_factor = torch.clamp(drift_magnitude * (2.0 - drift_consistency) * 5.0, 0.0, 1.0)

        self.failure_probability = (
            0.4 * self.stress
            + 0.3 * instability_factor
            + 0.2 * proximity_to_extreme
            + 0.1 * drift_factor
        )
        return self.failure_probability

    def get_critical_devices(
        self,
        health_threshold: float = 10,
        probability_threshold: float = 0.8,
        extreme_threshold: float = 0.02,
    ) -> list[tuple[int, int]]:
        """Vectorized replacement for the original's Python double for-loop.

        Computes the full-array condition as one boolean mask, then converts
        to a Python list only over the (small) filtered result via
        torch.nonzero - never iterates every cell in Python.
        """
        self.predict_faults()
        g_normalized = torch.clamp(
            (self.handle.crossbar.conductance_matrix - self.g_min) / self.g_range, 0, 1
        )
        mask = (
            (self.health_scores < health_threshold)
            | (self.failure_probability > probability_threshold)
            | (g_normalized < extreme_threshold)
            | (g_normalized > (1 - extreme_threshold))
        )
        rows, cols = torch.nonzero(mask, as_tuple=True)
        return list(zip(rows.tolist(), cols.tolist()))

    def get_health_summary(self) -> dict:
        self.predict_faults()
        return {
            "avg_health": self.health_scores.mean().item(),
            "min_health": self.health_scores.min().item(),
            "avg_stability": self.stability_index.mean().item(),
            "avg_failure_prob": self.failure_probability.mean().item(),
            "critical_count": (self.health_scores < 20).sum().item(),
        }
