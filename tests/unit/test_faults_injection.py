from neurofault.config import FaultConfig
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.faults.injection import inject_faults
from tests.unit.fakes import make_handle


def test_custom_stuck_at_injects_requested_density():
    handle = make_handle(shape=(10, 10))  # 100 cells
    monitor = HealthMonitor(handle)
    config = FaultConfig(density=0.2, backend="custom")

    injected = inject_faults(handle, monitor, config)

    assert injected == 20
    matrix = handle.accessor.read()
    at_extreme = ((matrix == monitor.g_min) | (matrix == monitor.g_max)).sum().item()
    assert at_extreme == 20


def test_zero_density_injects_nothing():
    handle = make_handle(shape=(10, 10))
    monitor = HealthMonitor(handle)
    config = FaultConfig(density=0.0, backend="custom")

    assert inject_faults(handle, monitor, config) == 0


def test_auto_backend_falls_back_to_custom_when_native_unverified():
    handle = make_handle(shape=(10, 10))
    monitor = HealthMonitor(handle)
    config = FaultConfig(density=0.1, backend="auto")

    injected = inject_faults(handle, monitor, config)

    assert injected == 10


def test_clustered_distribution_stays_within_bounds():
    handle = make_handle(shape=(8, 8))
    monitor = HealthMonitor(handle)
    config = FaultConfig(density=0.1, distribution="clustered", backend="custom")

    injected = inject_faults(handle, monitor, config)

    assert injected > 0  # exact count varies with cluster overlap, just must be sane
    assert injected <= int(64 * 0.1) + 9  # generous bound: centers * up to 9 neighbors
