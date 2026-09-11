from neurofault.crossbar.health_monitor import HealthMonitor
from tests.unit.fakes import make_handle


def test_get_critical_devices_finds_extreme_conductance_cells():
    handle = make_handle(shape=(4, 4))
    monitor = HealthMonitor(handle)

    # Force two cells to the extremes; everything else stays mid-range and healthy.
    handle.crossbar.conductance_matrix[0, 0] = monitor.g_min
    handle.crossbar.conductance_matrix[2, 3] = monitor.g_max

    critical = set(monitor.get_critical_devices(extreme_threshold=0.02))

    assert (0, 0) in critical
    assert (2, 3) in critical
    # a mid-range, undisturbed cell must not be flagged
    assert (1, 1) not in critical


def test_get_critical_devices_uses_health_score_threshold():
    handle = make_handle(shape=(3, 3))
    monitor = HealthMonitor(handle)
    monitor.health_scores[1, 1] = 5.0  # below default health_threshold=10

    critical = monitor.get_critical_devices()

    assert (1, 1) in critical


def test_update_health_metrics_reduces_health_under_stress():
    handle = make_handle(shape=(2, 2))
    monitor = HealthMonitor(handle)
    initial_health = monitor.health_scores.clone()

    for _ in range(5):
        monitor.update_health_metrics(write_op=True)

    assert monitor.health_scores.mean().item() <= initial_health.mean().item()
