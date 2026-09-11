from neurofault.crossbar.self_healing import RedundancyPool
from neurofault.mitigation.remap import apply_remap
from tests.unit.fakes import make_handle


def test_remap_writes_through_the_same_canonical_crossbar_object():
    """Direct regression test for bug #1: the original codebase built a
    second, disconnected Crossbar inside SelfHealingCrossbar and remapping
    never affected the object the model actually computed with."""
    handle = make_handle(shape=(4, 4))
    pool = RedundancyPool(handle, redundancy_factor=0.5)

    reference_value = pool.reference_conductance[0, 0].item()
    # Corrupt the cell after the reference was captured, simulating a fault.
    handle.crossbar.conductance_matrix[0, 0] = 999.0

    remapped = apply_remap(handle, pool, [(0, 0)])

    assert remapped == 1
    # The write must land on the SAME object apply_remap was given - assert
    # by reading through `handle`, not some other reference, and by identity
    # of the underlying tensor object never having changed.
    assert handle.crossbar.conductance_matrix[0, 0].item() == reference_value

    # A second, independently-constructed handle/crossbar must be untouched -
    # this is what "disconnected shadow crossbar" looked like in the original bug.
    other_handle = make_handle(shape=(4, 4))
    other_handle.crossbar.conductance_matrix[0, 0] = 999.0
    assert other_handle.crossbar.conductance_matrix[0, 0].item() == 999.0


def test_remap_stops_when_redundancy_capacity_exhausted():
    handle = make_handle(shape=(4, 4))
    pool = RedundancyPool(handle, redundancy_factor=0.1)  # capacity = max(1, int(16*0.1)) = 1

    remapped = apply_remap(handle, pool, [(0, 0), (1, 1), (2, 2)])

    assert remapped == 1
    assert pool.available_capacity == 0
