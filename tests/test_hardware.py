import networkx as nx
import numpy as np
import pytest

from rl_qtranspiler.hardware import build_hardware_graph, load_ibm_boston


def test_boston_graph_and_distances():
    hardware = load_ibm_boston()
    graph = hardware.as_networkx()
    expected = dict(nx.all_pairs_shortest_path_length(graph))

    assert hardware.num_qubits == 156
    assert len(hardware.edges) == 176
    assert hardware.diameter == 32
    assert nx.is_connected(graph)
    assert np.array_equal(hardware.hop_distances, hardware.hop_distances.T)
    assert all(
        hardware.hop_distances[left, right] == distance
        for left, values in expected.items()
        for right, distance in values.items()
    )


def test_calibration_cost_can_prefer_a_longer_cleaner_path():
    hardware = build_hardware_graph(
        "fidelity-first",
        3,
        [(0, 1, 0.2), (0, 2, 0.01), (2, 1, 0.01)],
    )

    assert hardware.hop_distances[0, 1] == 1
    assert np.isclose(
        hardware.calibration_distances[0, 1],
        2 * -np.log1p(-0.01),
    )
    assert hardware.calibration_distances[0, 1] < -np.log1p(-0.2)


def test_cached_boston_arrays_are_immutable():
    hardware = load_ibm_boston()
    arrays = (
        hardware.edges,
        hardware.cz_errors,
        hardware.hop_distances,
        hardware.calibration_distances,
        hardware.static_node_features,
        hardware.directed_edge_index,
        hardware.directed_edge_features,
    )

    assert all(not values.flags.writeable for values in arrays)
    with pytest.raises(ValueError):
        hardware.static_node_features[0, 0] = 10.0
    assert load_ibm_boston() is hardware
