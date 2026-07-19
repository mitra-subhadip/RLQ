import networkx as nx
import numpy as np
import pytest

from rl_qtranspiler.hardware import load_ibm_boston


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


def test_calibration_cost_prefers_minimum_hop_paths():
    hardware = load_ibm_boston()
    for (left, right), error in zip(
        hardware.edges, hardware.cz_errors, strict=True
    ):
        assert hardware.hop_distances[left, right] == 1
        assert np.isclose(
            hardware.calibration_distances[left, right],
            -np.log1p(-error),
        )


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
