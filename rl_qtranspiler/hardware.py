"""Hardware-graph features and shortest-path calibration costs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import inf, log1p

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class HardwareGraph:
    name: str
    num_qubits: int
    edges: np.ndarray
    cz_errors: np.ndarray
    hop_distances: np.ndarray
    calibration_distances: np.ndarray
    static_node_features: np.ndarray
    directed_edge_index: np.ndarray
    directed_edge_features: np.ndarray
    diameter: int
    max_calibration_distance: float

    def as_networkx(self) -> nx.Graph:
        graph = nx.Graph(name=self.name)
        graph.add_nodes_from(range(self.num_qubits))
        for (left, right), error in zip(self.edges, self.cz_errors, strict=True):
            graph.add_edge(int(left), int(right), cz_error=float(error))
        return graph


def _lexicographic_floyd_warshall(
    num_nodes: int,
    edges: np.ndarray,
    errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimize hops first, then calibration cost among minimum-hop paths."""
    hops = np.full((num_nodes, num_nodes), num_nodes + 1, dtype=np.int16)
    costs = np.full((num_nodes, num_nodes), inf, dtype=np.float64)
    np.fill_diagonal(hops, 0)
    np.fill_diagonal(costs, 0.0)

    for (left, right), error in zip(edges, errors, strict=True):
        cost = -log1p(-float(error))
        hops[left, right] = hops[right, left] = 1
        costs[left, right] = costs[right, left] = cost

    for middle in range(num_nodes):
        for source in range(num_nodes):
            candidate_hops = int(hops[source, middle]) + hops[middle]
            candidate_costs = costs[source, middle] + costs[middle]
            shorter = candidate_hops < hops[source]
            equal_but_cleaner = (candidate_hops == hops[source]) & (
                candidate_costs < costs[source]
            )
            update = shorter | equal_but_cleaner
            hops[source, update] = candidate_hops[update]
            costs[source, update] = candidate_costs[update]

    if np.any(hops > num_nodes):
        raise ValueError("The hardware coupling graph must be connected.")
    return hops, costs


def build_hardware_graph(
    name: str,
    num_qubits: int,
    coupling_data: list[tuple[int, int, float]],
) -> HardwareGraph:
    edges = np.asarray([(a, b) for a, b, _ in coupling_data], dtype=np.int64)
    errors = np.asarray([error for _, _, error in coupling_data], dtype=np.float64)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_qubits))
    graph.add_weighted_edges_from(coupling_data, weight="cz_error")

    hops, calibration = _lexicographic_floyd_warshall(
        num_qubits, edges, errors
    )
    diameter = int(hops.max())
    max_calibration = float(calibration.max())

    degrees = np.asarray([graph.degree(node) for node in graph], dtype=float)
    closeness_map = nx.closeness_centrality(graph)
    betweenness_map = nx.betweenness_centrality(graph, normalized=True)
    eccentricity_map = nx.eccentricity(graph)
    closeness = np.asarray([closeness_map[node] for node in graph])
    betweenness = np.asarray([betweenness_map[node] for node in graph])
    eccentricity = np.asarray([eccentricity_map[node] for node in graph])

    incident = [[] for _ in range(num_qubits)]
    for (left, right), error in zip(edges, errors, strict=True):
        incident[left].append(error)
        incident[right].append(error)
    error_min = np.asarray([min(values) for values in incident])
    error_mean = np.asarray([np.mean(values) for values in incident])
    error_max = np.asarray([max(values) for values in incident])

    def normalize(values: np.ndarray) -> np.ndarray:
        maximum = float(values.max())
        return values / maximum if maximum else values

    static_features = np.column_stack(
        [
            normalize(degrees),
            normalize(closeness),
            normalize(betweenness),
            eccentricity / max(diameter, 1),
            error_min / max(float(errors.max()), 1e-12),
            error_mean / max(float(errors.max()), 1e-12),
            error_max / max(float(errors.max()), 1e-12),
        ]
    ).astype(np.float32)

    directed_edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    directed_errors = np.concatenate([errors, errors], axis=0)
    directed_features = (
        directed_errors / max(float(errors.max()), 1e-12)
    ).reshape(-1, 1).astype(np.float32)

    return HardwareGraph(
        name=name,
        num_qubits=num_qubits,
        edges=edges,
        cz_errors=errors,
        hop_distances=hops,
        calibration_distances=calibration,
        static_node_features=static_features,
        directed_edge_index=directed_edges.T,
        directed_edge_features=directed_features,
        diameter=diameter,
        max_calibration_distance=max_calibration,
    )


@lru_cache(maxsize=1)
def load_ibm_boston() -> HardwareGraph:
    """Load the static calibration snapshot once per process."""
    from rl_qtranspiler.ibm_boston_connectivity_snapshot import (
        BACKEND_NAME,
        COUPLING_DATA,
        NUM_QUBITS,
    )

    hardware = build_hardware_graph(BACKEND_NAME, NUM_QUBITS, COUPLING_DATA)
    for values in (
        hardware.edges,
        hardware.cz_errors,
        hardware.hop_distances,
        hardware.calibration_distances,
        hardware.static_node_features,
        hardware.directed_edge_index,
        hardware.directed_edge_features,
    ):
        values.setflags(write=False)
    return hardware
