"""Placement-problem construction and graph feature generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import Gate

from .hardware import HardwareGraph
from .preprocessing import PreprocessingResult


@dataclass(frozen=True)
class PlacementProblem:
    problem_id: str
    hardware: HardwareGraph
    num_logical_qubits: int
    interaction_weights: np.ndarray
    interaction_counts: np.ndarray
    earliest_interactions: np.ndarray
    placement_order: tuple[int, ...]
    logical_node_features: np.ndarray
    logical_edge_index: np.ndarray
    logical_edge_features: np.ndarray
    total_interaction_weight: float
    allowed_physical_mask: np.ndarray

    def valid_action_mask(self, physical_to_logical: tuple[int, ...]) -> np.ndarray:
        unoccupied = np.asarray(physical_to_logical) < 0
        return self.allowed_physical_mask & unoccupied


def build_placement_problem(
    result: PreprocessingResult,
    hardware: HardwareGraph,
    *,
    problem_id: str | None = None,
    temporal_discount: float = 0.99,
    allowed_physical_qubits: tuple[int, ...] | None = None,
) -> PlacementProblem:
    if not 0 < temporal_discount <= 1:
        raise ValueError("temporal_discount must be in (0, 1].")
    num_logical = result.original.num_qubits
    if num_logical > 30:
        raise ValueError("Version one supports at most 30 logical qubits.")
    if num_logical > hardware.num_qubits:
        raise ValueError("The circuit has more qubits than the hardware.")

    weights = np.zeros((num_logical, num_logical), dtype=np.float64)
    counts = np.zeros((num_logical, num_logical), dtype=np.int32)
    earliest = np.full((num_logical, num_logical), np.inf, dtype=np.float64)

    two_qubit_position = 0
    for instruction in result.instructions:
        if (
            len(instruction.qargs) != 2
            or not isinstance(instruction.operation, Gate)
        ):
            continue
        left, right = instruction.qargs
        if left == right:
            continue
        multiplicity = len(instruction.source_indices)
        contribution = sum(
            temporal_discount ** (two_qubit_position + offset)
            for offset in range(multiplicity)
        )
        weights[left, right] += contribution
        weights[right, left] += contribution
        counts[left, right] += multiplicity
        counts[right, left] += multiplicity
        earliest[left, right] = earliest[right, left] = min(
            earliest[left, right], two_qubit_position
        )
        two_qubit_position += multiplicity

    weighted_degree = weights.sum(axis=1)
    interaction_count = counts.sum(axis=1)
    earliest_by_qubit = np.min(earliest, axis=1)
    fallback_earliest = float(max(two_qubit_position, 1))
    earliest_by_qubit[~np.isfinite(earliest_by_qubit)] = fallback_earliest
    order = tuple(
        sorted(
            range(num_logical),
            key=lambda q: (
                weighted_degree[q] == 0,
                -weighted_degree[q],
                earliest_by_qubit[q],
                q,
            ),
        )
    )

    max_weighted_degree = max(float(weighted_degree.max()), 1.0)
    max_count = max(int(interaction_count.max()), 1)
    logical_node_features = np.column_stack(
        [
            weighted_degree / max_weighted_degree,
            interaction_count / max_count,
            earliest_by_qubit / fallback_earliest,
        ]
    ).astype(np.float32)

    logical_edges: list[tuple[int, int]] = []
    logical_edge_weights: list[float] = []
    max_edge_weight = max(float(weights.max()), 1.0)
    for left in range(num_logical):
        for right in range(num_logical):
            if weights[left, right] > 0:
                logical_edges.append((left, right))
                logical_edge_weights.append(weights[left, right] / max_edge_weight)
    edge_index = (
        np.asarray(logical_edges, dtype=np.int64).T
        if logical_edges
        else np.empty((2, 0), dtype=np.int64)
    )
    edge_features = np.asarray(
        logical_edge_weights, dtype=np.float32
    ).reshape(-1, 1)

    allowed = np.zeros(hardware.num_qubits, dtype=bool)
    if allowed_physical_qubits is None:
        allowed[:] = True
    else:
        allowed[list(allowed_physical_qubits)] = True
    if int(allowed.sum()) < num_logical:
        raise ValueError("Too few allowed physical qubits for this circuit.")

    return PlacementProblem(
        problem_id=problem_id or result.original.name,
        hardware=hardware,
        num_logical_qubits=num_logical,
        interaction_weights=weights,
        interaction_counts=counts,
        earliest_interactions=earliest,
        placement_order=order,
        logical_node_features=logical_node_features,
        logical_edge_index=edge_index,
        logical_edge_features=edge_features,
        total_interaction_weight=max(float(np.triu(weights, 1).sum()), 1.0),
        allowed_physical_mask=allowed,
    )
