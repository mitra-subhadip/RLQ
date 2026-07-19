"""Placement-problem construction and graph feature generation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from qiskit.circuit import Gate

from .hardware import HardwareGraph
from .preprocessing import PreprocessingResult


@dataclass(frozen=True)
class PlacementProblem:
    problem_id: str
    hardware: HardwareGraph
    routing_pairs: np.ndarray
    original_two_qubit_gate_count: int
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

    @cached_property
    def max_weighted_degree(self) -> float:
        """Largest logical weighted degree, cached across episode states."""
        return max(float(self.interaction_weights.sum(axis=1).max()), 1.0)

    @cached_property
    def allowed_physical_features(self) -> np.ndarray:
        """Float representation used in every model observation."""
        return self.allowed_physical_mask.astype(np.float32)

    @cached_property
    def interaction_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """Upper-triangle logical pairs with nonzero interaction weight."""
        return np.nonzero(np.triu(self.interaction_weights, k=1) > 0)

    @cached_property
    def normalized_pair_weights(self) -> np.ndarray:
        """Normalized weights aligned with :attr:`interaction_pairs`."""
        left, right = self.interaction_pairs
        return (
            self.interaction_weights[left, right]
            / self.total_interaction_weight
        )

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
    if num_logical > 156:
        raise ValueError("Version one supports at most 156 logical qubits.")
    if num_logical > hardware.num_qubits:
        raise ValueError("The circuit has more qubits than the hardware.")

    weights = np.zeros((num_logical, num_logical), dtype=np.float64)
    counts = np.zeros((num_logical, num_logical), dtype=np.int32)
    earliest = np.full((num_logical, num_logical), np.inf, dtype=np.float64)

    # Schedule two-qubit interactions into ASAP layers. Gates on disjoint
    # logical qubits share a layer regardless of their order in circuit.data,
    # while interactions that share a qubit remain sequential.
    last_interaction_layer = np.full(num_logical, -1, dtype=np.int32)
    maximum_interaction_layer = -1
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
        first_layer = (
            max(
                int(last_interaction_layer[left]),
                int(last_interaction_layer[right]),
            )
            + 1
        )
        if temporal_discount == 1.0:
            contribution = float(multiplicity)
        elif multiplicity == 1:
            contribution = temporal_discount**first_layer
        else:
            contribution = (
                temporal_discount**first_layer
                * (1.0 - temporal_discount**multiplicity)
                / (1.0 - temporal_discount)
            )
        weights[left, right] += contribution
        weights[right, left] += contribution
        counts[left, right] += multiplicity
        counts[right, left] += multiplicity
        earliest[left, right] = earliest[right, left] = min(
            earliest[left, right], first_layer
        )
        last_layer = first_layer + multiplicity - 1
        last_interaction_layer[left] = last_layer
        last_interaction_layer[right] = last_layer
        maximum_interaction_layer = max(maximum_interaction_layer, last_layer)

    weighted_degree = weights.sum(axis=1)
    interaction_count = counts.sum(axis=1)
    earliest_by_qubit = np.min(earliest, axis=1)
    interaction_horizon = max(maximum_interaction_layer + 1, 1)
    fallback_earliest = float(interaction_horizon)
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

    max_edge_weight = max(float(weights.max()), 1.0)
    edge_left, edge_right = np.nonzero(weights > 0)
    edge_index = np.stack([edge_left, edge_right]).astype(
        np.int64, copy=False
    )
    edge_features = (
        weights[edge_left, edge_right] / max_edge_weight
    ).astype(np.float32).reshape(-1, 1)

    allowed = np.zeros(hardware.num_qubits, dtype=bool)
    if allowed_physical_qubits is None:
        allowed[:] = True
    else:
        allowed[list(allowed_physical_qubits)] = True
    if int(allowed.sum()) < num_logical:
        raise ValueError("Too few allowed physical qubits for this circuit.")

    routing_pairs = np.asarray(
        [
            instruction.qargs
            for instruction in result.instructions
            if len(instruction.qargs) == 2
            and isinstance(instruction.operation, Gate)
            for _source_index in instruction.source_indices
        ],
        dtype=np.int16,
    ).reshape(-1, 2)

    return PlacementProblem(
        problem_id=problem_id or result.original.name,
        hardware=hardware,
        routing_pairs=routing_pairs,
        original_two_qubit_gate_count=len(routing_pairs),
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
