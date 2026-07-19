"""Finite-horizon MDP for sequential initial placement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problem import PlacementProblem


@dataclass(frozen=True)
class PlacementState:
    logical_to_physical: tuple[int, ...]
    physical_to_logical: tuple[int, ...]
    step_index: int

    @property
    def done(self) -> bool:
        return self.step_index >= len(self.logical_to_physical)


@dataclass(frozen=True)
class PlacementScore:
    distance: float
    calibration: float
    combined: float


class PlacementEnvironment:
    def __init__(
        self,
        problem: PlacementProblem,
        *,
        distance_weight: float = 0.9,
        calibration_weight: float = 0.1,
    ) -> None:
        if not np.isclose(distance_weight + calibration_weight, 1.0):
            raise ValueError("Reward weights must sum to one.")
        self.problem = problem
        self.distance_weight = distance_weight
        self.calibration_weight = calibration_weight
        self._state = self._empty_state()
        self._score = PlacementScore(0.0, 0.0, 0.0)

    def _empty_state(self) -> PlacementState:
        return PlacementState(
            logical_to_physical=(-1,) * self.problem.num_logical_qubits,
            physical_to_logical=(-1,) * self.problem.hardware.num_qubits,
            step_index=0,
        )

    def reset(self) -> PlacementState:
        self._state = self._empty_state()
        self._score = PlacementScore(0.0, 0.0, 0.0)
        return self._state

    @property
    def state(self) -> PlacementState:
        return self._state

    @property
    def score(self) -> PlacementScore:
        return self._score

    def valid_action_mask(self, state: PlacementState | None = None) -> np.ndarray:
        selected = state or self._state
        return self.problem.valid_action_mask(selected.physical_to_logical)

    def current_logical_qubit(self, state: PlacementState | None = None) -> int:
        selected = state or self._state
        if selected.step_index >= self.problem.num_logical_qubits:
            raise ValueError("The placement episode is complete.")
        return self.problem.placement_order[selected.step_index]

    def incremental_score(
        self,
        state: PlacementState,
        logical_qubit: int,
        physical_qubit: int,
    ) -> PlacementScore:
        hardware = self.problem.hardware
        distance_denominator = max(hardware.diameter - 1, 1)
        calibration_denominator = max(
            hardware.max_calibration_distance, 1e-12
        )

        logical_to_physical = np.asarray(state.logical_to_physical)
        mapped_logical = np.flatnonzero(logical_to_physical >= 0)
        weights = self.problem.interaction_weights[
            logical_qubit, mapped_logical
        ]
        interacting = weights > 0
        if not np.any(interacting):
            return PlacementScore(0.0, 0.0, 0.0)

        physical = logical_to_physical[mapped_logical[interacting]]
        normalized_weights = (
            weights[interacting] / self.problem.total_interaction_weight
        )
        hops = hardware.hop_distances[physical_qubit, physical]
        calibration_costs = hardware.calibration_distances[
            physical_qubit, physical
        ]
        distance = float(
            normalized_weights @ np.maximum(hops - 1, 0)
            / distance_denominator
        )
        calibration = float(
            normalized_weights @ calibration_costs
            / calibration_denominator
        )

        combined = (
            self.distance_weight * distance
            + self.calibration_weight * calibration
        )
        return PlacementScore(distance, calibration, combined)

    def step(
        self, action: int
    ) -> tuple[PlacementState, float, bool, dict[str, float | int]]:
        if self._state.step_index >= self.problem.num_logical_qubits:
            raise RuntimeError("Cannot step a completed placement episode.")
        if not 0 <= action < self.problem.hardware.num_qubits:
            raise ValueError(f"Physical-qubit action {action} is out of range.")
        if (
            not self.problem.allowed_physical_mask[action]
            or self._state.physical_to_logical[action] >= 0
        ):
            raise ValueError(f"Physical qubit {action} is not a valid action.")

        logical = self.current_logical_qubit()
        increment = self.incremental_score(self._state, logical, action)
        logical_to_physical = list(self._state.logical_to_physical)
        physical_to_logical = list(self._state.physical_to_logical)
        logical_to_physical[logical] = action
        physical_to_logical[action] = logical
        self._state = PlacementState(
            tuple(logical_to_physical),
            tuple(physical_to_logical),
            self._state.step_index + 1,
        )
        self._score = PlacementScore(
            self._score.distance + increment.distance,
            self._score.calibration + increment.calibration,
            self._score.combined + increment.combined,
        )
        return (
            self._state,
            -increment.combined,
            self._state.step_index == self.problem.num_logical_qubits,
            {
                "logical_qubit": logical,
                "distance_cost": increment.distance,
                "calibration_cost": increment.calibration,
                "combined_cost": increment.combined,
            },
        )

    def score_mapping(
        self, logical_to_physical: tuple[int, ...] | list[int]
    ) -> PlacementScore:
        if len(logical_to_physical) != self.problem.num_logical_qubits:
            raise ValueError("Mapping length does not match the circuit.")
        mapping = np.asarray(logical_to_physical, dtype=np.int64)
        if np.unique(mapping).size != mapping.size:
            raise ValueError("Mapping must be injective.")
        hardware = self.problem.hardware
        if (
            np.any(mapping < 0)
            or np.any(mapping >= hardware.num_qubits)
            or not np.all(self.problem.allowed_physical_mask[mapping])
        ):
            raise ValueError("Mapping contains a disallowed physical qubit.")

        left, right = self.problem.interaction_pairs
        if left.size == 0:
            return PlacementScore(0.0, 0.0, 0.0)
        weights = self.problem.normalized_pair_weights
        physical_left = mapping[left]
        physical_right = mapping[right]
        hops = hardware.hop_distances[physical_left, physical_right]
        calibration_costs = hardware.calibration_distances[
            physical_left, physical_right
        ]
        distance = float(
            weights @ np.maximum(hops - 1, 0)
            / max(hardware.diameter - 1, 1)
        )
        calibration = float(
            weights @ calibration_costs
            / max(hardware.max_calibration_distance, 1e-12)
        )
        combined = (
            self.distance_weight * distance
            + self.calibration_weight * calibration
        )
        return PlacementScore(distance, calibration, combined)


def build_state_features(
    problem: PlacementProblem,
    state: PlacementState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dynamic physical and logical node features for the GNN."""
    hardware = problem.hardware
    physical_to_logical = np.asarray(state.physical_to_logical)
    logical_to_physical = np.asarray(state.logical_to_physical)
    occupied = physical_to_logical >= 0
    current = (
        problem.placement_order[state.step_index]
        if state.step_index < problem.num_logical_qubits
        else -1
    )
    assigned_degree = np.zeros(hardware.num_qubits, dtype=np.float32)
    current_affinity = np.zeros(hardware.num_qubits, dtype=np.float32)
    assigned_physical = np.flatnonzero(occupied)
    assigned_logical = physical_to_logical[assigned_physical]
    assigned_degree[assigned_physical] = problem.logical_node_features[
        assigned_logical, 0
    ]
    if current >= 0:
        current_affinity[assigned_physical] = (
            problem.interaction_weights[current, assigned_logical]
            / problem.max_weighted_degree
        )

    average_hop = np.zeros(hardware.num_qubits, dtype=np.float32)
    average_calibration = np.zeros(hardware.num_qubits, dtype=np.float32)
    related_count = np.zeros(hardware.num_qubits, dtype=np.float32)
    if current >= 0:
        mapped_logical = np.flatnonzero(logical_to_physical >= 0)
        neighbor_weights = problem.interaction_weights[
            current, mapped_logical
        ]
        interacting = neighbor_weights > 0
        if np.any(interacting):
            weights = neighbor_weights[interacting]
            mapped_physical = logical_to_physical[
                mapped_logical[interacting]
            ]
            denominator = float(weights.sum())
            average_hop[:] = (
                hardware.hop_distances[:, mapped_physical] @ weights
                / denominator
                / max(hardware.diameter, 1)
            )
            average_calibration[:] = (
                hardware.calibration_distances[:, mapped_physical] @ weights
                / denominator
                / max(hardware.max_calibration_distance, 1e-12)
            )
            related_count.fill(
                mapped_physical.size / max(problem.num_logical_qubits, 1)
            )

    physical_features = np.empty(
        (hardware.num_qubits, hardware.static_node_features.shape[1] + 7),
        dtype=np.float32,
    )
    physical_features[:, :7] = hardware.static_node_features
    physical_features[:, 7] = problem.allowed_physical_features
    physical_features[:, 8] = occupied
    physical_features[:, 9] = assigned_degree
    physical_features[:, 10] = current_affinity
    physical_features[:, 11] = average_hop
    physical_features[:, 12] = average_calibration
    physical_features[:, 13] = related_count

    logical_features = np.empty(
        (problem.num_logical_qubits, problem.logical_node_features.shape[1] + 2),
        dtype=np.float32,
    )
    logical_features[:, :3] = problem.logical_node_features
    logical_features[:, 3] = logical_to_physical >= 0
    logical_features[:, 4] = 0.0
    if current >= 0:
        logical_features[current, 4] = 1.0
    return physical_features, logical_features
