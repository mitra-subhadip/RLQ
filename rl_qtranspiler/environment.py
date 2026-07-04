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
        return all(physical >= 0 for physical in self.logical_to_physical)


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
        distance_weight: float = 0.5,
        calibration_weight: float = 0.5,
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
        distance = 0.0
        calibration = 0.0
        hardware = self.problem.hardware
        distance_denominator = max(hardware.diameter - 1, 1)
        calibration_denominator = max(
            hardware.max_calibration_distance, 1e-12
        )

        for other_logical, other_physical in enumerate(
            state.logical_to_physical
        ):
            if other_physical < 0:
                continue
            weight = self.problem.interaction_weights[
                logical_qubit, other_logical
            ]
            if weight <= 0:
                continue
            normalized_weight = weight / self.problem.total_interaction_weight
            hops = int(
                hardware.hop_distances[physical_qubit, other_physical]
            )
            error_cost = float(
                hardware.calibration_distances[
                    physical_qubit, other_physical
                ]
            )
            distance += normalized_weight * max(hops - 1, 0) / (
                distance_denominator
            )
            calibration += (
                normalized_weight
                * error_cost
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
        if not self.valid_action_mask()[action]:
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
        if len(set(logical_to_physical)) != len(logical_to_physical):
            raise ValueError("Mapping must be injective.")

        replay = PlacementEnvironment(
            self.problem,
            distance_weight=self.distance_weight,
            calibration_weight=self.calibration_weight,
        )
        for logical in self.problem.placement_order:
            replay.step(int(logical_to_physical[logical]))
        return replay.score


def build_state_features(
    problem: PlacementProblem,
    state: PlacementState,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dynamic physical and logical node features for the GNN."""
    hardware = problem.hardware
    physical_to_logical = np.asarray(state.physical_to_logical)
    occupied = physical_to_logical >= 0
    current = (
        problem.placement_order[state.step_index]
        if state.step_index < problem.num_logical_qubits
        else -1
    )
    weighted_degree = problem.interaction_weights.sum(axis=1)
    max_degree = max(float(weighted_degree.max()), 1.0)

    assigned_degree = np.zeros(hardware.num_qubits)
    current_affinity = np.zeros(hardware.num_qubits)
    for physical, logical in enumerate(physical_to_logical):
        if logical < 0:
            continue
        assigned_degree[physical] = weighted_degree[logical] / max_degree
        if current >= 0:
            current_affinity[physical] = (
                problem.interaction_weights[current, logical] / max_degree
            )

    average_hop = np.zeros(hardware.num_qubits)
    average_calibration = np.zeros(hardware.num_qubits)
    related_count = np.zeros(hardware.num_qubits)
    if current >= 0:
        mapped_neighbors = [
            (logical, physical)
            for logical, physical in enumerate(state.logical_to_physical)
            if physical >= 0 and problem.interaction_weights[current, logical] > 0
        ]
        denominator = sum(
            problem.interaction_weights[current, logical]
            for logical, _ in mapped_neighbors
        )
        if denominator > 0:
            for candidate in range(hardware.num_qubits):
                average_hop[candidate] = sum(
                    problem.interaction_weights[current, logical]
                    * hardware.hop_distances[candidate, physical]
                    for logical, physical in mapped_neighbors
                ) / denominator / max(hardware.diameter, 1)
                average_calibration[candidate] = sum(
                    problem.interaction_weights[current, logical]
                    * hardware.calibration_distances[candidate, physical]
                    for logical, physical in mapped_neighbors
                ) / denominator / max(hardware.max_calibration_distance, 1e-12)
                related_count[candidate] = len(mapped_neighbors) / max(
                    problem.num_logical_qubits, 1
                )

    physical_dynamic = np.column_stack(
        [
            problem.allowed_physical_mask.astype(float),
            occupied.astype(float),
            assigned_degree,
            current_affinity,
            average_hop,
            average_calibration,
            related_count,
        ]
    ).astype(np.float32)
    physical_features = np.concatenate(
        [hardware.static_node_features, physical_dynamic], axis=1
    )

    placed = (np.asarray(state.logical_to_physical) >= 0).astype(float)
    current_flag = np.zeros(problem.num_logical_qubits)
    if current >= 0:
        current_flag[current] = 1.0
    logical_features = np.column_stack(
        [problem.logical_node_features, placed, current_flag]
    ).astype(np.float32)
    return physical_features, logical_features
