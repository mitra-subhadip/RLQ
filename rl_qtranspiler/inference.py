"""Greedy and deterministic beam-search placement inference."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .environment import PlacementEnvironment, PlacementScore, PlacementState
from .evaluation import evaluate_problem_mapping
from .problem import PlacementProblem

if TYPE_CHECKING:
    from .model import GraphDQN


class QValueModel(Protocol):
    def action_values(
        self, problem: PlacementProblem, state: PlacementState
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class PlacementResult:
    logical_to_physical: dict[int, int]
    physical_to_logical: dict[int, int]
    placement_order: tuple[int, ...]
    distance_score: float
    calibration_score: float
    combined_score: float
    routed_log_infidelity: float | None
    routed_normalized_log_infidelity: float | None
    estimated_success_probability: float | None
    beam_width: int
    runtime_seconds: float


@dataclass(frozen=True)
class _Beam:
    state: PlacementState
    accumulated_reward: float


def _advance(
    problem: PlacementProblem,
    state: PlacementState,
    action: int,
) -> tuple[PlacementState, float]:
    environment = PlacementEnvironment(problem)
    logical = problem.placement_order[state.step_index]
    increment = environment.incremental_score(state, logical, action)
    logical_to_physical = list(state.logical_to_physical)
    physical_to_logical = list(state.physical_to_logical)
    logical_to_physical[logical] = action
    physical_to_logical[action] = logical
    return (
        PlacementState(
            tuple(logical_to_physical),
            tuple(physical_to_logical),
            state.step_index + 1,
        ),
        -increment.combined,
    )


def _greedy_state(
    model: QValueModel, problem: PlacementProblem
) -> PlacementState:
    environment = PlacementEnvironment(problem)
    state = environment.reset()
    while not state.done:
        action = int(np.argmax(model.action_values(problem, state)))
        state, _, _, _ = environment.step(action)
    return state


def place(
    model: QValueModel,
    problem: PlacementProblem,
    *,
    beam_width: int = 8,
    expansions_per_state: int = 4,
    selection_metric: str = "routed_fidelity",
    routing_seed: int = 0,
) -> PlacementResult:
    if beam_width < 1 or expansions_per_state < 1:
        raise ValueError("Beam width and expansion count must be positive.")
    if selection_metric not in {"routed_fidelity", "proxy"}:
        raise ValueError(
            "selection_metric must be 'routed_fidelity' or 'proxy'."
        )
    start_time = perf_counter()
    greedy = _greedy_state(model, problem)

    empty = PlacementEnvironment(problem).reset()
    beams = [_Beam(empty, 0.0)]
    for _ in range(problem.num_logical_qubits):
        candidates: list[tuple[float, _Beam]] = []
        for beam in beams:
            q_values = model.action_values(problem, beam.state)
            valid_actions = np.flatnonzero(
                problem.valid_action_mask(beam.state.physical_to_logical)
            )
            ranked = valid_actions[
                np.argsort(q_values[valid_actions])[::-1]
            ][:expansions_per_state]
            for action in ranked:
                next_state, reward = _advance(
                    problem, beam.state, int(action)
                )
                accumulated = beam.accumulated_reward + reward
                predicted_remaining = (
                    0.0
                    if next_state.done
                    else float(
                        np.max(model.action_values(problem, next_state))
                    )
                )
                candidates.append(
                    (
                        accumulated + predicted_remaining,
                        _Beam(next_state, accumulated),
                    )
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        beams = [beam for _, beam in candidates[:beam_width]]

    completed_by_mapping = {
        state.logical_to_physical: state
        for state in [beam.state for beam in beams] + [greedy]
    }
    completed_states = list(completed_by_mapping.values())
    scorer = PlacementEnvironment(problem)
    routed_log_infidelity: float | None = None
    routed_normalized: float | None = None
    estimated_success: float | None = None
    if selection_metric == "routed_fidelity":
        routed_candidates = [
            (
                evaluate_problem_mapping(
                    problem,
                    state.logical_to_physical,
                    seed=routing_seed,
                ),
                state,
            )
            for state in completed_states
        ]
        routed, best = min(
            routed_candidates,
            key=lambda item: (
                item[0].normalized_log_infidelity,
                item[1].logical_to_physical,
            ),
        )
        score = scorer.score_mapping(best.logical_to_physical)
        routed_log_infidelity = routed.estimated_log_infidelity
        routed_normalized = routed.normalized_log_infidelity
        estimated_success = routed.estimated_success_probability
    else:
        scored: list[tuple[PlacementScore, PlacementState]] = [
            (scorer.score_mapping(state.logical_to_physical), state)
            for state in completed_states
        ]
        score, best = min(scored, key=lambda item: item[0].combined)
    logical_mapping = {
        logical: physical
        for logical, physical in enumerate(best.logical_to_physical)
    }
    physical_mapping = {
        physical: logical
        for physical, logical in enumerate(best.physical_to_logical)
        if logical >= 0
    }
    return PlacementResult(
        logical_mapping,
        physical_mapping,
        problem.placement_order,
        score.distance,
        score.calibration,
        score.combined,
        routed_log_infidelity,
        routed_normalized,
        estimated_success,
        beam_width,
        perf_counter() - start_time,
    )
