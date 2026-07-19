"""Exact teachers, warm starts, and curriculum progression."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain, permutations
from math import factorial

import numpy as np
import torch
from torch import nn

from .environment import (
    DEFAULT_CALIBRATION_WEIGHT,
    DEFAULT_DISTANCE_WEIGHT,
    PlacementEnvironment,
    PlacementState,
)
from .problem import PlacementProblem
from .trainer import DoubleDQNTrainer


@dataclass(frozen=True)
class ExactSolution:
    logical_to_physical: tuple[int, ...]
    combined_score: float


def solve_exact(problem: PlacementProblem) -> ExactSolution:
    """Exhaustively solve a small problem restricted to equally many nodes."""
    allowed = tuple(np.flatnonzero(problem.allowed_physical_mask).tolist())
    if problem.num_logical_qubits > 8:
        raise ValueError("Exact enumeration is limited to eight logical qubits.")
    if len(allowed) != problem.num_logical_qubits:
        raise ValueError(
            "Exact training problems must allow exactly one physical node "
            "per logical qubit."
        )

    num_logical = problem.num_logical_qubits
    assignment_count = factorial(num_logical)
    assignments = np.fromiter(
        chain.from_iterable(permutations(allowed)),
        dtype=np.int64,
        count=assignment_count * num_logical,
    ).reshape(assignment_count, num_logical)
    mappings = np.empty_like(assignments)
    mappings[:, np.asarray(problem.placement_order)] = assignments

    left, right = problem.interaction_pairs
    if left.size:
        hardware = problem.hardware
        weights = problem.normalized_pair_weights
        physical_left = mappings[:, left]
        physical_right = mappings[:, right]
        distance_scores = (
            np.maximum(
                hardware.hop_distances[physical_left, physical_right] - 1,
                0,
            )
            @ weights
            / max(hardware.diameter - 1, 1)
        )
        calibration_scores = (
            hardware.calibration_distances[physical_left, physical_right]
            @ weights
            / max(hardware.max_calibration_distance, 1e-12)
        )
        combined_scores = (
            DEFAULT_DISTANCE_WEIGHT * distance_scores
            + DEFAULT_CALIBRATION_WEIGHT * calibration_scores
        )
    else:
        combined_scores = np.zeros(assignment_count)
    best_index = int(np.argmin(combined_scores))
    return ExactSolution(
        tuple(int(value) for value in mappings[best_index]),
        float(combined_scores[best_index]),
    )


def expert_trajectory(
    problem: PlacementProblem, solution: ExactSolution | None = None
) -> list[tuple[PlacementState, int]]:
    solution = solution or solve_exact(problem)
    environment = PlacementEnvironment(problem)
    state = environment.reset()
    trajectory: list[tuple[PlacementState, int]] = []
    for logical in problem.placement_order:
        action = solution.logical_to_physical[logical]
        trajectory.append((state, action))
        state, _, _, _ = environment.step(action)
    return trajectory


def supervised_warm_start(
    trainer: DoubleDQNTrainer,
    problems: list[PlacementProblem],
    *,
    epochs: int = 10,
    batch_size: int = 128,
    updates: int | None = None,
) -> list[float]:
    """Pretrain action ranking from exact small-instance trajectories.

    Unless ``updates`` is supplied, the optimizer-update budget matches the
    original sequential implementation: ``epochs * number_of_examples``.
    Batching and update count are independent so vectorization cannot silently
    reduce the amount of optimization.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if epochs < 0:
        raise ValueError("epochs must be nonnegative.")
    if updates is not None and updates < 0:
        raise ValueError("updates must be nonnegative.")
    examples: list[tuple[PlacementProblem, PlacementState, int]] = []
    for problem in problems:
        trainer.register_problem(problem)
        examples.extend(
            (problem, state, action)
            for state, action in expert_trajectory(problem)
        )

    update_count = epochs * len(examples) if updates is None else updates
    if update_count and not examples:
        raise ValueError("Warm-start updates require at least one example.")

    update_losses: list[float] = []
    trainer.online.train()
    order = np.arange(len(examples))
    cursor = len(order)
    for _ in range(update_count):
        if cursor >= len(order):
            np.random.shuffle(order)
            cursor = 0
        stop = min(cursor + batch_size, len(order))
        batch = [examples[int(index)] for index in order[cursor:stop]]
        cursor = stop
        q_values = trainer.online.forward_batch(
            [problem for problem, _, _ in batch],
            [state for _, state, _ in batch],
        )
        actions = torch.as_tensor(
            [action for _, _, action in batch],
            device=trainer.device,
            dtype=torch.long,
        )
        loss = nn.functional.cross_entropy(q_values, actions)
        trainer.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            trainer.online.parameters(), trainer.config.gradient_clip
        )
        trainer.optimizer.step()
        update_losses.append(float(loss.item()))
    trainer.target.load_state_dict(trainer.online.state_dict())
    for problem in problems:
        trainer.release_problem(problem.problem_id)
    return update_losses


@torch.inference_mode()
def expert_action_accuracy(
    trainer: DoubleDQNTrainer,
    problems: list[PlacementProblem],
    *,
    batch_size: int = 128,
) -> float:
    """Measure top-one action accuracy on exact expert trajectories."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    examples = [
        (problem, state, action)
        for problem in problems
        for state, action in expert_trajectory(problem)
    ]
    if not examples:
        raise ValueError("Expert accuracy requires at least one example.")

    was_training = trainer.online.training
    trainer.online.eval()
    correct = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        q_values = trainer.online.forward_batch(
            [problem for problem, _, _ in batch],
            [state for _, state, _ in batch],
        )
        actions = torch.as_tensor(
            [action for _, _, action in batch],
            device=trainer.device,
            dtype=torch.long,
        )
        correct += int((q_values.argmax(dim=1) == actions).sum().item())
    trainer.online.train(was_training)
    return correct / len(examples)


@dataclass(frozen=True)
class CurriculumStage:
    minimum_qubits: int
    maximum_qubits: int
    name: str


class Curriculum:
    """Advance after validation fails to improve for a fixed patience."""

    def __init__(self, patience: int = 5) -> None:
        if patience < 1:
            raise ValueError("patience must be positive.")
        self.stages = (
            CurriculumStage(4, 8, "small"),
            CurriculumStage(9, 16, "medium"),
            CurriculumStage(17, 156, "large"),
            CurriculumStage(4, 156, "mixed"),
        )
        self.patience = patience
        self.stage_index = 0
        self.best_validation_score = float("inf")
        self.stale_evaluations = 0

    @property
    def current(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def complete(self) -> bool:
        return self.stage_index == len(self.stages) - 1

    def observe(self, validation_score: float) -> bool:
        if validation_score < self.best_validation_score:
            self.best_validation_score = validation_score
            self.stale_evaluations = 0
            return False
        self.stale_evaluations += 1
        if self.stale_evaluations < self.patience or self.complete:
            return False
        self.stage_index += 1
        self.best_validation_score = float("inf")
        self.stale_evaluations = 0
        return True

    def reset_validation_tracking(self) -> None:
        """Forget scores collected from a previous validation distribution."""
        self.best_validation_score = float("inf")
        self.stale_evaluations = 0

    def state_dict(self) -> dict[str, float | int]:
        return {
            "stage_index": self.stage_index,
            "best_validation_score": self.best_validation_score,
            "stale_evaluations": self.stale_evaluations,
            "patience": self.patience,
        }

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        self.stage_index = int(state["stage_index"])
        self.best_validation_score = float(state["best_validation_score"])
        self.stale_evaluations = int(state["stale_evaluations"])
        self.patience = int(state["patience"])
