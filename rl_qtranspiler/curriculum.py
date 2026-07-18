"""Exact teachers, warm starts, and curriculum progression."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
import torch
from torch import nn

from .environment import PlacementEnvironment, PlacementState
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

    environment = PlacementEnvironment(problem)
    best_mapping: tuple[int, ...] | None = None
    best_score = float("inf")
    for assignment in permutations(allowed):
        mapping = [-1] * problem.num_logical_qubits
        for logical, physical in zip(
            problem.placement_order, assignment, strict=True
        ):
            mapping[logical] = physical
        score = environment.score_mapping(mapping).combined
        if score < best_score:
            best_score = score
            best_mapping = tuple(mapping)
    if best_mapping is None:
        raise RuntimeError("Exact solver did not evaluate any mappings.")
    return ExactSolution(best_mapping, best_score)


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
) -> list[float]:
    """Pretrain action ranking from exact small-instance trajectories."""
    examples: list[tuple[PlacementProblem, PlacementState, int]] = []
    for problem in problems:
        trainer.register_problem(problem)
        examples.extend(
            (problem, state, action)
            for state, action in expert_trajectory(problem)
        )

    epoch_losses: list[float] = []
    trainer.online.train()
    for _ in range(epochs):
        np.random.shuffle(examples)
        losses: list[float] = []
        for problem, state, action in examples:
            q_values = trainer.online(problem, state)
            loss = nn.functional.cross_entropy(
                q_values.unsqueeze(0),
                torch.tensor([action], device=trainer.device),
            )
            trainer.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                trainer.online.parameters(), trainer.config.gradient_clip
            )
            trainer.optimizer.step()
            losses.append(float(loss.item()))
        epoch_losses.append(float(np.mean(losses)))
    trainer.target.load_state_dict(trainer.online.state_dict())
    for problem in problems:
        trainer.release_problem(problem.problem_id)
    return epoch_losses


@dataclass(frozen=True)
class CurriculumStage:
    minimum_qubits: int
    maximum_qubits: int
    name: str


class Curriculum:
    """Advance after validation fails to improve for a fixed patience."""

    def __init__(self, patience: int = 2) -> None:
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
