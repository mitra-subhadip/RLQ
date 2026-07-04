"""Double-DQN training loop for sequential placement."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from random import choice, random

import numpy as np
import torch
from torch import nn

from .environment import PlacementEnvironment
from .model import GraphDQN
from .problem import PlacementProblem
from .replay import PrioritizedReplayBuffer, Transition


@dataclass(frozen=True)
class TrainerConfig:
    replay_capacity: int = 200_000
    warmup_transitions: int = 10_000
    batch_size: int = 128
    learning_rate: float = 1e-4
    target_update_steps: int = 2_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 250_000
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_steps: int = 1_000_000
    gamma: float = 1.0
    gradient_clip: float = 10.0


class DoubleDQNTrainer:
    def __init__(
        self,
        model: GraphDQN,
        *,
        config: TrainerConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or TrainerConfig()
        self.device = torch.device(device)
        self.online = model.to(self.device)
        self.target = copy.deepcopy(model).to(self.device)
        self.target.eval()
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=self.config.learning_rate
        )
        self.replay = PrioritizedReplayBuffer(
            self.config.replay_capacity,
            alpha=self.config.priority_alpha,
        )
        self.problems: dict[str, PlacementProblem] = {}
        self.environment_steps = 0
        self.optimization_steps = 0

    def register_problem(self, problem: PlacementProblem) -> None:
        existing = self.problems.get(problem.problem_id)
        if existing is not None and existing is not problem:
            raise ValueError(f"Duplicate problem id {problem.problem_id!r}.")
        self.problems[problem.problem_id] = problem

    @property
    def epsilon(self) -> float:
        fraction = min(
            self.environment_steps / self.config.epsilon_decay_steps, 1.0
        )
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    @property
    def priority_beta(self) -> float:
        fraction = min(
            self.environment_steps / self.config.priority_beta_steps, 1.0
        )
        return self.config.priority_beta_start + fraction * (
            1.0 - self.config.priority_beta_start
        )

    def select_action(
        self,
        problem: PlacementProblem,
        state,
        *,
        explore: bool = True,
    ) -> int:
        valid = np.flatnonzero(
            problem.valid_action_mask(state.physical_to_logical)
        )
        if explore and random() < self.epsilon:
            return int(choice(valid.tolist()))
        return int(np.argmax(self.online.action_values(problem, state)))

    def collect_episode(
        self,
        problem: PlacementProblem,
        *,
        explore: bool = True,
        optimize: bool = True,
    ) -> float:
        self.register_problem(problem)
        environment = PlacementEnvironment(problem)
        state = environment.reset()
        total_reward = 0.0
        while not state.done:
            action = self.select_action(problem, state, explore=explore)
            next_state, reward, done, _ = environment.step(action)
            if explore:
                self.replay.add(
                    Transition(
                        problem.problem_id,
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    )
                )
                self.environment_steps += 1
                if (
                    optimize
                    and len(self.replay) >= self.config.warmup_transitions
                ):
                    self.optimize_step()
            state = next_state
            total_reward += reward
        return total_reward

    def optimize_step(self) -> float:
        batch = self.replay.sample(
            self.config.batch_size, self.priority_beta
        )
        predicted_values: list[torch.Tensor] = []
        target_values: list[torch.Tensor] = []

        for transition in batch.transitions:
            problem = self.problems[transition.problem_id]
            q_values = self.online(problem, transition.state)
            predicted_values.append(q_values[transition.action])
            with torch.no_grad():
                if transition.done:
                    bootstrap = torch.tensor(0.0, device=self.device)
                else:
                    online_next = self.online(problem, transition.next_state)
                    best_action = int(torch.argmax(online_next).item())
                    bootstrap = self.target(
                        problem, transition.next_state
                    )[best_action]
                target_values.append(
                    torch.tensor(
                        transition.reward,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    + self.config.gamma * bootstrap
                )

        predictions = torch.stack(predicted_values)
        targets = torch.stack(target_values)
        weights = torch.as_tensor(
            batch.importance_weights,
            device=self.device,
            dtype=torch.float32,
        )
        element_losses = nn.functional.smooth_l1_loss(
            predictions, targets, reduction="none"
        )
        loss = (element_losses * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.gradient_clip
        )
        self.optimizer.step()
        td_errors = (targets - predictions).detach().abs().cpu().numpy()
        self.replay.update_priorities(batch.tree_indices, td_errors)

        self.optimization_steps += 1
        if self.optimization_steps % self.config.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "model": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "environment_steps": self.environment_steps,
                "optimization_steps": self.optimization_steps,
            },
            Path(path),
        )

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        self.online.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.environment_steps = checkpoint["environment_steps"]
        self.optimization_steps = checkpoint["optimization_steps"]
