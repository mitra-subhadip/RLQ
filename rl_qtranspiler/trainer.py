"""Double-DQN training loop for sequential placement."""

from __future__ import annotations

import copy
import random as random_module
from dataclasses import asdict, dataclass
from pathlib import Path

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
    learning_rate: float = 5e-5
    target_update_steps: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_steps: int = 100_000
    gamma: float = 1.0
    gradient_clip: float = 10.0
    training_frequency: int = 4


class DoubleDQNTrainer:
    def __init__(
        self,
        model: GraphDQN,
        *,
        config: TrainerConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config or TrainerConfig()
        if str(device) == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
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
        self._problem_ref_counts: dict[str, int] = {}
        self.environment_steps = 0
        self.optimization_steps = 0

    def register_problem(self, problem: PlacementProblem) -> None:
        existing = self.problems.get(problem.problem_id)
        if existing is not None and existing is not problem:
            raise ValueError(f"Duplicate problem id {problem.problem_id!r}.")
        self.problems[problem.problem_id] = problem

    def release_problem(self, problem_id: str) -> None:
        if self._problem_ref_counts.get(problem_id, 0) == 0:
            self.problems.pop(problem_id, None)

    def _retain_transition_problem(self, problem: PlacementProblem) -> None:
        self.register_problem(problem)
        self._problem_ref_counts[problem.problem_id] = (
            self._problem_ref_counts.get(problem.problem_id, 0) + 1
        )

    def _release_transition_problem(self, problem_id: str) -> None:
        remaining = self._problem_ref_counts[problem_id] - 1
        if remaining:
            self._problem_ref_counts[problem_id] = remaining
        else:
            del self._problem_ref_counts[problem_id]
            self.problems.pop(problem_id, None)

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

    @property
    def has_started_learning(self) -> bool:
        return self.optimization_steps > 0

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
        if explore and random_module.random() < self.epsilon:
            return int(random_module.choice(valid))
        return int(np.argmax(self.online.action_values(problem, state)))

    def collect_episode(
        self,
        problem: PlacementProblem,
        *,
        explore: bool = True,
        optimize: bool = True,
    ) -> float:
        environment = PlacementEnvironment(problem)
        state = environment.reset()
        total_reward = 0.0
        while not state.done:
            action = self.select_action(problem, state, explore=explore)
            next_state, reward, done, _ = environment.step(action)
            if explore:
                transition = Transition(
                    problem.problem_id,
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )
                self._retain_transition_problem(problem)
                evicted = self.replay.add(transition)
                if evicted is not None:
                    self._release_transition_problem(evicted.problem_id)
                self.environment_steps += 1
                if (
                    optimize
                    and len(self.replay) >= self.config.warmup_transitions
                    and self.environment_steps
                    % self.config.training_frequency
                    == 0
                ):
                    self.optimize_step()
            state = next_state
            total_reward += reward
        return total_reward

    def optimize_step(self) -> float:
        batch = self.replay.sample(
            self.config.batch_size, self.priority_beta
        )
        problems = [
            self.problems[transition.problem_id]
            for transition in batch.transitions
        ]
        states = [transition.state for transition in batch.transitions]
        actions = torch.from_numpy(
            np.fromiter(
                (transition.action for transition in batch.transitions),
                dtype=np.int64,
                count=len(batch.transitions),
            )
        ).to(
            device=self.device,
            dtype=torch.long,
        )
        self.online.train()
        q_values = self.online.forward_batch(problems, states)
        predictions = q_values.gather(1, actions[:, None]).squeeze(1)
        targets = torch.from_numpy(
            np.fromiter(
                (transition.reward for transition in batch.transitions),
                dtype=np.float32,
                count=len(batch.transitions),
            )
        ).to(
            device=self.device,
            dtype=torch.float32,
        )
        nonterminal_indices = [
            index
            for index, transition in enumerate(batch.transitions)
            if not transition.done
        ]
        with torch.no_grad():
            if nonterminal_indices:
                next_problems = [problems[index] for index in nonterminal_indices]
                next_states = [
                    batch.transitions[index].next_state
                    for index in nonterminal_indices
                ]
                online_next = self.online.forward_batch(
                    next_problems, next_states
                )
                best_actions = online_next.argmax(dim=1)
                target_next = self.target.forward_batch(
                    next_problems, next_states
                )
                bootstrap = target_next.gather(
                    1, best_actions[:, None]
                ).squeeze(1)
                target_indices = torch.as_tensor(
                    nonterminal_indices,
                    device=self.device,
                    dtype=torch.long,
                )
                targets[target_indices] += self.config.gamma * bootstrap
        weights = torch.as_tensor(
            batch.importance_weights,
            device=self.device,
            dtype=torch.float32,
        )
        element_losses = nn.functional.smooth_l1_loss(
            predictions, targets, reduction="none"
        )
        loss = (element_losses * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
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

    def save(
        self,
        path: str | Path,
        *,
        extra_state: dict[str, object] | None = None,
    ) -> None:
        torch.save(
            {
                "model": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "environment_steps": self.environment_steps,
                "optimization_steps": self.optimization_steps,
                "replay": self.replay.state_dict(),
                "problems": self.problems,
                "problem_ref_counts": self._problem_ref_counts,
                "python_random_state": random_module.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "extra_state": extra_state or {},
            },
            Path(path),
        )

    def load(self, path: str | Path) -> dict[str, object]:
        checkpoint = torch.load(
            Path(path), map_location=self.device, weights_only=False
        )
        self.config = TrainerConfig(**checkpoint["config"])
        self.online.load_state_dict(checkpoint["model"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.environment_steps = checkpoint["environment_steps"]
        self.optimization_steps = checkpoint["optimization_steps"]
        self.replay = PrioritizedReplayBuffer(
            self.config.replay_capacity,
            alpha=self.config.priority_alpha,
        )
        if "replay" in checkpoint:
            self.replay.load_state_dict(checkpoint["replay"])
        self.problems = checkpoint.get("problems", {})
        self._problem_ref_counts = checkpoint.get("problem_ref_counts", {})
        if "python_random_state" in checkpoint:
            random_module.setstate(checkpoint["python_random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        return checkpoint.get("extra_state", {})
