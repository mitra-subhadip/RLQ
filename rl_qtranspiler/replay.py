"""Prioritized replay with compact placement transitions."""

from __future__ import annotations

from dataclasses import dataclass
from random import random

import numpy as np

from .environment import PlacementState


@dataclass(frozen=True)
class Transition:
    problem_id: str
    state: PlacementState
    action: int
    reward: float
    next_state: PlacementState
    done: bool


@dataclass(frozen=True)
class ReplayBatch:
    transitions: tuple[Transition, ...]
    tree_indices: np.ndarray
    importance_weights: np.ndarray


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int = 200_000,
        alpha: float = 0.6,
        priority_epsilon: float = 1e-6,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self.alpha = alpha
        self.priority_epsilon = priority_epsilon
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._data: list[Transition | None] = [None] * capacity
        self._write_index = 0
        self._size = 0
        self._maximum_priority = 1.0

    def __len__(self) -> int:
        return self._size

    @property
    def total_priority(self) -> float:
        return float(self._tree[1])

    def _set_tree_priority(self, tree_index: int, priority: float) -> None:
        difference = priority - self._tree[tree_index]
        self._tree[tree_index] = priority
        parent = tree_index // 2
        while parent:
            self._tree[parent] += difference
            parent //= 2

    def add(
        self, transition: Transition, priority: float | None = None
    ) -> Transition | None:
        raw_priority = self._maximum_priority if priority is None else priority
        scaled = (abs(raw_priority) + self.priority_epsilon) ** self.alpha
        data_index = self._write_index
        evicted = self._data[data_index] if self._size == self.capacity else None
        self._data[data_index] = transition
        self._set_tree_priority(data_index + self.capacity, scaled)
        self._write_index = (self._write_index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._maximum_priority = max(self._maximum_priority, abs(raw_priority))
        return evicted

    def _find_leaf(self, value: float) -> int:
        index = 1
        while index < self.capacity:
            left = 2 * index
            if value <= self._tree[left]:
                index = left
            else:
                value -= self._tree[left]
                index = left + 1
        return index

    def sample(self, batch_size: int, beta: float) -> ReplayBatch:
        if batch_size > self._size:
            raise ValueError("Not enough transitions to sample this batch.")
        if self.total_priority <= 0:
            raise RuntimeError("Replay priorities sum to zero.")

        segment = self.total_priority / batch_size
        leaves: list[int] = []
        transitions: list[Transition] = []
        probabilities: list[float] = []
        for sample_index in range(batch_size):
            value = (sample_index + random()) * segment
            leaf = self._find_leaf(value)
            data_index = leaf - self.capacity
            transition = self._data[data_index]
            if transition is None:
                raise RuntimeError("Sampled an uninitialized replay entry.")
            leaves.append(leaf)
            transitions.append(transition)
            probabilities.append(self._tree[leaf] / self.total_priority)

        probabilities_array = np.asarray(probabilities)
        weights = (self._size * probabilities_array) ** (-beta)
        weights /= weights.max()
        return ReplayBatch(
            tuple(transitions),
            np.asarray(leaves, dtype=np.int64),
            weights.astype(np.float32),
        )

    def update_priorities(
        self, tree_indices: np.ndarray, priorities: np.ndarray
    ) -> None:
        for tree_index, priority in zip(
            tree_indices, priorities, strict=True
        ):
            raw = float(abs(priority))
            scaled = (raw + self.priority_epsilon) ** self.alpha
            self._set_tree_priority(int(tree_index), scaled)
            self._maximum_priority = max(self._maximum_priority, raw)

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "priority_epsilon": self.priority_epsilon,
            "tree": self._tree.copy(),
            "data": list(self._data),
            "write_index": self._write_index,
            "size": self._size,
            "maximum_priority": self._maximum_priority,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["capacity"]) != self.capacity:
            raise ValueError("Replay capacity does not match the checkpoint.")
        self.alpha = float(state["alpha"])
        self.priority_epsilon = float(state["priority_epsilon"])
        self._tree = np.asarray(state["tree"], dtype=np.float64).copy()
        self._data = list(state["data"])
        self._write_index = int(state["write_index"])
        self._size = int(state["size"])
        self._maximum_priority = float(state["maximum_priority"])
