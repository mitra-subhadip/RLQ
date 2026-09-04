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

        total_priority = self.total_priority
        segment = total_priority / batch_size
        values = (
            np.arange(batch_size, dtype=np.float64)
            + np.fromiter(
                (random() for _ in range(batch_size)),
                dtype=np.float64,
                count=batch_size,
            )
        ) * segment
        leaves = np.ones(batch_size, dtype=np.int64)
        active = leaves < self.capacity
        while np.any(active):
            positions = np.flatnonzero(active)
            left = 2 * leaves[positions]
            choose_left = values[positions] <= self._tree[left]
            values[positions] -= np.where(
                choose_left, 0.0, self._tree[left]
            )
            leaves[positions] = left + (~choose_left)
            active = leaves < self.capacity

        transitions: list[Transition] = []
        for leaf in leaves:
            transition = self._data[int(leaf) - self.capacity]
            if transition is None:
                raise RuntimeError("Sampled an uninitialized replay entry.")
            transitions.append(transition)

        probabilities_array = self._tree[leaves] / total_priority
        weights = (self._size * probabilities_array) ** (-beta)
        weights /= weights.max()
        return ReplayBatch(
            tuple(transitions),
            leaves,
            weights.astype(np.float32),
        )

    def update_priorities(
        self, tree_indices: np.ndarray, priorities: np.ndarray
    ) -> None:
        tree_indices = np.asarray(tree_indices, dtype=np.int64)
        raw_priorities = np.abs(np.asarray(priorities, dtype=np.float64))
        if tree_indices.shape != raw_priorities.shape:
            raise ValueError("tree_indices and priorities must have equal shape.")
        if tree_indices.size == 0:
            return

        # Sampling with replacement can return a leaf more than once. Match
        # sequential update semantics by keeping its final supplied priority.
        unique_reversed, reversed_positions = np.unique(
            tree_indices[::-1], return_index=True
        )
        final_positions = tree_indices.size - 1 - reversed_positions
        scaled = (
            raw_priorities[final_positions] + self.priority_epsilon
        ) ** self.alpha
        self._tree[unique_reversed] = scaled

        parents = np.unique(unique_reversed // 2)
        parents = parents[parents > 0]
        while parents.size:
            self._tree[parents] = (
                self._tree[2 * parents] + self._tree[2 * parents + 1]
            )
            parents = np.unique(parents // 2)
            parents = parents[parents > 0]
        self._maximum_priority = max(
            self._maximum_priority, float(raw_priorities.max())
        )

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
