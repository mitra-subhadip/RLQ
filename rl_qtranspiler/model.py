"""Edge-aware graph DQN implemented with core PyTorch only."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch import nn

from .environment import PlacementState, build_state_features

if TYPE_CHECKING:
    from .problem import PlacementProblem


class EdgeAwareMessagePassing(nn.Module):
    def __init__(self, input_dim: int, edge_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.self_projection = nn.Linear(input_dim, hidden_dim)
        self.message = nn.Sequential(
            nn.Linear(input_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.self_projection(node_features)
        if edge_index.shape[1] == 0:
            return torch.relu(self.normalization(projected))

        source, destination = edge_index
        messages = self.message(
            torch.cat([node_features[source], edge_features], dim=-1)
        )
        aggregated = torch.zeros_like(projected)
        aggregated.index_add_(0, destination, messages)
        counts = torch.zeros(
            node_features.shape[0],
            device=node_features.device,
            dtype=node_features.dtype,
        )
        counts.index_add_(
            0,
            destination,
            torch.ones(
                destination.shape[0],
                device=node_features.device,
                dtype=node_features.dtype,
            ),
        )
        aggregated = aggregated / counts.clamp_min(1.0).unsqueeze(-1)
        return torch.relu(self.normalization(projected + aggregated))


class GraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        layers: int = 3,
    ) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (layers - 1)
        self.layers = nn.ModuleList(
            EdgeAwareMessagePassing(dimension, edge_dim, hidden_dim)
            for dimension in dimensions
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        encoded = node_features
        for layer in self.layers:
            encoded = layer(encoded, edge_index, edge_features)
        return encoded


class GraphDQN(nn.Module):
    """Dueling graph DQN producing one Q-value per physical qubit."""

    PHYSICAL_FEATURES = 14
    LOGICAL_FEATURES = 5

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.physical_encoder = GraphEncoder(
            self.PHYSICAL_FEATURES + hidden_dim, 1, hidden_dim
        )
        self.logical_encoder = GraphEncoder(
            self.LOGICAL_FEATURES, 1, hidden_dim
        )
        state_dim = hidden_dim * 3
        candidate_dim = hidden_dim * 4 + self.PHYSICAL_FEATURES
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(candidate_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _tensor(
        self, values: np.ndarray, *, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.as_tensor(values, dtype=dtype, device=self.device)

    def forward(
        self,
        problem: PlacementProblem,
        state: PlacementState,
        valid_action_mask: np.ndarray | None = None,
    ) -> torch.Tensor:
        masks = None if valid_action_mask is None else [valid_action_mask]
        return self.forward_batch([problem], [state], masks)[0]

    def forward_batch(
        self,
        problems: Sequence[PlacementProblem],
        states: Sequence[PlacementState],
        valid_action_masks: Sequence[np.ndarray] | None = None,
    ) -> torch.Tensor:
        """Evaluate a heterogeneous logical-graph batch in three GNN passes."""
        if not problems or len(problems) != len(states):
            raise ValueError("problems and states must have equal nonzero length.")
        batch_size = len(problems)
        num_physical = problems[0].hardware.num_qubits
        if any(
            problem.hardware.num_qubits != num_physical for problem in problems
        ):
            raise ValueError("All batched hardware graphs must have equal size.")

        physical_arrays: list[np.ndarray] = []
        logical_arrays: list[np.ndarray] = []
        physical_edge_arrays: list[np.ndarray] = []
        physical_edge_features: list[np.ndarray] = []
        logical_edge_arrays: list[np.ndarray] = []
        logical_edge_features: list[np.ndarray] = []
        logical_offsets: list[int] = []
        logical_offset = 0

        for batch_index, (problem, state) in enumerate(
            zip(problems, states, strict=True)
        ):
            physical, logical = build_state_features(problem, state)
            physical_arrays.append(physical)
            logical_arrays.append(logical)
            physical_edge_arrays.append(
                problem.hardware.directed_edge_index
                + batch_index * num_physical
            )
            physical_edge_features.append(
                problem.hardware.directed_edge_features
            )
            logical_offsets.append(logical_offset)
            logical_edge_arrays.append(
                problem.logical_edge_index + logical_offset
            )
            logical_edge_features.append(problem.logical_edge_features)
            logical_offset += problem.num_logical_qubits

        physical_x = self._tensor(
            np.concatenate(physical_arrays), dtype=torch.float32
        )
        logical_x = self._tensor(
            np.concatenate(logical_arrays), dtype=torch.float32
        )
        physical_edges = self._tensor(
            np.concatenate(physical_edge_arrays, axis=1), dtype=torch.long
        )
        physical_edge_x = self._tensor(
            np.concatenate(physical_edge_features), dtype=torch.float32
        )
        logical_edges = self._tensor(
            np.concatenate(logical_edge_arrays, axis=1), dtype=torch.long
        )
        logical_edge_x = self._tensor(
            np.concatenate(logical_edge_features), dtype=torch.float32
        )

        logical_embeddings = self.logical_encoder(
            logical_x, logical_edges, logical_edge_x
        )
        mapped_logical_embeddings = torch.zeros(
            (batch_size * num_physical, self.hidden_dim),
            device=self.device,
            dtype=logical_embeddings.dtype,
        )
        current_embeddings: list[torch.Tensor] = []
        logical_pools: list[torch.Tensor] = []
        for batch_index, (problem, state, offset) in enumerate(
            zip(problems, states, logical_offsets, strict=True)
        ):
            logical_slice = logical_embeddings[
                offset : offset + problem.num_logical_qubits
            ]
            logical_pools.append(logical_slice.mean(dim=0))
            current = problem.placement_order[state.step_index]
            current_embeddings.append(logical_slice[current])
            occupied_physical = [
                physical
                for physical, logical in enumerate(state.physical_to_logical)
                if logical >= 0
            ]
            if occupied_physical:
                logical_indices = [
                    state.physical_to_logical[physical]
                    for physical in occupied_physical
                ]
                destinations = self._tensor(
                    np.asarray(occupied_physical)
                    + batch_index * num_physical,
                    dtype=torch.long,
                )
                sources = self._tensor(
                    np.asarray(logical_indices) + offset,
                    dtype=torch.long,
                )
                mapped_logical_embeddings[destinations] = logical_embeddings[
                    sources
                ]

        physical_input = torch.cat(
            [physical_x, mapped_logical_embeddings], dim=-1
        )
        physical_embeddings = self.physical_encoder(
            physical_input, physical_edges, physical_edge_x
        ).reshape(batch_size, num_physical, self.hidden_dim)
        physical_pool = physical_embeddings.mean(dim=1)
        logical_pool = torch.stack(logical_pools)
        current_embedding = torch.stack(current_embeddings)

        state_embedding = torch.cat(
            [current_embedding, physical_pool, logical_pool], dim=-1
        )
        values = self.value_head(state_embedding).squeeze(-1)
        candidate_input = torch.cat(
            [
                physical_embeddings,
                current_embedding[:, None, :].expand(-1, num_physical, -1),
                physical_pool[:, None, :].expand(-1, num_physical, -1),
                logical_pool[:, None, :].expand(-1, num_physical, -1),
                physical_x.reshape(
                    batch_size, num_physical, self.PHYSICAL_FEATURES
                ),
            ],
            dim=-1,
        )
        advantages = self.advantage_head(candidate_input).squeeze(-1)

        mask_arrays = (
            valid_action_masks
            if valid_action_masks is not None
            else [
                problem.valid_action_mask(state.physical_to_logical)
                for problem, state in zip(problems, states, strict=True)
            ]
        )
        mask = self._tensor(np.stack(mask_arrays), dtype=torch.bool)
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("A batched state has no valid physical actions.")
        mean_advantages = (
            advantages.masked_fill(~mask, 0.0).sum(dim=1)
            / mask.sum(dim=1)
        )
        q_values = values[:, None] + advantages - mean_advantages[:, None]
        return q_values.masked_fill(~mask, torch.finfo(q_values.dtype).min)

    @torch.no_grad()
    def action_values(
        self,
        problem: PlacementProblem,
        state: PlacementState,
    ) -> np.ndarray:
        self.eval()
        return self(problem, state).detach().cpu().numpy()
