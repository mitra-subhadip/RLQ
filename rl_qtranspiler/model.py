"""Edge-aware graph DQN implemented with core PyTorch only."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    PHYSICAL_FEATURES = 13
    LOGICAL_FEATURES = 5

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.physical_encoder = GraphEncoder(
            self.PHYSICAL_FEATURES, 1, hidden_dim
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
        physical_features, logical_features = build_state_features(
            problem, state
        )
        physical_x = self._tensor(physical_features, dtype=torch.float32)
        logical_x = self._tensor(logical_features, dtype=torch.float32)
        physical_edges = self._tensor(
            problem.hardware.directed_edge_index, dtype=torch.long
        )
        physical_edge_features = self._tensor(
            problem.hardware.directed_edge_features, dtype=torch.float32
        )
        logical_edges = self._tensor(
            problem.logical_edge_index, dtype=torch.long
        )
        logical_edge_features = self._tensor(
            problem.logical_edge_features, dtype=torch.float32
        )

        physical_embeddings = self.physical_encoder(
            physical_x, physical_edges, physical_edge_features
        )
        logical_embeddings = self.logical_encoder(
            logical_x, logical_edges, logical_edge_features
        )
        current_logical = problem.placement_order[state.step_index]
        current_embedding = logical_embeddings[current_logical]
        physical_pool = physical_embeddings.mean(dim=0)
        logical_pool = logical_embeddings.mean(dim=0)

        state_embedding = torch.cat(
            [current_embedding, physical_pool, logical_pool]
        )
        value = self.value_head(state_embedding).squeeze(-1)
        num_physical = problem.hardware.num_qubits
        candidate_input = torch.cat(
            [
                physical_embeddings,
                current_embedding.expand(num_physical, -1),
                physical_pool.expand(num_physical, -1),
                logical_pool.expand(num_physical, -1),
                physical_x,
            ],
            dim=-1,
        )
        advantages = self.advantage_head(candidate_input).squeeze(-1)

        mask_array = (
            valid_action_mask
            if valid_action_mask is not None
            else problem.valid_action_mask(state.physical_to_logical)
        )
        mask = self._tensor(mask_array, dtype=torch.bool)
        if not torch.any(mask):
            raise ValueError("State has no valid physical-qubit actions.")
        centered_advantages = advantages - advantages[mask].mean()
        q_values = value + centered_advantages
        return q_values.masked_fill(~mask, torch.finfo(q_values.dtype).min)

    @torch.no_grad()
    def action_values(
        self,
        problem: PlacementProblem,
        state: PlacementState,
    ) -> np.ndarray:
        self.eval()
        return self(problem, state).detach().cpu().numpy()
