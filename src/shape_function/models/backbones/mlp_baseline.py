from __future__ import annotations

import torch
import torch.nn as nn


class MLPBaselineBackbone(nn.Module):
    def __init__(self, input_dim: int, k_neighbors: int = 16, hidden_dim: int = 128):
        super().__init__()
        flat_dim = input_dim * k_neighbors
        self.network = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, k_neighbors),
        )
        self.double()

    def forward(self, node_features: torch.Tensor, rel_coords: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        distances = torch.linalg.norm(rel_coords, dim=-1)
        order = torch.argsort(distances, dim=-1)
        gather_idx = order.unsqueeze(-1).expand_as(node_features)
        ordered = torch.gather(node_features, 1, gather_idx)
        logits = self.network(ordered.reshape(node_features.shape[0], -1))
        if mask is not None:
            logits = logits * mask
        return logits
