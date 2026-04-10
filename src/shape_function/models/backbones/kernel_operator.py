from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from shape_function.models.heads.windows import wendland_c2_window


class KernelOperatorBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        spatial_dim: int = 2,
        kernel_radius_scale: float = 2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.spatial_dim = spatial_dim
        self.kernel_radius_scale = kernel_radius_scale
        self.lift = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        pair_dim = hidden_dim + self.spatial_dim + self.spatial_dim + 1
        self.modulation = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(pair_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_layers)
            ]
        )
        self.value_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.update_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, 1)
        self.double()

    def forward(self, node_features: torch.Tensor, rel_coords: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if rel_coords.shape[-1] != self.spatial_dim:
            raise ValueError(
                f"Expected rel_coords last dimension {self.spatial_dim}, got {rel_coords.shape[-1]}"
            )
        h = self.lift(node_features)
        for layer_idx in range(self.num_layers):
            h_i = h.unsqueeze(2)
            h_j = h.unsqueeze(1)
            r_i = rel_coords.unsqueeze(2)
            r_j = rel_coords.unsqueeze(1)
            pair_dist = torch.linalg.norm(r_i - r_j, dim=-1, keepdim=True)
            pair_feat = torch.cat(
                [
                    h_i - h_j,
                    r_i.expand(-1, -1, rel_coords.shape[1], -1),
                    r_j.expand(-1, rel_coords.shape[1], -1, -1),
                    pair_dist,
                ],
                dim=-1,
            )
            modulation = F.softplus(self.modulation[layer_idx](pair_feat).squeeze(-1))
            kernel = wendland_c2_window(
                (pair_dist.squeeze(-1) / self.kernel_radius_scale).clamp_min(0.0)
            )
            alpha = modulation * kernel
            if mask is not None:
                pair_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
                alpha = alpha * pair_mask
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
            values = self.value_layers[layer_idx](h)
            message = torch.einsum("bij,bjc->bic", alpha, values)
            h = h + F.gelu(self.update_layers[layer_idx](h) + message)
        logits = self.out(h).squeeze(-1)
        if mask is not None:
            logits = logits * mask
        return logits
