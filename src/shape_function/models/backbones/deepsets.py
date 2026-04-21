from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU()])
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class DeepSetsBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        use_moment_summary: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.use_moment_summary = use_moment_summary
        self.encoder = _mlp(input_dim, hidden_dim, hidden_dim, num_encoder_layers)
        moment_dim = 6 if use_moment_summary else 0
        self.decoder = _mlp(input_dim + hidden_dim + hidden_dim + moment_dim, hidden_dim, 1, num_decoder_layers)
        self.double()

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return values.mean(dim=1)
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _moment_summary(rel_coords: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = rel_coords[..., 0:1]
        y = rel_coords[..., 1:2]
        moments = torch.cat([x, y, x.square(), x * y, y.square(), torch.linalg.norm(rel_coords, dim=-1, keepdim=True)], dim=-1)
        return DeepSetsBackbone._masked_mean(moments, mask)

    def forward(self, node_features: torch.Tensor, rel_coords: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [B, K, F]")
        h = self.encoder(node_features)
        pooled = self._masked_mean(h, mask)
        pieces = [node_features, h, pooled[:, None, :].expand(-1, node_features.shape[1], -1)]
        if self.use_moment_summary:
            summary = self._moment_summary(rel_coords, mask)
            pieces.append(summary[:, None, :].expand(-1, node_features.shape[1], -1))
        logits = self.decoder(torch.cat(pieces, dim=-1)).squeeze(-1)
        if mask is not None:
            logits = logits * mask.to(logits.dtype)
        return logits
