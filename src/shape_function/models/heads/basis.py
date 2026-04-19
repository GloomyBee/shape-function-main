from __future__ import annotations

import torch


def build_linear_basis_2d(X: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(X[..., :1])
    return torch.cat([ones, X], dim=-1)


def build_query_centered_polynomial_basis_2d(
    X: torch.Tensor,
    x_q: torch.Tensor,
    r_max: torch.Tensor,
    order: int,
) -> torch.Tensor:
    if order not in (1, 2):
        raise ValueError(f"Unsupported polynomial order: {order}")
    scale = r_max.clamp_min(1.0e-12)
    if scale.ndim == X.ndim - 1:
        scale = scale.unsqueeze(-1)
    rel = (X - x_q[:, None, :]) / scale
    x_tilde = rel[..., 0:1]
    y_tilde = rel[..., 1:2]
    ones = torch.ones_like(x_tilde)
    if order == 1:
        return torch.cat([ones, x_tilde, y_tilde], dim=-1)
    return torch.cat(
        [
            ones,
            x_tilde,
            y_tilde,
            x_tilde.square(),
            x_tilde * y_tilde,
            y_tilde.square(),
        ],
        dim=-1,
    )
