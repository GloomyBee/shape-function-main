from __future__ import annotations

import torch


def build_linear_basis_2d(X: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(X[..., :1])
    return torch.cat([ones, X], dim=-1)
