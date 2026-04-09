from __future__ import annotations

import numpy as np
import torch


def pairwise_distance_matrix(points: np.ndarray) -> np.ndarray:
    diffs = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def covariance_condition_number(points: np.ndarray) -> float:
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = centered.T @ centered / max(points.shape[0], 1)
    singular_values = np.linalg.svd(cov, compute_uv=False)
    min_sv = float(np.min(singular_values))
    max_sv = float(np.max(singular_values))
    if min_sv <= 1.0e-14:
        return float("inf")
    return max_sv / min_sv


def rel_coords_and_radius_torch(X: torch.Tensor, x_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rel = X - x_q[:, None, :]
    dist = torch.linalg.norm(rel, dim=-1)
    r_max = dist.max(dim=-1, keepdim=True).values.clamp_min(1.0e-12)
    rel_hat = rel / r_max.unsqueeze(-1)
    return rel, rel_hat, r_max
