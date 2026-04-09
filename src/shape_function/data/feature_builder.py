from __future__ import annotations

from typing import Any

import numpy as np


def _mean_nearest_neighbor_distance(points: np.ndarray) -> float:
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    dists += np.eye(points.shape[0]) * 1.0e9
    return float(np.min(dists, axis=1).mean())


def build_patch_features(patch: dict[str, Any], feature_mode: str = "minimal", k_max: int = 16) -> dict[str, np.ndarray]:
    x_q = np.asarray(patch["x_q"], dtype=np.float64)
    X = np.asarray(patch["X"], dtype=np.float64)
    beta = float(patch["beta"])
    rel = X - x_q[None, :]
    r_max = float(patch["r_max"])
    rel_hat = rel / max(r_max, 1.0e-12)
    dist_hat = np.linalg.norm(rel_hat, axis=1, keepdims=True)
    base = np.concatenate([rel_hat, dist_hat, np.full((X.shape[0], 1), beta, dtype=np.float64)], axis=1)
    if feature_mode == "minimal":
        rho_q = np.zeros((0,), dtype=np.float64)
        node_features = base
    elif feature_mode == "enhanced":
        centered = rel_hat - np.mean(rel_hat, axis=0, keepdims=True)
        cov = centered.T @ centered / max(X.shape[0], 1)
        eigvals = np.linalg.eigvalsh(cov)
        eig_ratio = float(eigvals[0] / eigvals[-1]) if eigvals[-1] > 1.0e-14 else 0.0
        mean_nn = _mean_nearest_neighbor_distance(X) / max(r_max, 1.0e-12)
        rho_q = np.asarray([mean_nn, X.shape[0] / float(k_max), eig_ratio], dtype=np.float64)
        repeated = np.repeat(rho_q[None, :], X.shape[0], axis=0)
        node_features = np.concatenate([base, repeated], axis=1)
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")
    return {"node_features": node_features, "rho_q": rho_q, "rel_coords": rel_hat}
