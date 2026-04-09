from __future__ import annotations

from typing import Any

import numpy as np

from shape_function.utils.geometry import covariance_condition_number, pairwise_distance_matrix


def validate_patch(x_q: np.ndarray, X: np.ndarray, min_distance: float = 1.0e-6) -> dict[str, Any]:
    x_q = np.asarray(x_q, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    distances = pairwise_distance_matrix(X)
    distances = distances + np.eye(X.shape[0]) * 1.0e9
    min_pair_distance = float(np.min(distances))
    centered = X - np.mean(X, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(X.shape[0], 1)
    eigvals = np.linalg.eigvalsh(covariance)
    eig_ratio = float(eigvals[0] / eigvals[-1]) if eigvals[-1] > 1.0e-14 else 0.0
    cond_geom = covariance_condition_number(X)
    warnings: list[str] = []
    is_valid = True
    if min_pair_distance < min_distance:
        warnings.append("near_duplicate_nodes")
        is_valid = False
    if not np.isfinite(cond_geom) or cond_geom > 1.0e8:
        warnings.append("degenerate_geometry")
        is_valid = False
    if np.linalg.matrix_rank(X - X[0]) < 2:
        warnings.append("rank_deficient_patch")
        is_valid = False
    return {
        "is_valid": is_valid,
        "warnings": warnings,
        "cond_geom": float(cond_geom),
        "eig_ratio": eig_ratio,
        "min_pair_distance": min_pair_distance,
        "query": x_q,
    }
