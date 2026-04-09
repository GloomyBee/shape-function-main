from __future__ import annotations

from typing import Any

import numpy as np


def _gaussian_prior(rel_coords: np.ndarray, beta: float, scale: float) -> np.ndarray:
    radius = np.linalg.norm(rel_coords, axis=1) / max(scale, 1.0e-12)
    return np.exp(-beta * radius * radius)


def _quartic_prior(rel_coords: np.ndarray, beta: float, scale: float) -> np.ndarray:
    support_radius = (1.0 + beta) * max(scale, 1.0e-12)
    q = np.linalg.norm(rel_coords, axis=1) / support_radius
    out = np.zeros_like(q)
    inside = q < 1.0
    out[inside] = (1.0 - q[inside] ** 2) ** 2
    return out


def _prior_weights(rel_coords: np.ndarray, beta: float, prior_type: str, scale: float) -> np.ndarray:
    if prior_type == "gaussian":
        return _gaussian_prior(rel_coords, beta, scale)
    if prior_type == "quartic_spline":
        return _quartic_prior(rel_coords, beta, scale)
    raise ValueError(f"Unsupported prior_type: {prior_type}")


def _objective_terms(rel_coords: np.ndarray, weights: np.ndarray, lam: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    raw = -rel_coords @ lam
    offset = float(np.max(raw))
    z_i = weights * np.exp(raw - offset)
    z = float(np.sum(z_i))
    if not np.isfinite(z) or z <= 0.0:
        raise FloatingPointError("invalid_partition_function")
    phi = z_i / z
    moment = phi @ rel_coords
    hessian = (rel_coords.T * phi) @ rel_coords - np.outer(moment, moment)
    objective = float(np.log(z) + offset)
    return objective, phi, moment, hessian


def solve_maxent_patch(
    x_q: np.ndarray,
    X: np.ndarray,
    beta: float,
    prior_type: str = "gaussian",
    rtol: float = 1.0e-10,
    max_iter: int = 100,
) -> dict[str, Any]:
    x_q = np.asarray(x_q, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("solve_maxent_patch currently supports 2D only")
    rel_coords = X - x_q[None, :]
    scale = float(np.max(np.linalg.norm(rel_coords, axis=1)))
    weights = _prior_weights(rel_coords, float(beta), prior_type=prior_type, scale=scale)
    support_mask = weights > 1.0e-14
    if int(np.sum(support_mask)) < 3:
        return {
            "phi_ref": np.full((X.shape[0],), np.nan, dtype=np.float64),
            "success": False,
            "status": "insufficient_support",
            "n_iter": 0,
            "lambda_opt": np.zeros((2,), dtype=np.float64),
            "objective": None,
            "pou_residual": float("inf"),
            "linear_residual": float("inf"),
            "support_mask": support_mask,
            "diagnostics": {"weights": weights},
        }
    lam = np.zeros((2,), dtype=np.float64)
    objective = None
    success = False
    status = "max_iter_reached"
    phi = np.full((X.shape[0],), np.nan, dtype=np.float64)
    hessian = np.full((2, 2), np.nan, dtype=np.float64)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        try:
            objective, phi, moment, hessian = _objective_terms(rel_coords, weights, lam)
        except FloatingPointError:
            status = "numerical_failure"
            break
        residual_norm = float(np.linalg.norm(moment))
        if residual_norm < rtol:
            success = True
            status = "converged"
            break
        hessian_reg = hessian + 1.0e-12 * np.eye(2, dtype=np.float64)
        if np.linalg.cond(hessian_reg) > 1.0e12:
            status = "ill_conditioned_hessian"
            break
        delta = np.linalg.solve(hessian_reg, moment)
        step = 1.0
        improved = False
        for _ in range(12):
            candidate = lam + step * delta
            _, _, candidate_moment, _ = _objective_terms(rel_coords, weights, candidate)
            if np.linalg.norm(candidate_moment) < residual_norm:
                lam = candidate
                improved = True
                break
            step *= 0.5
        if not improved:
            status = "line_search_failed"
            break
    pou_residual = float(abs(np.sum(phi) - 1.0)) if np.all(np.isfinite(phi)) else float("inf")
    linear_residual = float(np.linalg.norm(phi @ X - x_q)) if np.all(np.isfinite(phi)) else float("inf")
    cond_hessian = float(np.linalg.cond(hessian + 1.0e-12 * np.eye(2))) if np.all(np.isfinite(hessian)) else float("inf")
    return {
        "phi_ref": phi,
        "success": success,
        "status": status,
        "n_iter": int(n_iter),
        "lambda_opt": lam,
        "objective": objective,
        "pou_residual": pou_residual,
        "linear_residual": linear_residual,
        "support_mask": support_mask,
        "diagnostics": {
            "weights": weights,
            "cond_hessian": cond_hessian,
        },
    }
