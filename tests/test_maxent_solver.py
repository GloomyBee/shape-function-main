from __future__ import annotations

import numpy as np

from shape_function.data.maxent_solver import solve_maxent_patch


def test_maxent_solver_reproduces_point() -> None:
    X = np.asarray(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.8, 0.8],
        ],
        dtype=np.float64,
    )
    x_q = np.asarray([0.35, 0.45], dtype=np.float64)
    result = solve_maxent_patch(x_q, X, beta=2.0, prior_type="gaussian")
    assert result["success"]
    assert abs(np.sum(result["phi_ref"]) - 1.0) < 1.0e-8
    assert np.linalg.norm(result["phi_ref"] @ X - x_q) < 1.0e-6


def test_maxent_solver_handles_rank_deficient_case() -> None:
    X = np.column_stack([np.linspace(0.0, 1.0, 16), np.zeros((16,))]).astype(np.float64)
    x_q = np.asarray([0.5, 0.0], dtype=np.float64)
    result = solve_maxent_patch(x_q, X, beta=2.0, prior_type="gaussian")
    assert (not result["success"]) or result["diagnostics"]["cond_hessian"] > 1.0e8
