from __future__ import annotations

import torch

from shape_function.train.metrics import compute_batch_metrics


def _base_aux() -> dict[str, torch.Tensor]:
    return {
        "base_linear_residual": torch.tensor([0.1, 0.3], dtype=torch.float64),
        "sum_phi": torch.tensor([1.0, 1.0], dtype=torch.float64),
        "linear_residual": torch.tensor([1.0e-12, 2.0e-12], dtype=torch.float64),
        "reproducing_residual_const": torch.tensor([1.0e-12, 2.0e-12], dtype=torch.float64),
        "reproducing_residual_linear": torch.tensor([3.0e-12, 4.0e-12], dtype=torch.float64),
        "reproducing_residual_quadratic": torch.tensor([0.1, 0.2], dtype=torch.float64),
        "neg_fraction": torch.tensor([0.0, 0.25], dtype=torch.float64),
        "max_neg_mag": torch.tensor([0.0, 0.1], dtype=torch.float64),
        "fallback_rate_batch": torch.tensor(0.5, dtype=torch.float64),
        "fallback_mask": torch.tensor([False, True]),
        "cond_M": torch.tensor([10.0, 100.0], dtype=torch.float64),
    }


def test_second_order_metrics_aggregate_structure_terms() -> None:
    metrics = compute_batch_metrics({"aux": _base_aux(), "phi_corr": torch.ones(2, 3, dtype=torch.float64) / 3.0})
    assert "mean_quad_residual" in metrics
    assert "fallback_rate" in metrics
    assert metrics["fallback_rate"] == 0.5
    assert metrics["worst_cond_M"] == 100.0


def test_teacher_quad_residual_and_quad_gain_are_independent_of_student_aux() -> None:
    X = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]],
        dtype=torch.float64,
    )
    x_q = torch.zeros((2, 2), dtype=torch.float64)
    r_max = torch.linalg.norm(X - x_q[:, None, :], dim=-1).max(dim=-1, keepdim=True).values
    phi_ref = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.25, 0.25]], dtype=torch.float64)
    phi_corr = torch.ones((2, 3), dtype=torch.float64) / 3.0
    metrics = compute_batch_metrics(
        {"aux": _base_aux(), "phi_corr": phi_corr},
        phi_ref,
        X=X,
        x_q=x_q,
        r_max=r_max,
        compute_teacher_metrics=True,
    )
    assert metrics["teacher_quad_residual"] > 0.0
    assert abs(metrics["quad_gain"] - (metrics["teacher_quad_residual"] - metrics["mean_quad_residual"])) < 1.0e-12


def test_fallback_rate_is_reported_by_patch_type() -> None:
    metrics = compute_batch_metrics(
        {"aux": _base_aux(), "phi_corr": torch.ones(2, 3, dtype=torch.float64) / 3.0},
        patch_type=["uniform", "clustered"],
    )
    assert metrics["fallback_rate_by_type_uniform"] == 0.0
    assert metrics["fallback_rate_by_type_clustered"] == 1.0
