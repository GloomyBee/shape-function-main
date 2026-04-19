from __future__ import annotations

import torch

from shape_function.train.metrics import compute_batch_metrics


def test_second_order_metrics_aggregate_structure_terms() -> None:
    aux = {
        "base_linear_residual": torch.tensor([0.1, 0.3], dtype=torch.float64),
        "sum_phi": torch.tensor([1.0, 1.0], dtype=torch.float64),
        "linear_residual": torch.tensor([1.0e-12, 2.0e-12], dtype=torch.float64),
        "reproducing_residual_const": torch.tensor([1.0e-12, 2.0e-12], dtype=torch.float64),
        "reproducing_residual_linear": torch.tensor([3.0e-12, 4.0e-12], dtype=torch.float64),
        "reproducing_residual_quadratic": torch.tensor([5.0e-12, 6.0e-12], dtype=torch.float64),
        "neg_fraction": torch.tensor([0.0, 0.25], dtype=torch.float64),
        "max_neg_mag": torch.tensor([0.0, 0.1], dtype=torch.float64),
        "fallback_rate_batch": torch.tensor(0.5, dtype=torch.float64),
        "cond_M": torch.tensor([10.0, 100.0], dtype=torch.float64),
    }
    metrics = compute_batch_metrics({"aux": aux, "phi_corr": torch.ones(2, 3, dtype=torch.float64) / 3.0})
    assert "mean_quad_residual" in metrics
    assert "fallback_rate" in metrics
    assert metrics["fallback_rate"] == 0.5
    assert metrics["worst_cond_M"] == 100.0
