from __future__ import annotations

import torch

from shape_function.train.losses import compute_losses


def _outputs() -> dict[str, torch.Tensor]:
    phi_base = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    phi_corr = torch.tensor([[0.2, -0.1, 0.9]], dtype=torch.float64)
    aux = {
        "sum_phi": phi_corr.sum(dim=-1),
        "linear_residual_sq": torch.tensor([0.04], dtype=torch.float64),
    }
    return {"phi_base": phi_base, "phi_corr": phi_corr, "aux": aux}


def test_unsupervised_losses_compute_expected_terms() -> None:
    X = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
    x_q = torch.tensor([[0.25, 0.25]], dtype=torch.float64)
    beta = torch.tensor([[1.0]], dtype=torch.float64)
    losses = compute_losses(
        _outputs(),
        X=X,
        x_q=x_q,
        beta=beta,
        loss_mode="unsupervised_v1",
        lambda_base_lin=1.0,
        lambda_ent=0.1,
        lambda_neg=0.01,
    )
    assert set(losses) == {"total", "loss_base_lin", "loss_ent", "loss_neg"}
    assert float(losses["loss_base_lin"]) > 0.0
    assert torch.isfinite(losses["total"])


def test_legacy_losses_still_work() -> None:
    phi_ref = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float64)
    losses = compute_losses(
        _outputs(),
        phi_ref,
        loss_mode="legacy_teacher_baseline",
        lambda_data=1.0,
        lambda_cons=0.1,
        lambda_neg=0.01,
    )
    assert set(losses) == {"total", "loss_data", "loss_cons", "loss_neg"}
    assert torch.isfinite(losses["total"])
