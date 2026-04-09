from __future__ import annotations

import torch


def compute_losses(
    outputs: dict[str, torch.Tensor],
    phi_ref: torch.Tensor,
    lambda_cons: float = 1.0e-4,
    lambda_neg: float = 1.0e-5,
) -> dict[str, torch.Tensor]:
    phi_corr = outputs["phi_corr"]
    aux = outputs["aux"]
    loss_data = torch.mean((phi_corr - phi_ref) ** 2)
    loss_cons = torch.mean((aux["sum_phi"] - 1.0) ** 2 + aux["linear_residual_sq"])
    loss_neg = torch.relu(-phi_corr).mean()
    total = loss_data + lambda_cons * loss_cons + lambda_neg * loss_neg
    return {
        "total": total,
        "loss_data": loss_data,
        "loss_cons": loss_cons,
        "loss_neg": loss_neg,
    }
