from __future__ import annotations

import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None, valid_count: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return torch.mean(values)
    mask = mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    denom = valid_count
    if denom is None:
        denom = mask.sum()
    if not torch.is_tensor(denom):
        denom = torch.as_tensor(denom, dtype=values.dtype, device=values.device)
    denom = denom.to(values.dtype).clamp_min(1.0)
    return torch.sum(values * mask) / denom


def compute_losses(
    outputs: dict[str, torch.Tensor],
    phi_ref: torch.Tensor,
    lambda_cons: float = 1.0e-4,
    lambda_neg: float = 1.0e-5,
    mask: torch.Tensor | None = None,
    valid_count: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    phi_corr = outputs["phi_corr"]
    aux = outputs["aux"]
    loss_data = _masked_mean((phi_corr - phi_ref) ** 2, mask=mask, valid_count=valid_count)
    loss_cons = torch.mean((aux["sum_phi"] - 1.0) ** 2 + aux["linear_residual_sq"])
    loss_neg = _masked_mean(torch.relu(-phi_corr), mask=mask, valid_count=valid_count)
    total = loss_data + lambda_cons * loss_cons + lambda_neg * loss_neg
    return {
        "total": total,
        "loss_data": loss_data,
        "loss_cons": loss_cons,
        "loss_neg": loss_neg,
    }
