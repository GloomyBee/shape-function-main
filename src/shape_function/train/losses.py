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


def _gaussian_prior_weights(X: torch.Tensor, x_q: torch.Tensor, beta: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    rel = X - x_q[:, None, :]
    scale = torch.linalg.norm(rel, dim=-1).max(dim=-1, keepdim=True).values.clamp_min(eps)
    radius = torch.linalg.norm(rel, dim=-1) / scale
    beta_flat = beta.squeeze(-1)
    weights = torch.exp(-beta_flat[:, None] * radius.square())
    weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    return weights / weights_sum


def compute_losses(
    outputs: dict[str, torch.Tensor],
    phi_ref: torch.Tensor | None = None,
    *,
    X: torch.Tensor | None = None,
    x_q: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    loss_mode: str = "unsupervised_v1",
    lambda_base_lin: float = 1.0,
    lambda_ent: float = 1.0e-2,
    lambda_neg: float = 1.0e-5,
    lambda_data: float = 1.0,
    lambda_cons: float = 1.0e-4,
    mask: torch.Tensor | None = None,
    valid_count: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    phi_corr = outputs["phi_corr"]
    aux = outputs["aux"]
    if loss_mode == "legacy_teacher_baseline":
        if phi_ref is None:
            raise ValueError("legacy_teacher_baseline requires phi_ref")
        loss_data = _masked_mean((phi_corr - phi_ref) ** 2, mask=mask, valid_count=valid_count)
        loss_cons = torch.mean((aux["sum_phi"] - 1.0) ** 2 + aux["linear_residual_sq"])
        loss_neg = _masked_mean(torch.relu(-phi_corr), mask=mask, valid_count=valid_count)
        total = lambda_data * loss_data + lambda_cons * loss_cons + lambda_neg * loss_neg
        return {
            "total": total,
            "loss_data": loss_data,
            "loss_cons": loss_cons,
            "loss_neg": loss_neg,
        }
    if loss_mode != "unsupervised_v1":
        raise ValueError(f"Unsupported loss_mode: {loss_mode}")
    if X is None or x_q is None or beta is None:
        raise ValueError("unsupervised_v1 requires X, x_q, beta")
    phi_base = outputs["phi_base"]
    base_linear_vec = torch.einsum("bk,bkd->bd", phi_base, X - x_q[:, None, :])
    loss_base_lin = torch.mean(torch.sum(base_linear_vec.square(), dim=-1))
    prior = _gaussian_prior_weights(X, x_q, beta)
    ratio = (phi_base + 1.0e-12) / (prior + 1.0e-12)
    loss_ent = torch.mean(torch.sum(phi_base * torch.log(ratio), dim=-1))
    loss_neg = _masked_mean(torch.relu(-phi_corr), mask=mask, valid_count=valid_count)
    total = lambda_base_lin * loss_base_lin + lambda_ent * loss_ent + lambda_neg * loss_neg
    return {
        "total": total,
        "loss_base_lin": loss_base_lin,
        "loss_ent": loss_ent,
        "loss_neg": loss_neg,
    }
