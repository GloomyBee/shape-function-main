from __future__ import annotations

import torch


def compute_batch_metrics(outputs: dict[str, torch.Tensor], phi_ref: torch.Tensor) -> dict[str, float]:
    phi_corr = outputs["phi_corr"]
    diff = phi_corr - phi_ref
    denom = torch.linalg.norm(phi_ref, dim=-1).clamp_min(1.0e-12)
    relative_l2 = torch.mean(torch.linalg.norm(diff, dim=-1) / denom)
    aux = outputs["aux"]
    return {
        "relative_l2": float(relative_l2.detach().cpu()),
        "global_linf": float(torch.max(torch.abs(diff)).detach().cpu()),
        "mean_pou_residual": float(torch.mean(torch.abs(aux["sum_phi"] - 1.0)).detach().cpu()),
        "mean_linear_residual": float(torch.mean(aux["linear_residual"]).detach().cpu()),
        "negative_fraction": float(torch.mean(aux["neg_fraction"]).detach().cpu()),
        "max_negative_magnitude": float(torch.max(aux["max_neg_mag"]).detach().cpu()),
        "mean_cond_M": float(torch.mean(aux["cond_M"]).detach().cpu()),
        "worst_cond_M": float(torch.max(aux["cond_M"]).detach().cpu()),
    }
