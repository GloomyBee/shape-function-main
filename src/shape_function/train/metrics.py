from __future__ import annotations

import torch


def compute_batch_metrics(
    outputs: dict[str, torch.Tensor],
    phi_ref: torch.Tensor | None = None,
    *,
    compute_teacher_metrics: bool = False,
) -> dict[str, float]:
    aux = outputs["aux"]
    metrics = {
        "base_linear_residual": float(torch.mean(aux["base_linear_residual"]).detach().cpu()),
        "mean_pou_residual": float(torch.mean(torch.abs(aux["sum_phi"] - 1.0)).detach().cpu()),
        "mean_linear_residual": float(torch.mean(aux["linear_residual"]).detach().cpu()),
        "mean_reproducing_residual": float(
            torch.mean(
                torch.maximum(
                    aux["reproducing_residual_const"],
                    torch.maximum(aux["reproducing_residual_linear"], aux["reproducing_residual_quadratic"]),
                )
            ).detach().cpu()
        ),
        "mean_quad_residual": float(torch.mean(aux["reproducing_residual_quadratic"]).detach().cpu()),
        "negative_fraction": float(torch.mean(aux["neg_fraction"]).detach().cpu()),
        "max_negative_magnitude": float(torch.max(aux["max_neg_mag"]).detach().cpu()),
        "fallback_rate": float(aux["fallback_rate_batch"].detach().cpu()),
        "mean_cond_M": float(torch.mean(aux["cond_M"]).detach().cpu()),
        "p95_cond_M": float(torch.quantile(aux["cond_M"], 0.95).detach().cpu()),
        "worst_cond_M": float(torch.max(aux["cond_M"]).detach().cpu()),
    }
    if phi_ref is not None and compute_teacher_metrics:
        phi_corr = outputs["phi_corr"]
        diff = phi_corr - phi_ref
        denom = torch.linalg.norm(phi_ref, dim=-1).clamp_min(1.0e-12)
        relative_l2 = torch.mean(torch.linalg.norm(diff, dim=-1) / denom)
        metrics.update(
            {
                "relative_l2": float(relative_l2.detach().cpu()),
                "global_linf": float(torch.max(torch.abs(diff)).detach().cpu()),
                "teacher_quad_residual": float(torch.mean(aux["reproducing_residual_quadratic"]).detach().cpu()),
                "quad_gain": 0.0,
            }
        )
    return metrics
