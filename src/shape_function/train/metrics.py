from __future__ import annotations

import torch

from shape_function.models.heads.basis import build_query_centered_polynomial_basis_2d


def _teacher_second_order_residual(
    phi_ref: torch.Tensor,
    X: torch.Tensor,
    x_q: torch.Tensor,
    r_max: torch.Tensor,
) -> torch.Tensor:
    basis = build_query_centered_polynomial_basis_2d(X, x_q, r_max, order=2)
    reproduced = torch.einsum("bk,bki->bi", phi_ref, basis)
    target = torch.zeros_like(reproduced)
    target[..., 0] = 1.0
    return torch.linalg.norm(reproduced - target, dim=-1)


def _add_fallback_by_type(metrics: dict[str, float], fallback_mask: torch.Tensor, patch_type) -> None:
    if patch_type is None:
        return
    labels = list(patch_type)
    if not labels:
        return
    fallback_cpu = fallback_mask.detach().cpu().to(torch.float64)
    for label in sorted(set(labels)):
        indices = [idx for idx, item in enumerate(labels) if item == label]
        if not indices:
            continue
        selected = fallback_cpu[torch.as_tensor(indices, dtype=torch.long)]
        safe_label = str(label).replace(" ", "_").replace("-", "_")
        metrics[f"fallback_rate_by_type_{safe_label}"] = float(torch.mean(selected).item())


def compute_batch_metrics(
    outputs: dict[str, torch.Tensor],
    phi_ref: torch.Tensor | None = None,
    *,
    X: torch.Tensor | None = None,
    x_q: torch.Tensor | None = None,
    r_max: torch.Tensor | None = None,
    patch_type=None,
    compute_teacher_metrics: bool = False,
) -> dict[str, float]:
    aux = outputs["aux"]
    mean_quad_residual = torch.mean(aux["reproducing_residual_quadratic"])
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
        "mean_quad_residual": float(mean_quad_residual.detach().cpu()),
        "negative_fraction": float(torch.mean(aux["neg_fraction"]).detach().cpu()),
        "max_negative_magnitude": float(torch.max(aux["max_neg_mag"]).detach().cpu()),
        "fallback_rate": float(aux["fallback_rate_batch"].detach().cpu()),
        "mean_cond_M": float(torch.mean(aux["cond_M"]).detach().cpu()),
        "p95_cond_M": float(torch.quantile(aux["cond_M"], 0.95).detach().cpu()),
        "worst_cond_M": float(torch.max(aux["cond_M"]).detach().cpu()),
    }
    _add_fallback_by_type(metrics, aux["fallback_mask"], patch_type)
    if phi_ref is not None and compute_teacher_metrics:
        if X is None or x_q is None or r_max is None:
            raise ValueError("teacher metrics require X, x_q, and r_max")
        phi_corr = outputs["phi_corr"]
        diff = phi_corr - phi_ref
        denom = torch.linalg.norm(phi_ref, dim=-1).clamp_min(1.0e-12)
        relative_l2 = torch.mean(torch.linalg.norm(diff, dim=-1) / denom)
        teacher_quad_residual = torch.mean(_teacher_second_order_residual(phi_ref, X, x_q, r_max))
        metrics.update(
            {
                "relative_l2": float(relative_l2.detach().cpu()),
                "global_linf": float(torch.max(torch.abs(diff)).detach().cpu()),
                "teacher_quad_residual": float(teacher_quad_residual.detach().cpu()),
                "quad_gain": float((teacher_quad_residual - mean_quad_residual).detach().cpu()),
            }
        )
    return metrics
