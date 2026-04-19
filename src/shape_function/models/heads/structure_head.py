from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from shape_function.models.heads.reproducing_correction import apply_reproducing_correction
from shape_function.models.heads.windows import quartic_spline_window, wendland_c2_window


class StructurePreservingHead(nn.Module):
    def __init__(
        self,
        window_type: str = "quartic",
        eps: float = 1.0e-12,
        eps_reg: float = 1.0e-10,
        support_radius_scale: float = 1.05,
        basis_order: int = 1,
        kappa_max: float | None = None,
        fallback_mode: str = "hard",
    ):
        super().__init__()
        self.window_type = window_type
        self.eps = eps
        self.eps_reg = eps_reg
        self.support_radius_scale = support_radius_scale
        self.basis_order = basis_order
        self.kappa_max = kappa_max
        self.fallback_mode = fallback_mode

    def _window(self, s: torch.Tensor) -> torch.Tensor:
        if self.window_type == "quartic":
            return quartic_spline_window(s)
        if self.window_type == "wendland_c2":
            return wendland_c2_window(s)
        raise ValueError(f"Unsupported window_type: {self.window_type}")

    def forward(
        self,
        logits: torch.Tensor,
        X: torch.Tensor,
        x_q: torch.Tensor,
        r_max: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        raw = F.softplus(logits)
        dist = torch.linalg.norm(X - x_q[:, None, :], dim=-1)
        support_radius = (self.support_radius_scale * r_max).clamp_min(self.eps)
        window = self._window(dist / support_radius)
        if mask is not None:
            raw = raw * mask
            window = window * mask
        weighted = raw * window
        denominator = weighted.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        phi_base = weighted / denominator
        correction = apply_reproducing_correction(
            phi_base,
            X,
            x_q,
            r_max,
            basis_order=self.basis_order,
            eps_reg=self.eps_reg,
            kappa_max=self.kappa_max,
            fallback_mode=self.fallback_mode,
        )
        phi_corr = correction["phi_corr"]
        sum_phi = phi_corr.sum(dim=-1)
        linear_moment = torch.einsum("bk,bkd->bd", phi_corr, X)
        linear_error = linear_moment - x_q
        linear_residual = torch.linalg.norm(linear_error, dim=-1)
        neg_part = torch.relu(-phi_corr)
        base_linear_vec = torch.einsum("bk,bkd->bd", phi_base, X - x_q[:, None, :])
        base_linear_residual = torch.linalg.norm(base_linear_vec, dim=-1)
        aux = {
            "sum_phi": sum_phi,
            "linear_residual": linear_residual,
            "linear_residual_sq": linear_residual.square(),
            "neg_fraction": (neg_part > 0.0).to(phi_corr.dtype).mean(dim=-1),
            "max_neg_mag": neg_part.max(dim=-1).values,
            "cond_M": correction["cond_M"],
            "cond_M2": correction["cond_M2"],
            "denominator_min": denominator.squeeze(-1),
            "fallback_mask": correction["fallback_mask"],
            "fallback_rate_batch": correction["fallback_mask"].to(phi_corr.dtype).mean(),
            "reproducing_residual_const": correction["reproducing_residual_const"],
            "reproducing_residual_linear": correction["reproducing_residual_linear"],
            "reproducing_residual_quadratic": correction["reproducing_residual_quadratic"],
            "base_linear_residual": base_linear_residual,
            "negative_fraction_2nd": (neg_part > 0.0).to(phi_corr.dtype).mean(dim=-1),
            "max_negative_magnitude_2nd": neg_part.max(dim=-1).values,
        }
        return phi_base, phi_corr, aux
