from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from shape_function.models.heads.reproducing_correction import apply_linear_reproducing_correction
from shape_function.models.heads.windows import quartic_spline_window, wendland_c2_window


class StructurePreservingHead(nn.Module):
    def __init__(self, window_type: str = "quartic", eps: float = 1.0e-12, eps_reg: float = 1.0e-10):
        super().__init__()
        self.window_type = window_type
        self.eps = eps
        self.eps_reg = eps_reg

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
        window = self._window(dist / r_max.clamp_min(self.eps))
        if mask is not None:
            raw = raw * mask
            window = window * mask
        weighted = raw * window
        denominator = weighted.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        phi_base = weighted / denominator
        phi_corr, _, cond_m = apply_linear_reproducing_correction(phi_base, X, x_q, eps_reg=self.eps_reg)
        sum_phi = phi_corr.sum(dim=-1)
        linear_moment = torch.einsum("bk,bkd->bd", phi_corr, X)
        linear_error = linear_moment - x_q
        linear_residual = torch.linalg.norm(linear_error, dim=-1)
        neg_part = torch.relu(-phi_corr)
        aux = {
            "sum_phi": sum_phi,
            "linear_residual": linear_residual,
            "linear_residual_sq": linear_residual.square(),
            "neg_fraction": (neg_part > 0.0).to(phi_corr.dtype).mean(dim=-1),
            "max_neg_mag": neg_part.max(dim=-1).values,
            "cond_M": cond_m,
            "denominator_min": denominator.squeeze(-1),
        }
        return phi_base, phi_corr, aux
