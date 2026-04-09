from __future__ import annotations

import torch

from shape_function.models.heads.basis import build_linear_basis_2d


def apply_linear_reproducing_correction(
    phi_base: torch.Tensor,
    X: torch.Tensor,
    x_q: torch.Tensor,
    eps_reg: float = 1.0e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    P = build_linear_basis_2d(X)
    p_xq = build_linear_basis_2d(x_q[:, None, :]).squeeze(1)
    moment = torch.einsum("bk,bki,bkj->bij", phi_base, P, P)
    identity = torch.eye(3, dtype=X.dtype, device=X.device).unsqueeze(0)
    moment_reg = moment + eps_reg * identity
    coeff = torch.linalg.solve(moment_reg, p_xq)
    correction = torch.einsum("bki,bi->bk", P, coeff)
    phi_corr = phi_base * correction
    singular_values = torch.linalg.svdvals(moment_reg)
    cond_m = singular_values[..., 0] / singular_values[..., -1].clamp_min(eps_reg)
    return phi_corr, moment_reg, cond_m
