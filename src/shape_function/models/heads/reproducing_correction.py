from __future__ import annotations

import torch

from shape_function.models.heads.basis import build_query_centered_polynomial_basis_2d


def _basis_eval_at_query(batch_size: int, order: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if order == 1:
        return torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).expand(batch_size, -1)
    if order == 2:
        return torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype, device=device).expand(batch_size, -1)
    raise ValueError(f"Unsupported polynomial order: {order}")


def _solve_reproducing_system(
    phi_base: torch.Tensor,
    X: torch.Tensor,
    x_q: torch.Tensor,
    r_max: torch.Tensor,
    order: int,
    eps_reg: float,
) -> dict[str, torch.Tensor]:
    basis = build_query_centered_polynomial_basis_2d(X, x_q, r_max, order=order)
    p_xq = _basis_eval_at_query(X.shape[0], order=order, dtype=X.dtype, device=X.device)
    moment = torch.einsum("bk,bki,bkj->bij", phi_base, basis, basis)
    identity = torch.eye(moment.shape[-1], dtype=X.dtype, device=X.device).unsqueeze(0)
    moment_reg = moment + eps_reg * identity
    coeff = torch.linalg.solve(moment_reg, p_xq)
    correction = torch.einsum("bki,bi->bk", basis, coeff)
    phi_corr = phi_base * correction
    reproduced = torch.einsum("bk,bki->bi", phi_corr, basis)
    residual = reproduced - p_xq
    singular_values = torch.linalg.svdvals(moment_reg)
    cond_m = singular_values[..., 0] / singular_values[..., -1].clamp_min(eps_reg)
    return {
        "phi_corr": phi_corr,
        "moment": moment_reg,
        "cond_M": cond_m,
        "residual_vector": residual,
    }


def _final_reproducing_residuals(
    phi: torch.Tensor,
    X: torch.Tensor,
    x_q: torch.Tensor,
    r_max: torch.Tensor,
) -> dict[str, torch.Tensor]:
    basis = build_query_centered_polynomial_basis_2d(X, x_q, r_max, order=2)
    target = _basis_eval_at_query(X.shape[0], order=2, dtype=X.dtype, device=X.device)
    residual = torch.einsum("bk,bki->bi", phi, basis) - target
    return {
        "const": torch.abs(residual[..., 0]),
        "linear": torch.linalg.norm(residual[..., 1:3], dim=-1),
        "quadratic": torch.linalg.norm(residual[..., 3:], dim=-1),
    }


def apply_linear_reproducing_correction(
    phi_base: torch.Tensor,
    X: torch.Tensor,
    x_q: torch.Tensor,
    eps_reg: float = 1.0e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r_max = torch.linalg.norm(X - x_q[:, None, :], dim=-1).max(dim=-1, keepdim=True).values.clamp_min(1.0e-12)
    solved = _solve_reproducing_system(phi_base, X, x_q, r_max, order=1, eps_reg=eps_reg)
    return solved["phi_corr"], solved["moment"], solved["cond_M"]


def apply_reproducing_correction(
    phi_base: torch.Tensor,
    X: torch.Tensor,
    x_q: torch.Tensor,
    r_max: torch.Tensor,
    basis_order: int = 1,
    eps_reg: float = 1.0e-10,
    kappa_max: float | None = None,
    fallback_mode: str = "hard",
) -> dict[str, torch.Tensor]:
    if basis_order not in (1, 2):
        raise ValueError(f"Unsupported basis_order: {basis_order}")
    if fallback_mode != "hard":
        raise ValueError(f"Unsupported fallback_mode: {fallback_mode}")

    order1 = _solve_reproducing_system(phi_base, X, x_q, r_max, order=1, eps_reg=eps_reg)
    if basis_order == 1:
        residuals = _final_reproducing_residuals(order1["phi_corr"], X, x_q, r_max)
        return {
            "phi_corr": order1["phi_corr"],
            "phi_corr_order1": order1["phi_corr"],
            "moment": order1["moment"],
            "moment_order1": order1["moment"],
            "cond_M": order1["cond_M"],
            "cond_M2": order1["cond_M"],
            "fallback_mask": torch.zeros_like(order1["cond_M"], dtype=torch.bool),
            "reproducing_residual_const": residuals["const"],
            "reproducing_residual_linear": residuals["linear"],
            "reproducing_residual_quadratic": residuals["quadratic"],
        }

    if kappa_max is None:
        raise ValueError("basis_order=2 requires kappa_max")
    order2 = _solve_reproducing_system(phi_base, X, x_q, r_max, order=2, eps_reg=eps_reg)
    fallback_mask = order2["cond_M"] > float(kappa_max)
    phi_corr = torch.where(fallback_mask[:, None], order1["phi_corr"], order2["phi_corr"])
    residuals = _final_reproducing_residuals(phi_corr, X, x_q, r_max)
    return {
        "phi_corr": phi_corr,
        "phi_corr_order1": order1["phi_corr"],
        "moment": order2["moment"],
        "moment_order1": order1["moment"],
        "cond_M": order2["cond_M"],
        "cond_M2": order2["cond_M"],
        "fallback_mask": fallback_mask,
        "reproducing_residual_const": residuals["const"],
        "reproducing_residual_linear": residuals["linear"],
        "reproducing_residual_quadratic": residuals["quadratic"],
    }
