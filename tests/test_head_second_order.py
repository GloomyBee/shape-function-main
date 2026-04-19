from __future__ import annotations

import torch

from shape_function.models.heads.reproducing_correction import apply_reproducing_correction


def _sample_patch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        [[
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [0.0, -1.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ]],
        dtype=torch.float64,
    )
    x_q = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    r_max = torch.linalg.norm(X - x_q[:, None, :], dim=-1).max(dim=-1, keepdim=True).values
    phi_base = torch.full((1, X.shape[1]), 1.0 / X.shape[1], dtype=torch.float64)
    return phi_base, X, x_q, r_max


def test_second_order_reproducing_residual_is_machine_small() -> None:
    phi_base, X, x_q, r_max = _sample_patch()
    result = apply_reproducing_correction(
        phi_base,
        X,
        x_q,
        r_max,
        basis_order=2,
        kappa_max=1.0e12,
        fallback_mode="hard",
        eps_reg=1.0e-14,
    )
    assert torch.max(result["reproducing_residual_const"]).item() < 1.0e-10
    assert torch.max(result["reproducing_residual_linear"]).item() < 1.0e-10
    assert torch.max(result["reproducing_residual_quadratic"]).item() < 1.0e-10
    assert not bool(result["fallback_mask"].any().item())


def test_second_order_fallback_returns_order1_when_cond_is_too_large() -> None:
    phi_base, X, x_q, r_max = _sample_patch()
    result = apply_reproducing_correction(
        phi_base,
        X,
        x_q,
        r_max,
        basis_order=2,
        kappa_max=1.0,
        fallback_mode="hard",
        eps_reg=1.0e-14,
    )
    assert bool(result["fallback_mask"].all().item())
    assert torch.allclose(result["phi_corr"], result["phi_corr_order1"])


def test_reproducing_correction_does_not_renormalize_after_b4() -> None:
    phi_base, X, x_q, r_max = _sample_patch()
    result = apply_reproducing_correction(
        phi_base,
        X,
        x_q,
        r_max,
        basis_order=2,
        kappa_max=1.0e12,
        fallback_mode="hard",
        eps_reg=1.0e-14,
    )
    assert torch.allclose(result["phi_corr"].sum(dim=-1), torch.ones(1, dtype=torch.float64), atol=1.0e-10)
