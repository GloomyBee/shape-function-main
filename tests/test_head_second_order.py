from __future__ import annotations

import torch

from shape_function.data.patch_sampler import PATCH_TYPES, sample_patches
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


def _uniform_base_from_patch(patch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X = torch.as_tensor(patch["X"][None, ...], dtype=torch.float64)
    x_q = torch.as_tensor(patch["x_q"][None, ...], dtype=torch.float64)
    r_max = torch.as_tensor([[patch["r_max"]]], dtype=torch.float64)
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
        eps_reg=1.0e-16,
    )
    assert torch.max(result["reproducing_residual_const"]).item() < 1.0e-10
    assert torch.max(result["reproducing_residual_linear"]).item() < 1.0e-10
    assert torch.max(result["reproducing_residual_quadratic"]).item() < 1.0e-10
    assert not bool(result["fallback_mask"].any().item())


def test_second_order_reproduces_on_all_sampler_patch_types() -> None:
    patches = sample_patches(
        num_patches=len(PATCH_TYPES) * 3,
        seed=13,
        patch_types=PATCH_TYPES,
        k_neighbors=16,
        beta_range=(1.0, 1.0),
    )
    seen = set()
    for patch in patches:
        phi_base, X, x_q, r_max = _uniform_base_from_patch(patch)
        result = apply_reproducing_correction(
            phi_base,
            X,
            x_q,
            r_max,
            basis_order=2,
            kappa_max=1.0e14,
            fallback_mode="hard",
            eps_reg=1.0e-16,
        )
        if bool(result["fallback_mask"].item()):
            continue
        seen.add(patch["patch_type"])
        assert torch.max(result["reproducing_residual_const"]).item() < 1.0e-10
        assert torch.max(result["reproducing_residual_linear"]).item() < 1.0e-10
        assert torch.max(result["reproducing_residual_quadratic"]).item() < 1.0e-10
    assert seen == set(PATCH_TYPES)


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
        eps_reg=1.0e-16,
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
        eps_reg=1.0e-16,
    )
    assert torch.allclose(result["phi_corr"].sum(dim=-1), torch.ones(1, dtype=torch.float64), atol=1.0e-10)
