from __future__ import annotations

import torch

from shape_function.models.heads.structure_head import StructurePreservingHead


def _sample_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    X = torch.rand(4, 16, 2, dtype=torch.float64)
    x_q = torch.rand(4, 2, dtype=torch.float64)
    logits = torch.randn(4, 16, dtype=torch.float64, requires_grad=True)
    r_max = torch.linalg.norm(X - x_q[:, None, :], dim=-1).max(dim=-1, keepdim=True).values
    return logits, X, x_q, r_max


def test_structure_head_preserves_constraints() -> None:
    logits, X, x_q, r_max = _sample_inputs()
    head = StructurePreservingHead()
    _, phi_corr, aux = head(logits, X, x_q, r_max)
    assert phi_corr.shape == (4, 16)
    assert torch.max(torch.abs(aux["sum_phi"] - 1.0)).item() < 1.0e-6
    assert torch.max(aux["linear_residual"]).item() < 1.0e-6


def test_structure_head_translation_invariance() -> None:
    logits, X, x_q, r_max = _sample_inputs()
    head = StructurePreservingHead()
    _, phi_a, _ = head(logits, X, x_q, r_max)
    shift = torch.tensor([3.0, -2.0], dtype=torch.float64)
    _, phi_b, _ = head(logits, X + shift, x_q + shift, r_max)
    assert torch.max(torch.abs(phi_a - phi_b)).item() < 1.0e-6


def test_structure_head_backward() -> None:
    logits, X, x_q, r_max = _sample_inputs()
    head = StructurePreservingHead()
    _, phi_corr, _ = head(logits, X, x_q, r_max)
    loss = phi_corr.square().mean()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_structure_head_default_support_radius_keeps_farthest_neighbor_active() -> None:
    head = StructurePreservingHead()
    X = torch.tensor(
        [[[0.0, 0.0], [0.8, 0.0], [1.0, 0.0]]],
        dtype=torch.float64,
    )
    x_q = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    logits = torch.zeros((1, 3), dtype=torch.float64)
    r_max = torch.linalg.norm(X - x_q[:, None, :], dim=-1).max(dim=-1, keepdim=True).values
    phi_base, _, _ = head(logits, X, x_q, r_max)
    farthest_idx = int(torch.argmax(torch.linalg.norm(X - x_q[:, None, :], dim=-1), dim=-1).item())
    assert phi_base[0, farthest_idx].item() > 0.0
