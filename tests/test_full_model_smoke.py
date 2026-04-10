from __future__ import annotations

import torch

from shape_function.models.full_model import ShapeFunctionModel
from shape_function.train.losses import compute_losses


def test_full_model_smoke_backward() -> None:
    torch.manual_seed(0)
    model = ShapeFunctionModel(backbone_name="kernel_operator", feature_dim=4)
    X = torch.rand(2, 16, 2, dtype=torch.float64)
    x_q = torch.rand(2, 2, dtype=torch.float64)
    beta = torch.rand(2, 1, dtype=torch.float64) + 0.5
    phi_ref = torch.softmax(torch.randn(2, 16, dtype=torch.float64), dim=-1)
    outputs = model(X, x_q, beta)
    losses = compute_losses(outputs, phi_ref)
    losses["total"].backward()
    grads = [param.grad for param in model.parameters() if param.requires_grad]
    assert any(grad is not None for grad in grads)
    assert all(torch.isfinite(grad).all() for grad in grads if grad is not None)


def test_compute_losses_accepts_mask() -> None:
    torch.manual_seed(1)
    model = ShapeFunctionModel(backbone_name="mlp_baseline", feature_dim=4)
    X = torch.rand(2, 16, 2, dtype=torch.float64)
    x_q = torch.rand(2, 2, dtype=torch.float64)
    beta = torch.rand(2, 1, dtype=torch.float64) + 0.5
    phi_ref = torch.softmax(torch.randn(2, 16, dtype=torch.float64), dim=-1)
    mask = torch.ones(2, 16, dtype=torch.float64)
    mask[:, -4:] = 0.0
    outputs = model(X, x_q, beta)
    losses = compute_losses(outputs, phi_ref, mask=mask, valid_count=mask.sum())
    assert torch.isfinite(losses["total"])
