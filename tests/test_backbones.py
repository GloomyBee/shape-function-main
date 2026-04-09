from __future__ import annotations

import torch

from shape_function.models.backbones.kernel_operator import KernelOperatorBackbone
from shape_function.models.backbones.mlp_baseline import MLPBaselineBackbone


def test_kernel_operator_output_shape() -> None:
    features = torch.randn(3, 16, 4, dtype=torch.float64)
    rel = torch.randn(3, 16, 2, dtype=torch.float64)
    model = KernelOperatorBackbone(input_dim=4)
    logits = model(features, rel)
    assert logits.shape == (3, 16)


def test_mlp_baseline_output_shape() -> None:
    features = torch.randn(3, 16, 4, dtype=torch.float64)
    rel = torch.randn(3, 16, 2, dtype=torch.float64)
    model = MLPBaselineBackbone(input_dim=4)
    logits = model(features, rel)
    assert logits.shape == (3, 16)
