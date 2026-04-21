from __future__ import annotations

import torch

from shape_function.models.backbones.deepsets import DeepSetsBackbone
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


def test_deepsets_output_shape() -> None:
    features = torch.randn(3, 16, 4, dtype=torch.float64)
    rel = torch.randn(3, 16, 2, dtype=torch.float64)
    model = DeepSetsBackbone(input_dim=4, hidden_dim=12, num_encoder_layers=2, num_decoder_layers=2)
    logits = model(features, rel)
    assert logits.shape == (3, 16)


def test_deepsets_is_permutation_equivariant_with_mask() -> None:
    torch.manual_seed(4)
    features = torch.randn(2, 16, 4, dtype=torch.float64)
    rel = torch.randn(2, 16, 2, dtype=torch.float64)
    mask = torch.ones(2, 16, dtype=torch.float64)
    mask[:, -3:] = 0.0
    model = DeepSetsBackbone(input_dim=4, hidden_dim=12, num_encoder_layers=2, num_decoder_layers=2)
    permutation = torch.randperm(16)
    logits = model(features, rel, mask=mask)
    permuted_logits = model(features[:, permutation], rel[:, permutation], mask=mask[:, permutation])
    assert torch.allclose(permuted_logits, logits[:, permutation], atol=1.0e-10)
