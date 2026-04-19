from __future__ import annotations

import pytest

from shape_function.models.full_model import build_shape_function_model, feature_dim_for_mode


def test_feature_dim_for_mode_matches_supported_modes() -> None:
    assert feature_dim_for_mode("minimal") == 4
    assert feature_dim_for_mode("enhanced") == 7
    with pytest.raises(ValueError, match="Unsupported feature_mode"):
        feature_dim_for_mode("unknown")


def test_build_shape_function_model_propagates_kernel_operator_params() -> None:
    model = build_shape_function_model(
        backbone_name="kernel_operator",
        feature_mode="enhanced",
        hidden_dim=24,
        num_layers=3,
        k_neighbors=16,
        head_kwargs={"basis_order": 2, "kappa_max": 1.0e8},
    )
    assert model.backbone.hidden_dim == 24
    assert model.backbone.num_layers == 3
    assert model.backbone.lift[0].in_features == 7
    assert model.head.basis_order == 2


def test_build_shape_function_model_propagates_mlp_params() -> None:
    model = build_shape_function_model(
        backbone_name="mlp_baseline",
        feature_mode="minimal",
        hidden_dim=20,
        k_neighbors=12,
    )
    first_linear = model.backbone.network[0]
    last_linear = model.backbone.network[-1]
    assert first_linear.in_features == 48
    assert first_linear.out_features == 20
    assert last_linear.out_features == 12
