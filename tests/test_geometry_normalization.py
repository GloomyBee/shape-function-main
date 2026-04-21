from __future__ import annotations

import numpy as np
import torch

from shape_function.data.feature_builder import build_patch_features
from shape_function.utils.geometry import rel_coords_and_radius_torch


def test_rel_coords_and_radius_torch_returns_rmax_normalized_coordinates() -> None:
    X = torch.tensor([[[2.0, 1.0], [4.0, 1.0], [2.0, 3.0]]], dtype=torch.float64)
    x_q = torch.tensor([[2.0, 1.0]], dtype=torch.float64)

    rel, rel_hat, r_max = rel_coords_and_radius_torch(X, x_q)

    expected_rel = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]], dtype=torch.float64)
    expected_r_max = torch.tensor([[2.0]], dtype=torch.float64)
    expected_rel_hat = expected_rel / expected_r_max.unsqueeze(-1)

    assert torch.allclose(rel, expected_rel)
    assert torch.allclose(r_max, expected_r_max)
    assert torch.allclose(rel_hat, expected_rel_hat)
    assert torch.allclose(torch.linalg.norm(rel_hat, dim=-1).amax(dim=-1), torch.ones(1, dtype=torch.float64))


def test_build_patch_features_minimal_uses_rmax_normalized_rel_coords() -> None:
    X = np.asarray([[13.0, -2.0], [10.0, 4.0], [7.0, -2.0]], dtype=np.float64)
    x_q = np.asarray([10.0, -2.0], dtype=np.float64)
    r_max = 6.0
    beta = 3.0
    patch = {
        "X": X,
        "x_q": x_q,
        "beta": beta,
        "r_max": r_max,
    }

    features = build_patch_features(patch, feature_mode="minimal")

    expected_rel_hat = (X - x_q[None, :]) / r_max
    expected_dist_hat = np.linalg.norm(expected_rel_hat, axis=1)
    node_features = features["node_features"]

    np.testing.assert_allclose(features["rel_coords"], expected_rel_hat)
    np.testing.assert_allclose(node_features[:, :2], expected_rel_hat)
    np.testing.assert_allclose(node_features[:, 2], expected_dist_hat)
    np.testing.assert_allclose(node_features[:, 3], np.full(3, beta))
