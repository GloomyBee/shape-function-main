from __future__ import annotations

import numpy as np
import pytest

from shape_function.data.feature_builder import build_patch_features
from shape_function.data.patch_sampler import PATCH_TYPES, sample_patch


@pytest.mark.parametrize("patch_type", PATCH_TYPES)
def test_sample_patch_shapes(patch_type: str) -> None:
    patch = sample_patch(np.random.default_rng(42), patch_type, k_neighbors=16)
    assert patch["X"].shape == (16, 2)
    assert patch["x_q"].shape == (2,)
    assert patch["r_max"] > 0.0
    assert patch["validation"]["is_valid"]


def test_feature_builder_modes() -> None:
    patch = sample_patch(np.random.default_rng(7), "uniform", k_neighbors=16)
    minimal = build_patch_features(patch, feature_mode="minimal")
    enhanced = build_patch_features(patch, feature_mode="enhanced")
    assert minimal["node_features"].shape == (16, 4)
    assert enhanced["node_features"].shape == (16, 7)


def test_anisotropic_patch_has_small_eigen_ratio() -> None:
    patch = sample_patch(np.random.default_rng(11), "anisotropic", k_neighbors=16)
    assert patch["validation"]["eig_ratio"] < 0.5


def test_sparse_dense_transition_patch_has_broad_neighbor_distance_spread() -> None:
    patch = sample_patch(np.random.default_rng(13), "sparse_dense_transition", k_neighbors=16)
    diffs = patch["X"][:, None, :] - patch["X"][None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    dists += np.eye(dists.shape[0]) * 1.0e9
    nearest = np.min(dists, axis=1)
    ratio = float(np.max(nearest) / np.min(nearest))
    assert ratio > 2.0
