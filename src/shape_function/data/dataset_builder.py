from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shape_function.data.feature_builder import build_patch_features
from shape_function.data.maxent_solver import solve_maxent_patch
from shape_function.data.patch_sampler import PATCH_TYPES, sample_patches


def build_dataset(
    num_patches: int,
    seed: int = 42,
    feature_mode: str = "minimal",
    k_neighbors: int = 16,
    prior_type: str = "gaussian",
) -> list[dict[str, Any]]:
    patches = sample_patches(num_patches, seed=seed, patch_types=PATCH_TYPES, k_neighbors=k_neighbors)
    dataset: list[dict[str, Any]] = []
    for patch in patches:
        teacher = solve_maxent_patch(patch["x_q"], patch["X"], patch["beta"], prior_type=prior_type)
        if not teacher["success"]:
            continue
        item = dict(patch)
        item["phi_ref"] = teacher["phi_ref"]
        item["teacher"] = teacher
        features = build_patch_features(item, feature_mode=feature_mode, k_max=k_neighbors)
        item["rho_q"] = features["rho_q"]
        item["node_features"] = features["node_features"]
        item["rel_coords"] = features["rel_coords"]
        dataset.append(item)
    return dataset


def save_dataset(dataset: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, patches=np.asarray(dataset, dtype=object))
