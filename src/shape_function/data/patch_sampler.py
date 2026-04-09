from __future__ import annotations

from typing import Any

import numpy as np

from shape_function.data.patch_validator import validate_patch

PATCH_TYPES = (
    "uniform",
    "mildly_perturbed",
    "highly_random",
    "clustered",
    "boundary_truncated",
)


def _nearest_neighbors(points: np.ndarray, x_q: np.ndarray, k_neighbors: int) -> np.ndarray:
    distances = np.linalg.norm(points - x_q[None, :], axis=1)
    order = np.argsort(distances)
    return points[order[:k_neighbors]]


def _uniform_grid(rng: np.random.Generator, n_side: int = 8, jitter: float = 0.0) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, n_side, dtype=np.float64)
    xg, yg = np.meshgrid(grid, grid, indexing="xy")
    points = np.column_stack([xg.ravel(), yg.ravel()])
    if jitter > 0.0:
        step = 1.0 / max(n_side - 1, 1)
        noise = rng.uniform(-jitter * step, jitter * step, size=points.shape)
        boundary = (
            np.isclose(points[:, 0], 0.0)
            | np.isclose(points[:, 0], 1.0)
            | np.isclose(points[:, 1], 0.0)
            | np.isclose(points[:, 1], 1.0)
        )
        points[~boundary] = np.clip(points[~boundary] + noise[~boundary], 0.0, 1.0)
    return points


def _sample_query(rng: np.random.Generator, patch_type: str) -> np.ndarray:
    if patch_type == "boundary_truncated":
        side = int(rng.integers(0, 4))
        value = float(rng.uniform(0.03, 0.10))
        other = float(rng.uniform(0.15, 0.85))
        options = (
            np.asarray([value, other], dtype=np.float64),
            np.asarray([1.0 - value, other], dtype=np.float64),
            np.asarray([other, value], dtype=np.float64),
            np.asarray([other, 1.0 - value], dtype=np.float64),
        )
        return options[side]
    return rng.uniform(0.18, 0.82, size=2).astype(np.float64)


def _candidate_points(rng: np.random.Generator, patch_type: str) -> np.ndarray:
    if patch_type == "uniform":
        return _uniform_grid(rng, n_side=8, jitter=0.0)
    if patch_type == "mildly_perturbed":
        return _uniform_grid(rng, n_side=8, jitter=0.18)
    if patch_type == "highly_random":
        return rng.uniform(0.0, 1.0, size=(96, 2)).astype(np.float64)
    if patch_type == "clustered":
        centers = rng.uniform(0.15, 0.85, size=(3, 2))
        cluster_ids = rng.integers(0, centers.shape[0], size=80)
        noise = rng.normal(scale=0.08, size=(80, 2))
        clustered = np.clip(centers[cluster_ids] + noise, 0.0, 1.0)
        uniform = rng.uniform(0.0, 1.0, size=(24, 2))
        return np.concatenate([clustered, uniform], axis=0).astype(np.float64)
    if patch_type == "boundary_truncated":
        return rng.uniform(0.0, 1.0, size=(96, 2)).astype(np.float64)
    raise ValueError(f"Unsupported patch type: {patch_type}")


def sample_patch(
    rng: np.random.Generator,
    patch_type: str,
    k_neighbors: int = 16,
    beta_range: tuple[float, float] = (0.5, 8.0),
) -> dict[str, Any]:
    if patch_type not in PATCH_TYPES:
        raise ValueError(f"Unsupported patch type: {patch_type}")
    x_q = _sample_query(rng, patch_type)
    candidates = _candidate_points(rng, patch_type)
    X = _nearest_neighbors(candidates, x_q, k_neighbors)
    beta = float(np.exp(rng.uniform(np.log(beta_range[0]), np.log(beta_range[1]))))
    distances = np.linalg.norm(X - x_q[None, :], axis=1)
    patch = {
        "x_q": x_q,
        "X": X.astype(np.float64),
        "beta": beta,
        "r_max": float(np.max(distances)),
        "patch_type": patch_type,
        "meta": {"distances": distances},
    }
    patch["validation"] = validate_patch(x_q, X)
    return patch


def sample_patches(
    num_patches: int,
    seed: int = 42,
    patch_types: tuple[str, ...] = PATCH_TYPES,
    k_neighbors: int = 16,
    beta_range: tuple[float, float] = (0.5, 8.0),
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    patches: list[dict[str, Any]] = []
    while len(patches) < num_patches:
        patch_type = patch_types[len(patches) % len(patch_types)]
        patch = sample_patch(rng, patch_type, k_neighbors=k_neighbors, beta_range=beta_range)
        if patch["validation"]["is_valid"]:
            patches.append(patch)
    return patches
