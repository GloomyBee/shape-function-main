from __future__ import annotations

import numpy as np

from shape_function.data.rk_solver import (
    find_support_indices,
    smoothed_shape_gradients_2d,
    solve_rk_patch_2d,
    support_extent_for_ispline,
)


def _sample_patch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(
        [
            [0.10, 0.15],
            [0.35, 0.12],
            [0.65, 0.10],
            [0.88, 0.18],
            [0.18, 0.42],
            [0.42, 0.38],
            [0.70, 0.36],
            [0.88, 0.45],
            [0.15, 0.70],
            [0.40, 0.68],
            [0.66, 0.64],
            [0.86, 0.74],
        ],
        dtype=np.float64,
    )
    query = np.asarray([0.52, 0.47], dtype=np.float64)
    dilation = np.full((nodes.shape[0], 2), 0.42, dtype=np.float64)
    return query, nodes, dilation


def test_rk_patch_preserves_partition_of_unity_and_linear_reproduction() -> None:
    query, nodes, dilation = _sample_patch()
    result = solve_rk_patch_2d(query, nodes, dilation, order=1, ispline=4)
    assert abs(np.sum(result["phi"]) - 1.0) < 1.0e-8
    assert np.linalg.norm(result["phi"] @ nodes - query) < 1.0e-8
    assert result["cond_moment"] < 1.0e8


def test_smoothed_shape_gradients_preserve_constant_and_linear_fields() -> None:
    query, nodes, dilation = _sample_patch()
    result = smoothed_shape_gradients_2d(
        center=query,
        area=0.08,
        nodes=nodes,
        dilation=dilation,
        order=1,
        ispline=4,
    )
    grad_phi = result["grad_phi"]
    assert np.linalg.norm(np.sum(grad_phi, axis=0)) < 1.0e-8
    assert np.linalg.norm(nodes.T @ grad_phi - np.eye(2)) < 5.0e-2


def test_find_support_indices_matches_window_extent() -> None:
    query, nodes, dilation = _sample_patch()
    support = find_support_indices(query, nodes, dilation, ispline=3)
    extent = support_extent_for_ispline(3)
    normalized = np.abs((query[None, :] - nodes[support]) / dilation[support])
    assert support.size > 0
    assert np.all(normalized <= extent + 1.0e-12)
