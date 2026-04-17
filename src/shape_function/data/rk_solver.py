from __future__ import annotations

from typing import Any

import numpy as np


def _basis_exponents_2d(order: int) -> list[tuple[int, int]]:
    exponents: list[tuple[int, int]] = []
    for total_degree in range(order + 1):
        for power_y in range(total_degree + 1):
            power_x = total_degree - power_y
            exponents.append((power_x, power_y))
    return exponents


def polynomial_basis_2d(rel_coords: np.ndarray, order: int = 1) -> np.ndarray:
    rel_coords = np.asarray(rel_coords, dtype=np.float64)
    if rel_coords.ndim != 2 or rel_coords.shape[1] != 2:
        raise ValueError("polynomial_basis_2d expects an array of shape (n, 2)")
    exponents = _basis_exponents_2d(order)
    basis = np.empty((rel_coords.shape[0], len(exponents)), dtype=np.float64)
    x = rel_coords[:, 0]
    y = rel_coords[:, 1]
    for column, (power_x, power_y) in enumerate(exponents):
        basis[:, column] = np.power(x, power_x) * np.power(y, power_y)
    return basis


def basis_dimension_2d(order: int) -> int:
    return len(_basis_exponents_2d(order))


def spline_window_1d(normalized_coordinate: np.ndarray, ispline: int = 3) -> np.ndarray:
    coord = np.asarray(normalized_coordinate, dtype=np.float64)
    out = np.zeros_like(coord, dtype=np.float64)
    if ispline == 3:
        mask = (-1.5 <= coord) & (coord < -0.5)
        out[mask] = np.square(2.0 * coord[mask] + 3.0) / 8.0
        mask = (-0.5 <= coord) & (coord < 0.5)
        out[mask] = -np.square(coord[mask]) + 0.75
        mask = (0.5 <= coord) & (coord <= 1.5)
        out[mask] = np.square(2.0 * coord[mask] - 3.0) / 8.0
        return out
    if ispline == 4:
        mask = (-2.0 <= coord) & (coord < -1.0)
        out[mask] = np.power(coord[mask] + 2.0, 3) / 6.0
        mask = (-1.0 <= coord) & (coord < 0.0)
        out[mask] = 2.0 / 3.0 - np.square(coord[mask]) * (1.0 + coord[mask] / 2.0)
        mask = (0.0 <= coord) & (coord < 1.0)
        out[mask] = 2.0 / 3.0 - np.square(coord[mask]) * (1.0 - coord[mask] / 2.0)
        mask = (1.0 <= coord) & (coord <= 2.0)
        out[mask] = -np.power(coord[mask] - 2.0, 3) / 6.0
        return out
    raise ValueError(f"Unsupported ispline: {ispline}")


def tensor_product_window_2d(
    query_point: np.ndarray,
    nodes: np.ndarray,
    dilation: np.ndarray,
    ispline: int = 3,
) -> np.ndarray:
    query_point = np.asarray(query_point, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    dilation = np.asarray(dilation, dtype=np.float64)
    if query_point.shape != (2,):
        raise ValueError("query_point must have shape (2,)")
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must have shape (n, 2)")
    if dilation.shape == (2,):
        dilation = np.repeat(dilation[None, :], nodes.shape[0], axis=0)
    if dilation.shape != nodes.shape:
        raise ValueError("dilation must have shape (2,) or match nodes shape (n, 2)")
    normalized = (query_point[None, :] - nodes) / np.clip(dilation, 1.0e-12, None)
    window_x = spline_window_1d(normalized[:, 0], ispline=ispline) / np.clip(dilation[:, 0], 1.0e-12, None)
    window_y = spline_window_1d(normalized[:, 1], ispline=ispline) / np.clip(dilation[:, 1], 1.0e-12, None)
    return window_x * window_y


def support_extent_for_ispline(ispline: int) -> float:
    if ispline == 3:
        return 1.5
    if ispline == 4:
        return 2.0
    raise ValueError(f"Unsupported ispline: {ispline}")


def find_support_indices(
    query_point: np.ndarray,
    nodes: np.ndarray,
    dilation: np.ndarray,
    ispline: int = 3,
) -> np.ndarray:
    query_point = np.asarray(query_point, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    dilation = np.asarray(dilation, dtype=np.float64)
    if dilation.shape == (2,):
        dilation = np.repeat(dilation[None, :], nodes.shape[0], axis=0)
    extent = support_extent_for_ispline(ispline)
    normalized = np.abs((query_point[None, :] - nodes) / np.clip(dilation, 1.0e-12, None))
    return np.flatnonzero(np.all(normalized <= extent, axis=1))


def compute_correction_vector_2d(
    query_point: np.ndarray,
    nodes: np.ndarray,
    dilation: np.ndarray,
    order: int = 1,
    ispline: int = 3,
    regularization: float = 1.0e-12,
) -> dict[str, Any]:
    query_point = np.asarray(query_point, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    rel_coords = query_point[None, :] - nodes
    basis = polynomial_basis_2d(rel_coords, order=order)
    weights = tensor_product_window_2d(query_point, nodes, dilation=dilation, ispline=ispline)
    moment = basis.T @ (weights[:, None] * basis)
    if regularization > 0.0:
        moment = moment + regularization * np.eye(moment.shape[0], dtype=np.float64)
    rhs = np.zeros((moment.shape[0],), dtype=np.float64)
    rhs[0] = 1.0
    correction = np.linalg.solve(moment, rhs)
    cond_number = float(np.linalg.cond(moment))
    return {
        "correction": correction,
        "moment": moment,
        "basis": basis,
        "weights": weights,
        "cond_moment": cond_number,
    }


def solve_rk_patch_2d(
    query_point: np.ndarray,
    nodes: np.ndarray,
    dilation: np.ndarray,
    order: int = 1,
    ispline: int = 3,
    regularization: float = 1.0e-12,
) -> dict[str, Any]:
    correction_result = compute_correction_vector_2d(
        query_point=query_point,
        nodes=nodes,
        dilation=dilation,
        order=order,
        ispline=ispline,
        regularization=regularization,
    )
    phi = correction_result["weights"] * (correction_result["basis"] @ correction_result["correction"])
    query_point = np.asarray(query_point, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    return {
        "phi": phi,
        "correction": correction_result["correction"],
        "moment": correction_result["moment"],
        "weights": correction_result["weights"],
        "basis": correction_result["basis"],
        "cond_moment": correction_result["cond_moment"],
        "pou_residual": float(abs(np.sum(phi) - 1.0)),
        "linear_residual": float(np.linalg.norm(phi @ nodes - query_point)),
    }


def square_smoothing_boundary_points_2d(
    center: np.ndarray,
    area: float,
    factor: float = 0.8,
) -> dict[str, np.ndarray]:
    center = np.asarray(center, dtype=np.float64)
    if center.shape != (2,):
        raise ValueError("center must have shape (2,)")
    scaled_area = max(float(area) * factor, 1.0e-12)
    side = float(np.sqrt(scaled_area))
    half = 0.5 * side
    edge_midpoints = np.asarray(
        [
            [center[0] - half, center[1]],
            [center[0] + half, center[1]],
            [center[0], center[1] - half],
            [center[0], center[1] + half],
        ],
        dtype=np.float64,
    )
    normals = np.asarray(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    edge_lengths = np.full((4,), side, dtype=np.float64)
    return {
        "edge_midpoints": edge_midpoints,
        "normals": normals,
        "edge_lengths": edge_lengths,
        "cell_area": np.asarray(scaled_area, dtype=np.float64),
        "cell_side": np.asarray(side, dtype=np.float64),
    }


def smoothed_shape_gradients_2d(
    center: np.ndarray,
    area: float,
    nodes: np.ndarray,
    dilation: np.ndarray,
    order: int = 1,
    ispline: int = 3,
    factor: float = 0.8,
    regularization: float = 1.0e-12,
) -> dict[str, Any]:
    boundary = square_smoothing_boundary_points_2d(center=center, area=area, factor=factor)
    edge_midpoints = boundary["edge_midpoints"]
    normals = boundary["normals"]
    edge_lengths = boundary["edge_lengths"]
    cell_area = float(boundary["cell_area"])
    nodes = np.asarray(nodes, dtype=np.float64)
    phi_boundary = np.empty((edge_midpoints.shape[0], nodes.shape[0]), dtype=np.float64)
    for edge_index, point in enumerate(edge_midpoints):
        phi_boundary[edge_index] = solve_rk_patch_2d(
            query_point=point,
            nodes=nodes,
            dilation=dilation,
            order=order,
            ispline=ispline,
            regularization=regularization,
        )["phi"]
    gradients = np.zeros((nodes.shape[0], 2), dtype=np.float64)
    for edge_index in range(edge_midpoints.shape[0]):
        gradients += (edge_lengths[edge_index] / cell_area) * phi_boundary[edge_index][:, None] * normals[edge_index][None, :]
    return {
        "grad_phi": gradients,
        "phi_boundary": phi_boundary,
        "boundary": boundary,
        "gradient_pou_residual": np.linalg.norm(np.sum(gradients, axis=0)),
        "gradient_linear_residual": np.linalg.norm(nodes.T @ gradients - np.eye(2, dtype=np.float64)),
    }
