from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
import yaml
from matplotlib.patches import Circle
from scipy.spatial import Delaunay, QhullError

from shape_function.data.feature_builder import build_patch_features
from shape_function.data.dataset_builder import build_dataset
from shape_function.data.maxent_solver import solve_maxent_patch
from shape_function.models.full_model import build_shape_function_model
from shape_function.utils.artifacts import save_json

REL_ERROR_FLOOR = 1.0e-3
REL_ERROR_CLIP = 2.0


def select_patch_indices(
    scores: list[float] | np.ndarray,
    *,
    num_patches: int,
    strategy: str = "spread",
) -> list[int]:
    score_array = np.asarray(scores, dtype=np.float64)
    if score_array.ndim != 1:
        raise ValueError("scores must be a 1D array")
    if num_patches <= 0:
        raise ValueError("num_patches must be positive")
    total = int(score_array.shape[0])
    if total == 0:
        return []
    order = np.argsort(score_array)
    if total <= num_patches:
        return order.tolist()
    if strategy == "best":
        return order[:num_patches].tolist()
    if strategy == "worst":
        return order[-num_patches:][::-1].tolist()
    if strategy == "spread":
        positions = np.linspace(0, total - 1, num_patches)
        rounded = np.rint(positions).astype(int)
        unique_positions: list[int] = []
        for position in rounded.tolist():
            if position not in unique_positions:
                unique_positions.append(position)
        cursor = 0
        while len(unique_positions) < num_patches:
            if cursor not in unique_positions:
                unique_positions.append(cursor)
            cursor += 1
        return [int(order[position]) for position in sorted(unique_positions[:num_patches])]
    raise ValueError(f"Unsupported selection strategy: {strategy}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def resolve_basis_node_index(
    patch: dict[str, Any],
    node_index: int | None = None,
    *,
    fallback_values: np.ndarray | None = None,
) -> int:
    if node_index is not None:
        resolved = int(node_index)
        basis_size = int(np.asarray(patch["X"], dtype=np.float64).shape[0])
        if resolved < 0 or resolved >= basis_size:
            raise ValueError(f"node_index {resolved} is out of range for patch of size {basis_size}")
        return resolved

    if "phi_ref" in patch:
        phi_ref = np.asarray(patch["phi_ref"], dtype=np.float64)
        if phi_ref.ndim == 1 and phi_ref.size > 0:
            return int(np.argmax(phi_ref))

    if fallback_values is not None:
        values = np.asarray(fallback_values, dtype=np.float64)
        if values.ndim == 1 and values.size > 0:
            return int(np.argmax(values))

    X = np.asarray(patch["X"], dtype=np.float64)
    x_q = np.asarray(patch["x_q"], dtype=np.float64)
    distances = np.linalg.norm(X - x_q[None, :], axis=1)
    return int(np.argmin(distances))


def _resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine")
    return requested_device


def _load_run_snapshot(run_dir: Path) -> dict[str, Any]:
    snapshot_path = run_dir / "config_snapshot.yaml"
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Missing config snapshot: {snapshot_path}")
    return _load_yaml(snapshot_path)


def _build_val_dataset(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    data_cfg = snapshot["data"]
    resolved = snapshot["resolved"]
    return build_dataset(
        num_patches=int(data_cfg["num_val"]),
        seed=int(resolved["val_seed"]),
        feature_mode=str(data_cfg["feature_mode"]),
        k_neighbors=int(data_cfg["k_neighbors"]),
        prior_type=str(resolved.get("prior_type", "gaussian")),
        patch_types=tuple(str(item) for item in data_cfg["patch_types"]),
        beta_range=tuple(float(value) for value in data_cfg["beta_range"]),
    )


def _build_model(snapshot: dict[str, Any], checkpoint_path: Path, device: str) -> torch.nn.Module:
    model_cfg = snapshot["train"]["model"]
    data_cfg = snapshot["data"]
    model = build_shape_function_model(
        backbone_name=str(model_cfg["backbone"]),
        feature_mode=str(data_cfg["feature_mode"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]) if "num_layers" in model_cfg else None,
        k_neighbors=int(data_cfg["k_neighbors"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _predict_patch(
    model: torch.nn.Module,
    patch: dict[str, Any],
    device: str,
    *,
    prior_type: str = "gaussian",
) -> dict[str, Any]:
    X = torch.as_tensor(np.asarray(patch["X"]), dtype=torch.float64, device=device).unsqueeze(0)
    x_q = torch.as_tensor(np.asarray(patch["x_q"]), dtype=torch.float64, device=device).unsqueeze(0)
    beta_value = float(patch["beta"])
    beta = torch.as_tensor([[beta_value]], dtype=torch.float64, device=device)
    rho_q_value = np.asarray(patch.get("rho_q", np.zeros((0,), dtype=np.float64)), dtype=np.float64)
    rho_q = None
    if rho_q_value.size > 0:
        rho_q = torch.as_tensor(rho_q_value, dtype=torch.float64, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(X, x_q, beta, rho_q=rho_q)
    phi_pred = outputs["phi_corr"][0].detach().cpu().numpy()
    phi_base = outputs["phi_base"][0].detach().cpu().numpy()
    phi_ref = patch.get("phi_ref")
    if phi_ref is None:
        teacher = solve_maxent_patch(
            np.asarray(patch["x_q"], dtype=np.float64),
            np.asarray(patch["X"], dtype=np.float64),
            beta_value,
            prior_type=prior_type,
        )
        if not teacher["success"]:
            raise RuntimeError("teacher solve failed while preparing visualization reference")
        phi_ref = teacher["phi_ref"]
    phi_ref = np.asarray(phi_ref, dtype=np.float64)
    abs_err = np.abs(phi_pred - phi_ref)
    denom = max(float(np.linalg.norm(phi_ref)), 1.0e-12)
    aux = outputs["aux"]
    support_radius_scale = float(getattr(model.head, "support_radius_scale", 1.0))
    support_radius = support_radius_scale * float(np.asarray(patch["r_max"], dtype=np.float64))
    return {
        "phi_pred": phi_pred,
        "phi_base": phi_base,
        "phi_ref": phi_ref,
        "relative_l2": float(np.linalg.norm(phi_pred - phi_ref) / denom),
        "global_linf": float(abs_err.max()),
        "neg_fraction": float(np.mean(phi_pred < 0.0)),
        "max_negative_magnitude": float(max(0.0, -float(phi_pred.min()))),
        "cond_M": float(aux["cond_M"][0].detach().cpu().item()),
        "support_radius": support_radius,
    }


def _build_query_grid_from_nodes(
    X: np.ndarray,
    *,
    grid_size: int = 60,
    padding_ratio: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    span = np.maximum(maxs - mins, 1.0e-6)
    padding = padding_ratio * span
    xs = np.linspace(mins[0] - padding[0], maxs[0] + padding[0], grid_size)
    ys = np.linspace(mins[1] - padding[1], maxs[1] + padding[1], grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    try:
        triangulation = Delaunay(X)
        valid_mask = triangulation.find_simplex(points) >= 0
    except QhullError:
        valid_mask = np.ones((points.shape[0],), dtype=bool)
    return grid_x, grid_y, valid_mask.reshape(grid_x.shape)


def _build_query_patch(base_patch: dict[str, Any], x_q: np.ndarray) -> dict[str, Any]:
    X = np.asarray(base_patch["X"], dtype=np.float64)
    distances = np.linalg.norm(X - x_q[None, :], axis=1)
    patch = {
        "X": X,
        "x_q": np.asarray(x_q, dtype=np.float64),
        "beta": float(base_patch["beta"]),
        "r_max": float(np.max(distances)),
        "patch_type": base_patch.get("patch_type", "unknown"),
    }
    return patch


def compute_basis_field(
    model: torch.nn.Module,
    patch: dict[str, Any],
    *,
    device: str,
    feature_mode: str,
    prior_type: str = "gaussian",
    grid_size: int = 60,
    node_index: int | None = None,
    fallback_values: np.ndarray | None = None,
) -> dict[str, Any]:
    X = np.asarray(patch["X"], dtype=np.float64)
    beta = float(patch["beta"])
    resolved_node_index = resolve_basis_node_index(
        patch,
        node_index=node_index,
        fallback_values=fallback_values,
    )
    grid_x, grid_y, valid_mask = _build_query_grid_from_nodes(X, grid_size=grid_size)
    teacher_field = np.full_like(grid_x, np.nan, dtype=np.float64)
    pred_field = np.full_like(grid_x, np.nan, dtype=np.float64)
    flat_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    valid_points = flat_points[valid_mask.ravel()]

    if valid_points.shape[0] == 0:
        raise RuntimeError("no valid query points were found inside the patch hull")

    rho_rows: list[np.ndarray] = []
    for point in valid_points:
        query_patch = _build_query_patch(patch, point)
        features = build_patch_features(query_patch, feature_mode=feature_mode, k_max=X.shape[0])
        rho_rows.append(features["rho_q"])
    rho_width = int(max((row.shape[0] for row in rho_rows), default=0))
    rho_q = None
    if rho_width > 0:
        rho_q = torch.as_tensor(np.stack(rho_rows, axis=0), dtype=torch.float64, device=device)

    X_batch = torch.as_tensor(np.repeat(X[None, :, :], valid_points.shape[0], axis=0), dtype=torch.float64, device=device)
    x_q_batch = torch.as_tensor(valid_points, dtype=torch.float64, device=device)
    beta_batch = torch.full((valid_points.shape[0], 1), beta, dtype=torch.float64, device=device)
    with torch.no_grad():
        outputs = model(X_batch, x_q_batch, beta_batch, rho_q=rho_q)
    pred_values = outputs["phi_corr"][:, resolved_node_index].detach().cpu().numpy()

    teacher_successes = 0
    valid_indices = np.flatnonzero(valid_mask.ravel())
    for offset, point in enumerate(valid_points):
        teacher = solve_maxent_patch(point, X, beta, prior_type=prior_type)
        if teacher["success"]:
            teacher_successes += 1
            teacher_field.flat[valid_indices[offset]] = float(teacher["phi_ref"][resolved_node_index])
            pred_field.flat[valid_indices[offset]] = float(pred_values[offset])

    abs_error = np.abs(pred_field - teacher_field)
    rel_error, rel_error_mask = build_clipped_relative_error(pred_field, teacher_field)
    abs_error[~np.isfinite(teacher_field)] = np.nan
    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "teacher_field": teacher_field,
        "pred_field": pred_field,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "rel_error_mask": rel_error_mask,
        "valid_mask": np.isfinite(teacher_field),
        "node_index": resolved_node_index,
        "node_coord": X[resolved_node_index],
        "teacher_success_ratio": float(teacher_successes / max(valid_points.shape[0], 1)),
    }


def save_patch_visualization(
    output_path: Path,
    patch: dict[str, Any],
    prediction: dict[str, Any],
    *,
    title_prefix: str = "",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(patch["X"], dtype=np.float64)
    x_q = np.asarray(patch["x_q"], dtype=np.float64)
    phi_ref = np.asarray(prediction.get("phi_ref", patch.get("phi_ref")), dtype=np.float64)
    phi_pred = np.asarray(prediction["phi_pred"], dtype=np.float64)
    distances = np.linalg.norm(X - x_q[None, :], axis=1)
    sort_idx = np.argsort(distances)
    value_min = float(min(phi_ref.min(), phi_pred.min(), 0.0))
    value_max = float(max(phi_ref.max(), phi_pred.max(), 1.0e-12))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    cmap = "coolwarm"

    for ax, values, title in (
        (axes[0, 0], phi_ref, "Reference Shape Function"),
        (axes[0, 1], phi_pred, "Predicted Shape Function"),
    ):
        scatter = ax.scatter(
            X[:, 0],
            X[:, 1],
            c=values,
            s=110,
            cmap=cmap,
            vmin=value_min,
            vmax=value_max,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.scatter(x_q[0], x_q[1], marker="*", s=220, c="gold", edgecolors="black", linewidths=0.8)
        for index, point in enumerate(X):
            ax.text(point[0] + 0.01, point[1] + 0.01, str(index), fontsize=8)
        r_max = float(np.asarray(patch["r_max"], dtype=np.float64))
        support_radius = float(prediction["support_radius"])
        ax.add_patch(Circle((x_q[0], x_q[1]), r_max, fill=False, linestyle="--", linewidth=1.2, edgecolor="black"))
        ax.add_patch(
            Circle((x_q[0], x_q[1]), support_radius, fill=False, linestyle=":", linewidth=1.2, edgecolor="tab:green")
        )
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        fig.colorbar(scatter, ax=ax, shrink=0.88)

    positions = np.arange(len(sort_idx))
    axes[1, 0].bar(positions - 0.18, phi_ref[sort_idx], width=0.36, label="reference", alpha=0.85)
    axes[1, 0].bar(positions + 0.18, phi_pred[sort_idx], width=0.36, label="prediction", alpha=0.85)
    axes[1, 0].set_title("Shape Function Values (sorted by distance)")
    axes[1, 0].set_xlabel("Neighbor rank")
    axes[1, 0].set_ylabel("phi")
    axes[1, 0].grid(True, axis="y", alpha=0.25)
    axes[1, 0].legend()

    abs_err = np.abs(phi_pred[sort_idx] - phi_ref[sort_idx])
    axes[1, 1].bar(positions, abs_err, color="tab:red", alpha=0.85)
    axes[1, 1].set_title("Absolute Error per Neighbor")
    axes[1, 1].set_xlabel("Neighbor rank")
    axes[1, 1].set_ylabel("|pred - ref|")
    axes[1, 1].grid(True, axis="y", alpha=0.25)

    metadata_lines = [
        f"patch_type = {patch.get('patch_type', 'unknown')}",
        f"beta = {float(patch.get('beta', 0.0)):.3f}",
        f"relative_l2 = {float(prediction['relative_l2']):.4f}",
        f"global_linf = {float(prediction['global_linf']):.4f}",
        f"neg_fraction = {float(prediction['neg_fraction']):.4f}",
        f"max_neg = {float(prediction['max_negative_magnitude']):.4f}",
        f"cond_M = {float(prediction['cond_M']):.2f}",
    ]
    axes[1, 1].text(
        0.98,
        0.98,
        "\n".join(metadata_lines),
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.7"},
    )

    title_parts = [part for part in (title_prefix, patch.get("patch_type", "patch")) if part]
    fig.suptitle(" | ".join(title_parts), fontsize=13)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_triangulation(X: np.ndarray) -> mtri.Triangulation:
    if X.shape[0] < 3:
        raise ValueError("At least three points are required for triangulated field visualization")
    return mtri.Triangulation(X[:, 0], X[:, 1])


def build_clipped_relative_error(
    phi_pred: np.ndarray,
    phi_ref: np.ndarray,
    *,
    eps: float = 1.0e-8,
    rel_error_floor: float = REL_ERROR_FLOOR,
    rel_error_clip: float = REL_ERROR_CLIP,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(phi_pred, dtype=np.float64)
    ref = np.asarray(phi_ref, dtype=np.float64)
    values = np.abs(pred - ref) / np.maximum(np.abs(ref), eps)
    values = np.clip(values, 0.0, rel_error_clip)
    masked = (~np.isfinite(pred)) | (~np.isfinite(ref)) | (np.abs(ref) < rel_error_floor)
    values = np.asarray(values, dtype=np.float64)
    values[masked] = np.nan
    return values, masked


def _masked_relative_error(
    phi_pred: np.ndarray,
    phi_ref: np.ndarray,
    *,
    rel_error_floor: float = REL_ERROR_FLOOR,
    rel_error_clip: float = REL_ERROR_CLIP,
) -> np.ma.MaskedArray:
    values, masked = build_clipped_relative_error(
        phi_pred,
        phi_ref,
        rel_error_floor=rel_error_floor,
        rel_error_clip=rel_error_clip,
    )
    return np.ma.array(values, mask=masked)


def _decorate_patch_axis(
    ax: Any,
    patch: dict[str, Any],
    support_radius: float,
) -> None:
    X = np.asarray(patch["X"], dtype=np.float64)
    x_q = np.asarray(patch["x_q"], dtype=np.float64)
    r_max = float(np.asarray(patch["r_max"], dtype=np.float64))
    ax.scatter(X[:, 0], X[:, 1], c="black", s=14, alpha=0.75)
    ax.scatter(x_q[0], x_q[1], marker="*", s=180, c="gold", edgecolors="black", linewidths=0.8)
    ax.add_patch(Circle((x_q[0], x_q[1]), r_max, fill=False, linestyle="--", linewidth=1.1, edgecolor="black"))
    ax.add_patch(Circle((x_q[0], x_q[1]), support_radius, fill=False, linestyle=":", linewidth=1.1, edgecolor="tab:green"))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.20)


def _plot_contour_field(
    ax: Any,
    X: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    patch: dict[str, Any],
    support_radius: float,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    triangulation = _build_triangulation(X)
    contour = ax.tricontourf(triangulation, values, levels=18, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.triplot(triangulation, color="white", linewidth=0.6, alpha=0.65)
    _decorate_patch_axis(ax, patch, support_radius)
    ax.set_title(title)
    plt.colorbar(contour, ax=ax, shrink=0.88)


def _plot_surface_field(
    ax: Any,
    X: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    triangulation = _build_triangulation(X)
    surface = ax.plot_trisurf(
        triangulation,
        values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.25,
        antialiased=True,
        alpha=0.96,
    )
    ax.scatter(X[:, 0], X[:, 1], values, c="black", s=12, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("phi")
    plt.colorbar(surface, ax=ax, shrink=0.70, pad=0.08)


def save_patch_field_visualization(
    output_path: Path,
    patch: dict[str, Any],
    prediction: dict[str, Any],
    *,
    view: str = "contour",
    title_prefix: str = "",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(patch["X"], dtype=np.float64)
    phi_ref = np.asarray(prediction.get("phi_ref", patch.get("phi_ref")), dtype=np.float64)
    phi_pred = np.asarray(prediction["phi_pred"], dtype=np.float64)
    abs_err = np.abs(phi_pred - phi_ref)
    rel_err = _masked_relative_error(phi_pred, phi_ref)
    support_radius = float(prediction["support_radius"])
    value_min = float(min(phi_ref.min(), phi_pred.min()))
    value_max = float(max(phi_ref.max(), phi_pred.max(), 1.0e-12))
    metadata_lines = [
        f"patch_type = {patch.get('patch_type', 'unknown')}",
        f"beta = {float(patch.get('beta', 0.0)):.3f}",
        f"relative_l2 = {float(prediction['relative_l2']):.4f}",
        f"global_linf = {float(prediction['global_linf']):.4f}",
        f"neg_fraction = {float(prediction['neg_fraction']):.4f}",
        f"max_neg = {float(prediction['max_negative_magnitude']):.4f}",
        f"cond_M = {float(prediction['cond_M']):.2f}",
        f"rel_err_floor = {REL_ERROR_FLOOR:.1e}",
        f"rel_err_clip = {REL_ERROR_CLIP:.2f}",
    ]
    if view == "contour":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        _plot_contour_field(
            axes[0, 0],
            X,
            phi_ref,
            title="Reference Shape Function",
            patch=patch,
            support_radius=support_radius,
            cmap="coolwarm",
            vmin=value_min,
            vmax=value_max,
        )
        _plot_contour_field(
            axes[0, 1],
            X,
            phi_pred,
            title="Predicted Shape Function",
            patch=patch,
            support_radius=support_radius,
            cmap="coolwarm",
            vmin=value_min,
            vmax=value_max,
        )
        _plot_contour_field(
            axes[1, 0],
            X,
            abs_err,
            title="Absolute Error Cloud",
            patch=patch,
            support_radius=support_radius,
            cmap="Reds",
        )
        _plot_contour_field(
            axes[1, 1],
            X,
            rel_err,
            title="Clipped Relative Error Cloud",
            patch=patch,
            support_radius=support_radius,
            cmap="magma",
            vmin=0.0,
            vmax=REL_ERROR_CLIP,
        )
        axes[1, 1].text(
            0.98,
            0.98,
            "\n".join(metadata_lines),
            transform=axes[1, 1].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.7"},
        )
    elif view == "hybrid3d":
        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        _plot_surface_field(ax1, X, phi_ref, title="Reference Surface", cmap="coolwarm", vmin=value_min, vmax=value_max)
        _plot_surface_field(ax2, X, phi_pred, title="Predicted Surface", cmap="coolwarm", vmin=value_min, vmax=value_max)
        _plot_contour_field(
            ax3,
            X,
            abs_err,
            title="Absolute Error Cloud",
            patch=patch,
            support_radius=support_radius,
            cmap="Reds",
        )
        _plot_contour_field(
            ax4,
            X,
            rel_err,
            title="Clipped Relative Error Cloud",
            patch=patch,
            support_radius=support_radius,
            cmap="magma",
            vmin=0.0,
            vmax=REL_ERROR_CLIP,
        )
        ax4.text(
            0.98,
            0.98,
            "\n".join(metadata_lines),
            transform=ax4.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.7"},
        )
    else:
        raise ValueError(f"Unsupported field visualization view: {view}")
    title_parts = [part for part in (title_prefix, patch.get("patch_type", "patch"), view) if part]
    fig.suptitle(" | ".join(title_parts), fontsize=13)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _mask_field(values: np.ndarray, valid_mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.array(values, mask=~valid_mask)


def _decorate_basis_axis(ax: Any, patch: dict[str, Any], node_coord: np.ndarray) -> None:
    X = np.asarray(patch["X"], dtype=np.float64)
    ax.scatter(X[:, 0], X[:, 1], c="black", s=18, alpha=0.85)
    ax.scatter(node_coord[0], node_coord[1], marker="*", s=220, c="gold", edgecolors="black", linewidths=0.8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.20)


def save_basis_field_visualization(
    output_path: Path,
    patch: dict[str, Any],
    field: dict[str, Any],
    *,
    view: str = "basis_contour",
    title_prefix: str = "",
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid_x = np.asarray(field["grid_x"], dtype=np.float64)
    grid_y = np.asarray(field["grid_y"], dtype=np.float64)
    valid_mask = np.asarray(field["valid_mask"], dtype=bool)
    teacher_field = _mask_field(np.asarray(field["teacher_field"], dtype=np.float64), valid_mask)
    pred_field = _mask_field(np.asarray(field["pred_field"], dtype=np.float64), valid_mask)
    abs_error = _mask_field(np.asarray(field["abs_error"], dtype=np.float64), valid_mask)
    rel_error_mask = np.asarray(field.get("rel_error_mask", np.zeros_like(valid_mask, dtype=bool)), dtype=bool)
    rel_error = np.ma.array(np.asarray(field["rel_error"], dtype=np.float64), mask=(~valid_mask) | rel_error_mask)
    node_index = int(field["node_index"])
    node_coord = np.asarray(field["node_coord"], dtype=np.float64)
    common_min = float(np.nanmin([teacher_field.min(), pred_field.min()]))
    common_max = float(np.nanmax([teacher_field.max(), pred_field.max(), 1.0e-12]))
    metadata_lines = [
        f"node_index = {node_index}",
        f"patch_type = {patch.get('patch_type', 'unknown')}",
        f"beta = {float(patch.get('beta', 0.0)):.3f}",
        f"teacher_ok = {float(field['teacher_success_ratio']):.3f}",
        f"rel_err_floor = {REL_ERROR_FLOOR:.1e}",
        f"rel_err_clip = {REL_ERROR_CLIP:.2f}",
    ]

    if view == "basis_contour":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        for ax, values, title, cmap, vmin, vmax in (
            (axes[0, 0], teacher_field, "Teacher Basis Field", "coolwarm", common_min, common_max),
            (axes[0, 1], pred_field, "Predicted Basis Field", "coolwarm", common_min, common_max),
            (axes[1, 0], abs_error, "Absolute Error Cloud", "Reds", None, None),
            (axes[1, 1], rel_error, "Clipped Relative Error Cloud", "magma", 0.0, REL_ERROR_CLIP),
        ):
            contour = ax.contourf(grid_x, grid_y, values, levels=18, cmap=cmap, vmin=vmin, vmax=vmax)
            _decorate_basis_axis(ax, patch, node_coord)
            ax.set_title(title)
            plt.colorbar(contour, ax=ax, shrink=0.88)
        axes[1, 1].text(
            0.98,
            0.98,
            "\n".join(metadata_lines),
            transform=axes[1, 1].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.7"},
        )
    elif view == "basis_hybrid3d":
        fig = plt.figure(figsize=(12, 10), constrained_layout=True)
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
        for ax, values, title in (
            (ax1, teacher_field.filled(np.nan), "Teacher Basis Surface"),
            (ax2, pred_field.filled(np.nan), "Predicted Basis Surface"),
        ):
            surface = ax.plot_surface(
                grid_x,
                grid_y,
                values,
                cmap="coolwarm",
                vmin=common_min,
                vmax=common_max,
                linewidth=0.0,
                antialiased=True,
                alpha=0.97,
            )
            X = np.asarray(patch["X"], dtype=np.float64)
            ax.scatter(X[:, 0], X[:, 1], np.zeros((X.shape[0],)), c="black", s=12, alpha=0.8)
            ax.scatter(node_coord[0], node_coord[1], 0.0, marker="*", s=180, c="gold", edgecolors="black", linewidths=0.8)
            ax.set_title(title)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("phi")
            ax.view_init(elev=28, azim=-55)
            plt.colorbar(surface, ax=ax, shrink=0.70, pad=0.08)
        for ax, values, title, cmap in (
            (ax3, abs_error, "Absolute Error Cloud", "Reds"),
            (ax4, rel_error, "Clipped Relative Error Cloud", "magma"),
        ):
            contour = ax.contourf(
                grid_x,
                grid_y,
                values,
                levels=18,
                cmap=cmap,
                vmin=0.0 if title.startswith("Clipped") else None,
                vmax=REL_ERROR_CLIP if title.startswith("Clipped") else None,
            )
            _decorate_basis_axis(ax, patch, node_coord)
            ax.set_title(title)
            plt.colorbar(contour, ax=ax, shrink=0.88)
        ax4.text(
            0.98,
            0.98,
            "\n".join(metadata_lines),
            transform=ax4.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.7"},
        )
    else:
        raise ValueError(f"Unsupported basis field visualization view: {view}")

    title_parts = [part for part in (title_prefix, patch.get("patch_type", "patch"), view) if part]
    fig.suptitle(" | ".join(title_parts), fontsize=13)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_curves(curves_path: Path) -> dict[str, np.ndarray]:
    if not curves_path.is_file():
        raise FileNotFoundError(f"Missing training curves: {curves_path}")
    with np.load(curves_path) as payload:
        return {key: np.asarray(payload[key], dtype=np.float64) for key in payload.files}


def save_training_curves_visualization(output_path: Path, curves: dict[str, np.ndarray]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray(curves.get("epoch", np.arange(1, len(curves.get("lr", [])) + 1)), dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    if "train_relative_l2" in curves or "val_relative_l2" in curves:
        panels = [
            ("train_total", "val_total", "Total Loss", "loss"),
            ("train_relative_l2", "val_relative_l2", "Relative L2", "relative_l2"),
            ("train_loss_data", "val_loss_data", "Data Loss", "loss_data"),
            ("lr", None, "Learning Rate", "lr"),
        ]
    else:
        panels = [
            ("train_total", "val_total", "Total Loss", "loss"),
            ("train_base_linear_residual", "val_base_linear_residual", "Base Linear Residual", "residual"),
            ("train_mean_quad_residual", "val_mean_quad_residual", "Quadratic Residual", "residual"),
            ("lr", None, "Learning Rate", "lr"),
        ]
    for ax, (train_key, val_key, title, ylabel) in zip(axes.flat, panels):
        if train_key in curves:
            ax.plot(epochs, curves[train_key], marker="o", linewidth=1.8, label=train_key)
        if val_key and val_key in curves:
            ax.plot(epochs, curves[val_key], marker="s", linewidth=1.8, label=val_key)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        if ax.lines:
            ax.legend()
    fig.suptitle("Training Curves", fontsize=13)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_run_visualizations(
    run_dir: Path,
    *,
    device: str = "auto",
    num_patches: int = 4,
    selection: str = "spread",
    view: str = "bars",
    node_index: int | None = None,
    grid_size: int = 60,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.resolve()
    if view == "training_curves":
        output_dir = resolved_run_dir / "figures" / "training_curves"
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / "training_curves.png"
        curves = _load_curves(resolved_run_dir / "curves.npz")
        save_training_curves_visualization(figure_path, curves)
        summary = {
            "run_dir": str(resolved_run_dir),
            "view": view,
            "num_patches": 0,
            "figures": [str(figure_path)],
            "curve_keys": sorted(curves.keys()),
        }
        save_json(output_dir / "summary.json", summary)
        return summary

    snapshot = _load_run_snapshot(resolved_run_dir)
    resolved_device = _resolve_device(device)
    checkpoint_path = resolved_run_dir / "checkpoint.pt"
    model = _build_model(snapshot, checkpoint_path, resolved_device)
    dataset = _build_val_dataset(snapshot)
    records: list[dict[str, Any]] = []
    skipped_indices: list[int] = []
    for index, patch in enumerate(dataset):
        try:
            prediction = _predict_patch(
                model,
                patch,
                resolved_device,
                prior_type=str(snapshot["resolved"].get("prior_type", "gaussian")),
            )
        except RuntimeError as exc:
            if "teacher solve failed" in str(exc):
                skipped_indices.append(index)
                continue
            raise
        records.append(
            {
                "index": index,
                "patch_type": str(patch.get("patch_type", "unknown")),
                "beta": float(patch.get("beta", 0.0)),
                "relative_l2": float(prediction["relative_l2"]),
                "global_linf": float(prediction["global_linf"]),
                "neg_fraction": float(prediction["neg_fraction"]),
                "max_negative_magnitude": float(prediction["max_negative_magnitude"]),
                "cond_M": float(prediction["cond_M"]),
                "prediction": prediction,
            }
        )
    if not records:
        raise RuntimeError("no visualizable patches were found because teacher reference reconstruction failed")
    scores = [record["relative_l2"] for record in records]
    selected_positions = select_patch_indices(scores, num_patches=min(num_patches, len(records)), strategy=selection)
    output_dir = resolved_run_dir / "figures" / "shape_function_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_figures: list[str] = []
    selected_records: list[dict[str, Any]] = []
    for order, record_position in enumerate(selected_positions, start=1):
        record = records[record_position]
        dataset_index = int(record["index"])
        patch = dataset[dataset_index]
        figure_name = (
            f"{order:02d}_{selection}_{view}_idx{dataset_index:04d}_{record['patch_type']}"
            f"_relL2_{record['relative_l2']:.4f}.png"
        )
        figure_path = output_dir / figure_name
        if view == "bars":
            save_patch_visualization(figure_path, patch, record["prediction"], title_prefix=f"{selection} sample #{order}")
        elif view in {"contour", "hybrid3d"}:
            save_patch_field_visualization(
                figure_path,
                patch,
                record["prediction"],
                view=view,
                title_prefix=f"{selection} sample #{order}",
            )
        elif view in {"basis_contour", "basis_hybrid3d"}:
            basis_field = compute_basis_field(
                model,
                patch,
                device=resolved_device,
                feature_mode=str(snapshot["data"]["feature_mode"]),
                prior_type=str(snapshot["resolved"].get("prior_type", "gaussian")),
                grid_size=grid_size,
                node_index=node_index,
                fallback_values=record["prediction"]["phi_pred"],
            )
            save_basis_field_visualization(
                figure_path,
                patch,
                basis_field,
                view=view,
                title_prefix=f"{selection} sample #{order}",
            )
            record["basis_node_index"] = int(basis_field["node_index"])
            record["basis_teacher_success_ratio"] = float(basis_field["teacher_success_ratio"])
        else:
            raise ValueError(f"Unsupported visualization view: {view}")
        saved_figures.append(str(figure_path))
        selected_records.append({key: value for key, value in record.items() if key != "prediction"})
    summary = {
        "run_dir": str(resolved_run_dir),
        "device": resolved_device,
        "selection": selection,
        "view": view,
        "num_patches": int(num_patches),
        "num_available": int(len(dataset)),
        "num_visualizable": int(len(records)),
        "num_skipped_teacher_failures": int(len(skipped_indices)),
        "skipped_teacher_failure_indices": skipped_indices,
        "rel_error_floor": REL_ERROR_FLOOR,
        "rel_error_clip": REL_ERROR_CLIP,
        "figures": saved_figures,
        "selected": selected_records,
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate patch-level shape function visualizations for a completed run")
    parser.add_argument("--run-dir", required=True, help="Path to runs/<run_name> directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-patches", type=int, default=4, help="Number of sample patches to visualize")
    parser.add_argument(
        "--selection",
        choices=("spread", "worst", "best"),
        default="spread",
        help="How to choose validation patches for visualization",
    )
    parser.add_argument(
        "--view",
        choices=("training_curves", "bars", "contour", "hybrid3d", "basis_contour", "basis_hybrid3d"),
        default="basis_contour",
        help="Visualization style for each selected patch",
    )
    parser.add_argument("--node-index", type=int, default=None, help="Optional fixed node index for basis-field views")
    parser.add_argument("--grid-size", type=int, default=60, help="Grid resolution for basis-field views")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = generate_run_visualizations(
        Path(args.run_dir),
        device=str(args.device),
        num_patches=int(args.num_patches),
        selection=str(args.selection),
        view=str(args.view),
        node_index=args.node_index,
        grid_size=int(args.grid_size),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
