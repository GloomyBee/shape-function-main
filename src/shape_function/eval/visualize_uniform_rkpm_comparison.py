from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Rectangle

from shape_function.data.feature_builder import build_patch_features
from shape_function.models.full_model import build_shape_function_model
from shape_function.models.heads.reproducing_correction import apply_reproducing_correction
from shape_function.utils.artifacts import save_json


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def _resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine")
    return requested_device


def _build_model(snapshot: dict[str, Any], checkpoint_path: Path, device: str) -> torch.nn.Module:
    model_cfg = snapshot["train"]["model"]
    data_cfg = snapshot["data"]
    head_cfg = snapshot["train"].get("head", {})
    model = build_shape_function_model(
        backbone_name=str(model_cfg["backbone"]),
        feature_mode=str(data_cfg["feature_mode"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]) if "num_layers" in model_cfg else None,
        k_neighbors=int(data_cfg["k_neighbors"]),
        head_kwargs=dict(head_cfg),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _build_uniform_patch(beta: float, spacing: float = 0.6666666667) -> dict[str, Any]:
    coords = np.linspace(-1.0, 1.0, 4, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="xy")
    X = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    x_q = np.zeros((2,), dtype=np.float64)
    r_max = float(np.linalg.norm(X - x_q[None, :], axis=1).max())
    return {
        "X": X,
        "x_q": x_q,
        "beta": float(beta),
        "r_max": r_max,
        "patch_type": "uniform_manual",
    }


def _select_representative_node(X: np.ndarray) -> int:
    target = np.asarray([0.34, 0.34], dtype=np.float64)
    distances = np.linalg.norm(X - target[None, :], axis=1)
    return int(np.argmin(distances))


def _build_query_grid(limit: float = 1.0, grid_size: int = 101) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(-limit, limit, grid_size, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(coords, coords, indexing="xy")
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    return grid_x, grid_y, points


def _build_rho_batch(
    X: np.ndarray,
    points: np.ndarray,
    beta: float,
    feature_mode: str,
    k_neighbors: int,
) -> torch.Tensor | None:
    if feature_mode == "minimal":
        return None
    rows: list[np.ndarray] = []
    for point in points:
        patch = {
            "X": X,
            "x_q": point,
            "beta": beta,
            "r_max": float(np.linalg.norm(X - point[None, :], axis=1).max()),
        }
        features = build_patch_features(patch, feature_mode=feature_mode, k_max=k_neighbors)
        rows.append(np.asarray(features["rho_q"], dtype=np.float64))
    return torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float64)


def _window_weights(dist: torch.Tensor, support_radius: torch.Tensor, window_type: str) -> torch.Tensor:
    s = dist / support_radius.clamp_min(1.0e-12)
    if window_type == "quartic":
        inside = s < 1.0
        out = torch.zeros_like(s)
        out[inside] = (1.0 - s[inside].square()) ** 2
        return out
    if window_type == "wendland_c2":
        inside = s < 1.0
        out = torch.zeros_like(s)
        t = 1.0 - s[inside]
        out[inside] = t.pow(4) * (4.0 * s[inside] + 1.0)
        return out
    raise ValueError(f"Unsupported window_type: {window_type}")


def _predict_model_field(
    model: torch.nn.Module,
    snapshot: dict[str, Any],
    patch: dict[str, Any],
    points: np.ndarray,
    device: str,
    batch_size: int,
    node_index: int,
) -> dict[str, np.ndarray]:
    X_np = np.asarray(patch["X"], dtype=np.float64)
    beta_value = float(patch["beta"])
    feature_mode = str(snapshot["data"]["feature_mode"])
    k_neighbors = int(snapshot["resolved"]["k_neighbors"])
    rho_batch = _build_rho_batch(X_np, points, beta_value, feature_mode, k_neighbors)

    phi_values: list[np.ndarray] = []
    quad_residuals: list[np.ndarray] = []
    linear_residuals: list[np.ndarray] = []
    for start in range(0, points.shape[0], batch_size):
        stop = min(start + batch_size, points.shape[0])
        batch_points = points[start:stop]
        X_batch = torch.as_tensor(np.repeat(X_np[None, :, :], batch_points.shape[0], axis=0), dtype=torch.float64, device=device)
        x_q_batch = torch.as_tensor(batch_points, dtype=torch.float64, device=device)
        beta_batch = torch.full((batch_points.shape[0], 1), beta_value, dtype=torch.float64, device=device)
        rho_q = None
        if rho_batch is not None:
            rho_q = rho_batch[start:stop].to(device)
        with torch.no_grad():
            outputs = model(X_batch, x_q_batch, beta_batch, rho_q=rho_q)
        phi_values.append(outputs["phi_corr"][:, node_index].detach().cpu().numpy())
        quad_residuals.append(outputs["aux"]["reproducing_residual_quadratic"].detach().cpu().numpy())
        linear_residuals.append(outputs["aux"]["reproducing_residual_linear"].detach().cpu().numpy())
    return {
        "phi": np.concatenate(phi_values, axis=0),
        "quad_residual": np.concatenate(quad_residuals, axis=0),
        "linear_residual": np.concatenate(linear_residuals, axis=0),
    }


def _predict_rkpm_field(
    model: torch.nn.Module,
    patch: dict[str, Any],
    points: np.ndarray,
    device: str,
    batch_size: int,
    node_index: int,
) -> dict[str, np.ndarray]:
    X_np = np.asarray(patch["X"], dtype=np.float64)
    X_all = torch.as_tensor(X_np, dtype=torch.float64, device=device)
    support_radius_scale = float(getattr(model.head, "support_radius_scale", 1.05))
    basis_order = int(getattr(model.head, "basis_order", 2))
    eps_reg = float(getattr(model.head, "eps_reg", 1.0e-10))
    kappa_max = getattr(model.head, "kappa_max", None)
    fallback_mode = str(getattr(model.head, "fallback_mode", "hard"))
    window_type = str(getattr(model.head, "window_type", "quartic"))

    phi_values: list[np.ndarray] = []
    quad_residuals: list[np.ndarray] = []
    linear_residuals: list[np.ndarray] = []
    for start in range(0, points.shape[0], batch_size):
        stop = min(start + batch_size, points.shape[0])
        batch_points = points[start:stop]
        x_q_batch = torch.as_tensor(batch_points, dtype=torch.float64, device=device)
        X_batch = X_all.unsqueeze(0).expand(batch_points.shape[0], -1, -1)
        rel = X_batch - x_q_batch[:, None, :]
        dist = torch.linalg.norm(rel, dim=-1)
        r_max = dist.max(dim=-1, keepdim=True).values.clamp_min(1.0e-12)
        support_radius = support_radius_scale * r_max
        raw = _window_weights(dist, support_radius, window_type=window_type)
        phi_base = raw / raw.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        correction = apply_reproducing_correction(
            phi_base,
            X_batch,
            x_q_batch,
            r_max,
            basis_order=basis_order,
            eps_reg=eps_reg,
            kappa_max=kappa_max,
            fallback_mode=fallback_mode,
        )
        phi_values.append(correction["phi_corr"][:, node_index].detach().cpu().numpy())
        quad_residuals.append(correction["reproducing_residual_quadratic"].detach().cpu().numpy())
        linear_residuals.append(correction["reproducing_residual_linear"].detach().cpu().numpy())
    return {
        "phi": np.concatenate(phi_values, axis=0),
        "quad_residual": np.concatenate(quad_residuals, axis=0),
        "linear_residual": np.concatenate(linear_residuals, axis=0),
    }


def _to_grid(values: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(grid_shape)


def _log10_residual(values: np.ndarray, floor: float = 1.0e-16) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=np.float64), floor)
    return np.log10(clipped)


def save_uniform_rkpm_comparison(
    output_path: Path,
    patch: dict[str, Any],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    node_index: int,
    rkpm: dict[str, np.ndarray],
    model_pred: dict[str, np.ndarray],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    X = np.asarray(patch["X"], dtype=np.float64)
    node_coord = X[node_index]
    rkpm_phi = _to_grid(rkpm["phi"], grid_x.shape)
    model_phi = _to_grid(model_pred["phi"], grid_x.shape)
    diff = np.abs(model_phi - rkpm_phi)
    rkpm_quad = _to_grid(rkpm["quad_residual"], grid_x.shape)
    model_quad = _to_grid(model_pred["quad_residual"], grid_x.shape)
    rkpm_quad_log = _log10_residual(rkpm_quad)
    model_quad_log = _log10_residual(model_quad)

    phi_min = float(min(np.nanmin(rkpm_phi), np.nanmin(model_phi)))
    phi_max = float(max(np.nanmax(rkpm_phi), np.nanmax(model_phi), 1.0e-12))
    diff_max = float(max(np.nanmax(diff), 1.0e-12))
    quad_min = float(min(np.nanmin(rkpm_quad_log), np.nanmin(model_quad_log)))
    quad_max = float(max(np.nanmax(rkpm_quad_log), np.nanmax(model_quad_log)))

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_geom = fig.add_subplot(gs[0, 0])
    ax_rkpm = fig.add_subplot(gs[0, 1], projection="3d")
    ax_model = fig.add_subplot(gs[0, 2], projection="3d")
    ax_diff = fig.add_subplot(gs[1, 0])
    ax_rkpm_quad = fig.add_subplot(gs[1, 1])
    ax_model_quad = fig.add_subplot(gs[1, 2])

    ax_geom.scatter(X[:, 0], X[:, 1], s=42, c="#203864", edgecolors="white", linewidths=0.6, zorder=3)
    ax_geom.scatter(node_coord[0], node_coord[1], s=220, marker="*", c="gold", edgecolors="black", linewidths=0.8, zorder=4)
    ax_geom.add_patch(Rectangle((-1.0, -1.0), 2.0, 2.0, fill=False, linestyle="--", linewidth=1.2, edgecolor="#666666"))
    for idx, point in enumerate(X):
        ax_geom.text(point[0] + 0.03, point[1] + 0.03, str(idx), fontsize=8)
    ax_geom.set_title("Uniform 4x4 Patch Geometry")
    ax_geom.set_aspect("equal")
    ax_geom.grid(True, alpha=0.20)
    ax_geom.set_xlim(-1.15, 1.15)
    ax_geom.set_ylim(-1.15, 1.15)

    info_lines = [
        f"selected node = {node_index}",
        f"node coord = ({node_coord[0]:.3f}, {node_coord[1]:.3f})",
        f"beta = {float(patch['beta']):.2f}",
        f"field relL2(model vs RKPM) = {np.linalg.norm(model_phi - rkpm_phi) / max(np.linalg.norm(rkpm_phi), 1.0e-12):.3e}",
        f"RKPM mean quad residual = {np.mean(rkpm['quad_residual']):.3e}",
        f"Ours mean quad residual = {np.mean(model_pred['quad_residual']):.3e}",
        f"RKPM max quad residual = {np.max(rkpm['quad_residual']):.3e}",
        f"Ours max quad residual = {np.max(model_pred['quad_residual']):.3e}",
    ]
    ax_geom.text(
        0.03,
        0.03,
        "\n".join(info_lines),
        transform=ax_geom.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "0.75"},
    )

    surf1 = ax_rkpm.plot_surface(grid_x, grid_y, rkpm_phi, cmap="coolwarm", vmin=phi_min, vmax=phi_max, linewidth=0.0, antialiased=True)
    ax_rkpm.set_title("RKPM Quadratic Basis Surface")
    ax_rkpm.set_xlabel("x")
    ax_rkpm.set_ylabel("y")
    ax_rkpm.set_zlabel("phi")
    ax_rkpm.view_init(elev=28, azim=-55)
    plt.colorbar(surf1, ax=ax_rkpm, shrink=0.72, pad=0.08)

    surf2 = ax_model.plot_surface(grid_x, grid_y, model_phi, cmap="coolwarm", vmin=phi_min, vmax=phi_max, linewidth=0.0, antialiased=True)
    ax_model.set_title("Neural O2 Basis Surface")
    ax_model.set_xlabel("x")
    ax_model.set_ylabel("y")
    ax_model.set_zlabel("phi")
    ax_model.view_init(elev=28, azim=-55)
    plt.colorbar(surf2, ax=ax_model, shrink=0.72, pad=0.08)

    contour1 = ax_diff.contourf(grid_x, grid_y, diff, levels=18, cmap="Reds", vmin=0.0, vmax=diff_max)
    ax_diff.scatter(X[:, 0], X[:, 1], s=12, c="black", alpha=0.7)
    ax_diff.scatter(node_coord[0], node_coord[1], s=120, marker="*", c="gold", edgecolors="black", linewidths=0.8)
    ax_diff.set_title("|Neural - RKPM| Basis Difference")
    ax_diff.set_aspect("equal")
    ax_diff.grid(True, alpha=0.20)
    plt.colorbar(contour1, ax=ax_diff, shrink=0.86)

    contour2 = ax_rkpm_quad.contourf(grid_x, grid_y, rkpm_quad_log, levels=18, cmap="viridis", vmin=quad_min, vmax=quad_max)
    ax_rkpm_quad.scatter(X[:, 0], X[:, 1], s=12, c="black", alpha=0.7)
    ax_rkpm_quad.scatter(node_coord[0], node_coord[1], s=120, marker="*", c="gold", edgecolors="black", linewidths=0.8)
    ax_rkpm_quad.set_title("RKPM log10 Quadratic Residual")
    ax_rkpm_quad.set_aspect("equal")
    ax_rkpm_quad.grid(True, alpha=0.20)
    plt.colorbar(contour2, ax=ax_rkpm_quad, shrink=0.86)

    contour3 = ax_model_quad.contourf(grid_x, grid_y, model_quad_log, levels=18, cmap="viridis", vmin=quad_min, vmax=quad_max)
    ax_model_quad.scatter(X[:, 0], X[:, 1], s=12, c="black", alpha=0.7)
    ax_model_quad.scatter(node_coord[0], node_coord[1], s=120, marker="*", c="gold", edgecolors="black", linewidths=0.8)
    ax_model_quad.set_title("Neural O2 log10 Quadratic Residual")
    ax_model_quad.set_aspect("equal")
    ax_model_quad.grid(True, alpha=0.20)
    plt.colorbar(contour3, ax=ax_model_quad, shrink=0.86)

    fig.suptitle("Uniform Patch RKPM vs Neural O2 | Focus on Quadratic Consistency", fontsize=15)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_uniform_rkpm_comparison(
    run_dir: Path,
    *,
    device: str = "auto",
    beta: float = 3.0,
    grid_size: int = 101,
    batch_size: int = 1024,
    node_index: int | None = None,
) -> dict[str, Any]:
    resolved_run_dir = run_dir.resolve()
    snapshot = _load_yaml(resolved_run_dir / "config_snapshot.yaml")
    resolved_device = _resolve_device(device)
    model = _build_model(snapshot, resolved_run_dir / "checkpoint.pt", resolved_device)
    patch = _build_uniform_patch(beta=beta)
    X = np.asarray(patch["X"], dtype=np.float64)
    resolved_node_index = _select_representative_node(X) if node_index is None else int(node_index)
    grid_x, grid_y, points = _build_query_grid(limit=1.0, grid_size=grid_size)
    rkpm = _predict_rkpm_field(model, patch, points, resolved_device, batch_size, resolved_node_index)
    model_pred = _predict_model_field(model, snapshot, patch, points, resolved_device, batch_size, resolved_node_index)

    output_dir = resolved_run_dir / "figures" / "uniform_rkpm_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "uniform_rkpm_comparison.png"
    save_uniform_rkpm_comparison(figure_path, patch, grid_x, grid_y, resolved_node_index, rkpm, model_pred)

    summary = {
        "run_dir": str(resolved_run_dir),
        "device": resolved_device,
        "beta": float(beta),
        "grid_size": int(grid_size),
        "node_index": int(resolved_node_index),
        "figure": str(figure_path),
        "field_relative_l2": float(np.linalg.norm(model_pred["phi"] - rkpm["phi"]) / max(np.linalg.norm(rkpm["phi"]), 1.0e-12)),
        "rkpm_mean_quad_residual": float(np.mean(rkpm["quad_residual"])),
        "neural_mean_quad_residual": float(np.mean(model_pred["quad_residual"])),
        "rkpm_max_quad_residual": float(np.max(rkpm["quad_residual"])),
        "neural_max_quad_residual": float(np.max(model_pred["quad_residual"])),
    }
    save_json(output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a uniform-patch RKPM vs neural O2 comparison figure")
    parser.add_argument("--run-dir", required=True, help="Path to runs/<run_name> directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--node-index", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = generate_uniform_rkpm_comparison(
        Path(args.run_dir),
        device=str(args.device),
        beta=float(args.beta),
        grid_size=int(args.grid_size),
        batch_size=int(args.batch_size),
        node_index=None if args.node_index is None else int(args.node_index),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
