from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from shape_function.data.dataloader import PatchDataset
from shape_function.data.feature_builder import build_patch_features
from shape_function.data.maxent_solver import solve_maxent_patch
from shape_function.data.patch_sampler import PATCH_TYPES, sample_patches
from shape_function.eval.eval_backbones import evaluate_model
from shape_function.models.full_model import build_shape_function_model
from shape_function.utils.artifacts import save_json
from shape_function.utils.seed import seed_everything


@dataclass(frozen=True)
class OODProtocol:
    name: str
    kind: str
    params: dict[str, Any]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return payload


def _resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine")
    return requested_device


def _default_eval_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_name(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace(" ", "_")


def expand_protocols(ood_config: dict[str, Any]) -> list[OODProtocol]:
    protocols = ood_config.get("protocols")
    if not isinstance(protocols, dict):
        raise ValueError("ood config requires a 'protocols' mapping")
    expanded: list[OODProtocol] = []
    for name in ("iid", "loto", "rotation", "k_ood", "scale", "boundary"):
        cfg = protocols.get(name, {})
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise ValueError(f"protocol {name!r} must be a mapping")
        if not bool(cfg.get("enabled", False)):
            continue
        base_params = {key: value for key, value in cfg.items() if key != "enabled"}
        if name == "rotation":
            for angle in base_params.get("angles", []):
                params = dict(base_params)
                params["angle_degrees"] = float(angle)
                expanded.append(OODProtocol(name=f"rotation_{_sanitize_name(angle)}", kind=name, params=params))
        elif name == "loto":
            for patch_type in _as_tuple_of_str(base_params.get("held_out", PATCH_TYPES)):
                params = dict(base_params)
                params["held_out"] = patch_type
                params["patch_types"] = [patch_type]
                expanded.append(OODProtocol(name=f"loto_{patch_type}", kind=name, params=params))
        elif name == "scale":
            beta_ranges = base_params.get("beta_ranges", [base_params.get("beta_range", [8.0, 16.0])])
            for beta_range in beta_ranges:
                resolved_range = _as_beta_range(beta_range, (8.0, 16.0))
                params = dict(base_params)
                params["beta_range"] = list(resolved_range)
                label = f"{_sanitize_name(resolved_range[0])}_{_sanitize_name(resolved_range[1])}"
                params["beta_range_label"] = label
                expanded.append(OODProtocol(name=f"scale_beta_{label}", kind=name, params=params))
        elif name == "k_ood":
            for k_value in base_params.get("k_values", []):
                params = dict(base_params)
                params["k_neighbors"] = int(k_value)
                expanded.append(OODProtocol(name=f"k_{int(k_value)}", kind=name, params=params))
        else:
            expanded.append(OODProtocol(name=name, kind=name, params=base_params))
    return expanded


def _load_model_from_run(run_dir: Path, device: str) -> tuple[torch.nn.Module, dict[str, Any], Path, Path]:
    snapshot_path = run_dir / "config_snapshot.yaml"
    checkpoint_path = run_dir / "checkpoint.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    snapshot = _load_yaml_mapping(snapshot_path)
    resolved = snapshot["resolved"]
    model_cfg = snapshot["train"]["model"]
    head_cfg = snapshot["train"]["head"]
    model = build_shape_function_model(
        backbone_name=str(resolved["backbone"]),
        feature_mode=str(resolved["feature_mode"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]) if "num_layers" in model_cfg else None,
        k_neighbors=int(resolved["k_neighbors"]),
        head_kwargs=dict(head_cfg),
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, snapshot, checkpoint_path, snapshot_path


def _recompute_r_max(patch: dict[str, Any]) -> None:
    X = np.asarray(patch["X"], dtype=np.float64)
    x_q = np.asarray(patch["x_q"], dtype=np.float64)
    distances = np.linalg.norm(X - x_q[None, :], axis=1)
    patch["r_max"] = float(np.max(distances))
    patch["meta"] = {**dict(patch.get("meta", {})), "distances": distances}


def rotate_patch_about_query(patch: dict[str, Any], angle_degrees: float) -> dict[str, Any]:
    item = dict(patch)
    X = np.asarray(item["X"], dtype=np.float64)
    x_q = np.asarray(item["x_q"], dtype=np.float64)
    theta = np.deg2rad(float(angle_degrees))
    rotation = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float64)
    item["X"] = (x_q[None, :] + (X - x_q[None, :]) @ rotation.T).astype(np.float64)
    _recompute_r_max(item)
    return item


def scale_patch_about_query(patch: dict[str, Any], scale_factor: float) -> dict[str, Any]:
    item = dict(patch)
    X = np.asarray(item["X"], dtype=np.float64)
    x_q = np.asarray(item["x_q"], dtype=np.float64)
    item["X"] = (x_q[None, :] + float(scale_factor) * (X - x_q[None, :])).astype(np.float64)
    _recompute_r_max(item)
    return item


def _as_tuple_of_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple(PATCH_TYPES)
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _as_beta_range(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("beta_range must contain exactly two numbers")
    return (float(value[0]), float(value[1]))


def _finalize_patch(
    patch: dict[str, Any],
    *,
    feature_mode: str,
    k_neighbors: int,
    prior_type: str,
    compute_teacher_metrics: bool,
) -> dict[str, Any] | None:
    item = dict(patch)
    features = build_patch_features(item, feature_mode=feature_mode, k_max=k_neighbors)
    item["rho_q"] = features["rho_q"]
    item["node_features"] = features["node_features"]
    item["rel_coords"] = features["rel_coords"]
    if compute_teacher_metrics:
        teacher = solve_maxent_patch(item["x_q"], item["X"], item["beta"], prior_type=prior_type)
        if not teacher["success"]:
            return None
        item["phi_ref"] = teacher["phi_ref"]
        item["teacher"] = teacher
    return item


def build_protocol_dataset(
    protocol: OODProtocol,
    snapshot: dict[str, Any],
    *,
    compute_teacher_metrics: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_cfg = snapshot["data"]
    resolved = snapshot["resolved"]
    feature_mode = str(resolved["feature_mode"])
    prior_type = str(resolved["prior_type"])
    default_beta = _as_beta_range(data_cfg["beta_range"], (0.5, 8.0))
    default_patch_types = tuple(str(item) for item in data_cfg["patch_types"])
    num_patches = int(protocol.params.get("num_patches", 64))
    seed = int(protocol.params.get("seed", int(resolved["seed"]) + 1000))
    k_neighbors = int(protocol.params.get("k_neighbors", int(resolved["k_neighbors"])))
    beta_range = _as_beta_range(protocol.params.get("beta_range"), default_beta)
    patch_types = _as_tuple_of_str(protocol.params.get("patch_types", default_patch_types))
    if protocol.kind == "boundary":
        patch_types = ("boundary_truncated",)
    raw_patches = sample_patches(
        num_patches,
        seed=seed,
        patch_types=patch_types,
        k_neighbors=k_neighbors,
        beta_range=beta_range,
    )
    dataset: list[dict[str, Any]] = []
    for patch in raw_patches:
        item = patch
        if protocol.kind == "rotation":
            item = rotate_patch_about_query(item, float(protocol.params["angle_degrees"]))
        finalized = _finalize_patch(
            item,
            feature_mode=feature_mode,
            k_neighbors=k_neighbors,
            prior_type=prior_type,
            compute_teacher_metrics=compute_teacher_metrics,
        )
        if finalized is not None:
            dataset.append(finalized)
    dataset_params = {
        "protocol": protocol.kind,
        "num_requested": num_patches,
        "num_built": len(dataset),
        "seed": seed,
        "k_neighbors": k_neighbors,
        "patch_types": list(patch_types),
        "beta_range": list(beta_range),
        "compute_teacher_metrics": compute_teacher_metrics,
    }
    if protocol.kind == "rotation":
        dataset_params["angle_degrees"] = float(protocol.params["angle_degrees"])
    if protocol.kind == "scale":
        dataset_params["scale_semantics"] = "beta_range_extrapolation"
    if protocol.kind == "loto":
        dataset_params["held_out"] = str(protocol.params.get("held_out", patch_types[0]))
    return dataset, dataset_params


def _backbone_supports_protocol(snapshot: dict[str, Any], protocol: OODProtocol) -> tuple[bool, str | None]:
    if protocol.kind != "k_ood":
        return True, None
    backbone = str(snapshot["resolved"]["backbone"])
    train_k = int(snapshot["resolved"]["k_neighbors"])
    eval_k = int(protocol.params["k_neighbors"])
    if backbone == "mlp_baseline" and eval_k != train_k:
        return False, f"mlp_baseline was trained with k_neighbors={train_k}, cannot evaluate k_neighbors={eval_k}"
    return True, None


def run_ood_eval(
    *,
    run_dir: Path,
    ood_config_path: Path,
    device: str = "auto",
    eval_name: str | None = None,
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    model, snapshot, checkpoint_path, snapshot_path = _load_model_from_run(run_dir, resolved_device)
    ood_config = _load_yaml_mapping(ood_config_path)
    seed_everything(int(ood_config.get("seed", snapshot["resolved"]["seed"])))
    output_root = run_dir / "ood_eval" / (eval_name or _default_eval_name())
    output_root.mkdir(parents=True, exist_ok=True)
    defaults = dict(ood_config.get("defaults", {}))
    results: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for protocol in expand_protocols(ood_config):
        supported, reason = _backbone_supports_protocol(snapshot, protocol)
        if not supported:
            skipped[protocol.name] = str(reason)
            continue
        compute_teacher_metrics = bool(protocol.params.get("compute_teacher_metrics", defaults.get("compute_teacher_metrics", False)))
        batch_size = int(protocol.params.get("batch_size", defaults.get("batch_size", 64)))
        dataset, dataset_params = build_protocol_dataset(protocol, snapshot, compute_teacher_metrics=compute_teacher_metrics)
        if not dataset:
            skipped[protocol.name] = "protocol produced zero valid samples"
            continue
        metrics = evaluate_model(
            model,
            DataLoader(PatchDataset(dataset), batch_size=batch_size, shuffle=False),
            device=resolved_device,
            compute_teacher_metrics=compute_teacher_metrics,
        )
        results[protocol.name] = {"metrics": metrics, "num_samples": len(dataset), "dataset_params": dataset_params}
    payload = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "config_snapshot_path": str(snapshot_path),
        "device": resolved_device,
        "ood_config": ood_config,
        "results": results,
        "skipped": skipped,
    }
    save_json(output_root / "ood_eval_metrics.json", payload)
    return {"output_dir": str(output_root), **payload}


def add_ood_eval_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("ood-eval", help="Evaluate a trained run on OOD patch protocols")
    parser.add_argument("--run-dir", required=True, help="Path to runs/<run_name> directory")
    parser.add_argument("--ood-config", required=True, help="Path to OOD evaluation YAML config")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Execution device")
    parser.add_argument("--eval-name", default=None, help="Optional output subdirectory name under run_dir/ood_eval")
    parser.set_defaults(handler=run_ood_eval_command)


def run_ood_eval_command(args: argparse.Namespace) -> int:
    payload = run_ood_eval(
        run_dir=Path(args.run_dir),
        ood_config_path=Path(args.ood_config),
        device=str(args.device),
        eval_name=args.eval_name,
    )
    print(f"ood_eval_dir: {payload['output_dir']}")
    print(f"results: {len(payload['results'])}")
    print(f"skipped: {len(payload['skipped'])}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m shape_function.eval.ood_eval")
    parser.add_argument("--run-dir", required=True, help="Path to runs/<run_name> directory")
    parser.add_argument("--ood-config", required=True, help="Path to OOD evaluation YAML config")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Execution device")
    parser.add_argument("--eval-name", default=None, help="Optional output subdirectory name under run_dir/ood_eval")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return run_ood_eval_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
