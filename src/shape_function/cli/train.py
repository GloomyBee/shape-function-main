from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from shape_function.cli.config import ResolvedTrainConfig, build_config_snapshot, resolve_train_config
from shape_function.data.dataloader import PatchDataset
from shape_function.data.dataset_builder import build_dataset
from shape_function.eval.eval_backbones import evaluate_model
from shape_function.models.full_model import build_shape_function_model
from shape_function.train.trainer import train_model
from shape_function.utils.artifacts import save_json
from shape_function.utils.logging import build_case_name
from shape_function.utils.seed import seed_everything


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def add_train_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("train", help="Train a shape function model from YAML configs")
    parser.add_argument("--data-config", required=True, help="Path to data YAML config")
    parser.add_argument("--train-config", required=True, help="Path to training YAML config")
    parser.add_argument("--run-name", default=None, help="Optional explicit run name")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Execution device")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override for data.yaml")
    parser.set_defaults(handler=run_train_command)


def _resolve_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this machine")
    return requested_device


def _default_run_name(resolved: ResolvedTrainConfig) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = build_case_name(
        backbone=resolved.backbone_name,
        feature_mode=resolved.feature_mode,
        k=resolved.k_neighbors,
        seed=resolved.seed,
    )
    return f"{base_name}_{timestamp}"


def _recommended_worker_count(device: str, dataset_size: int) -> int:
    if device != "cuda" or dataset_size < 1024:
        return 0
    cpu_count = os.cpu_count() or 0
    if cpu_count <= 2:
        return 0
    return min(4, max(1, cpu_count // 2))


def _build_loader(dataset: list[dict[str, Any]], batch_size: int, shuffle: bool, device: str) -> DataLoader:
    if not dataset:
        raise RuntimeError("dataset construction produced zero valid samples")
    worker_count = _recommended_worker_count(device, len(dataset))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": device == "cuda",
        "num_workers": worker_count,
    }
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = True
    return DataLoader(PatchDataset(dataset), **loader_kwargs)


def _build_datasets(resolved: ResolvedTrainConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    supervision_mode = "teacher" if resolved.loss["mode"] == "legacy_teacher_baseline" else "none"
    common_kwargs = {
        "feature_mode": resolved.feature_mode,
        "k_neighbors": resolved.k_neighbors,
        "prior_type": resolved.train["prior_type"],
        "patch_types": tuple(str(item) for item in resolved.data["patch_types"]),
        "beta_range": tuple(float(value) for value in resolved.data["beta_range"]),
        "supervision_mode": supervision_mode,
    }
    train_dataset = build_dataset(
        num_patches=int(resolved.data["num_train"]),
        seed=resolved.seed,
        **common_kwargs,
    )
    val_dataset = build_dataset(
        num_patches=int(resolved.data["num_val"]),
        seed=resolved.seed + 1,
        **common_kwargs,
    )
    return train_dataset, val_dataset


def _save_config_snapshot(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    resolved: ResolvedTrainConfig,
    *,
    run_name: str,
    device: str,
    best_epoch: int,
    best_val_total: float,
) -> None:
    checkpoint = {
        "model_state_dict": _cpu_state_dict(model),
        "data_config": resolved.data,
        "train_config": {
            "model": resolved.model,
            "train": resolved.train,
            "head": resolved.head,
            "loss": resolved.loss,
            "eval": resolved.eval,
        },
        "metadata": {
            "run_name": run_name,
            "device": device,
            "seed": resolved.seed,
            "feature_mode": resolved.feature_mode,
            "backbone": resolved.backbone_name,
            "k_neighbors": resolved.k_neighbors,
            "best_epoch": int(best_epoch),
            "best_val_total": float(best_val_total),
            "basis_order": int(resolved.head["basis_order"]),
            "loss_mode": str(resolved.loss["mode"]),
        },
    }
    torch.save(checkpoint, path)


def run_train_command(args: argparse.Namespace) -> int:
    repo_root = Path.cwd()
    resolved = resolve_train_config(
        Path(args.data_config),
        Path(args.train_config),
        seed_override=args.seed,
    )
    run_name = str(args.run_name) if args.run_name else _default_run_name(resolved)
    run_root = repo_root / "runs" / run_name
    if run_root.exists():
        raise FileExistsError(f"run directory already exists: {run_root}")
    device = _resolve_device(str(args.device))
    seed_everything(resolved.seed)
    train_dataset, val_dataset = _build_datasets(resolved)
    batch_size = int(resolved.train["batch_size"])
    train_loader = _build_loader(train_dataset, batch_size=batch_size, shuffle=True, device=device)
    val_loader = _build_loader(val_dataset, batch_size=batch_size, shuffle=False, device=device)
    model = build_shape_function_model(
        backbone_name=resolved.backbone_name,
        feature_mode=resolved.feature_mode,
        hidden_dim=int(resolved.model["hidden_dim"]),
        num_layers=int(resolved.model["num_layers"]) if "num_layers" in resolved.model else None,
        k_neighbors=resolved.k_neighbors,
        head_kwargs=dict(resolved.head),
    )
    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        repo_root=repo_root,
        run_name=run_name,
        epochs=int(resolved.train["epochs"]),
        learning_rate=float(resolved.train["learning_rate"]),
        device=device,
        loss_mode=str(resolved.loss["mode"]),
        loss_weights={key: float(value) for key, value in resolved.loss.items() if key != "mode"},
        compute_teacher_metrics=bool(resolved.eval["compute_teacher_metrics"]),
    )
    eval_metrics = evaluate_model(
        model,
        val_loader,
        device=device,
        compute_teacher_metrics=bool(resolved.eval["compute_teacher_metrics"]),
    )
    artifacts = train_result["artifacts"]
    save_json(artifacts.root_dir / "eval_metrics.json", eval_metrics)
    _save_checkpoint(
        artifacts.root_dir / "checkpoint.pt",
        model,
        resolved,
        run_name=run_name,
        device=device,
        best_epoch=int(train_result["metrics"]["best_epoch"]),
        best_val_total=float(train_result["metrics"]["best_val_total"]),
    )
    snapshot = build_config_snapshot(resolved, run_name=run_name, device=device, repo_root=repo_root)
    _save_config_snapshot(artifacts.root_dir / "config_snapshot.yaml", snapshot)
    print(f"run_name: {run_name}")
    print(f"device: {device}")
    print(f"train_samples: {len(train_dataset)}")
    print(f"val_samples: {len(val_dataset)}")
    print(f"loss_mode: {resolved.loss['mode']}")
    print("training complete")
    print(f"best_val_total: {train_result['metrics']['best_val_total']:.6e}")
    print(f"final_val_total: {train_result['metrics']['final_val_total']:.6e}")
    if "relative_l2" in eval_metrics:
        print(f"eval_relative_l2: {eval_metrics['relative_l2']:.6e}")
    else:
        print(f"eval_base_linear_residual: {eval_metrics['base_linear_residual']:.6e}")
    print(f"artifacts_dir: {artifacts.root_dir}")
    return 0
