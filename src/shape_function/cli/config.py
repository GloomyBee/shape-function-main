from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from shape_function.models.full_model import feature_dim_for_mode


class ConfigError(ValueError):
    """Raised when a CLI configuration file is invalid."""


@dataclass(frozen=True)
class ResolvedTrainConfig:
    data_path: Path
    train_path: Path
    data: dict[str, Any]
    model: dict[str, Any]
    train: dict[str, Any]
    head: dict[str, Any]
    loss: dict[str, Any]
    eval: dict[str, Any]
    seed: int
    feature_mode: str
    feature_dim: int
    backbone_name: str
    k_neighbors: int


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ConfigError(f"Configuration file must contain a mapping: {path}")
    return payload


def _require_keys(payload: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ConfigError(f"Missing required keys in {context}: {', '.join(missing)}")


def resolve_train_config(
    data_config_path: Path,
    train_config_path: Path,
    seed_override: int | None = None,
) -> ResolvedTrainConfig:
    data_payload = _load_yaml_mapping(data_config_path)
    train_payload = _load_yaml_mapping(train_config_path)
    _require_keys(
        data_payload,
        ("seed", "num_train", "num_val", "k_neighbors", "beta_range", "feature_mode", "patch_types"),
        f"data config {data_config_path}",
    )
    _require_keys(train_payload, ("model", "train"), f"train config {train_config_path}")
    if not isinstance(train_payload["model"], dict):
        raise ConfigError("train config field 'model' must be a mapping")
    if not isinstance(train_payload["train"], dict):
        raise ConfigError("train config field 'train' must be a mapping")
    model_payload = dict(train_payload["model"])
    train_section = dict(train_payload["train"])
    head_section = dict(train_payload.get("head", {}))
    loss_section = dict(train_payload.get("loss", {}))
    eval_section = dict(train_payload.get("eval", {}))
    _require_keys(model_payload, ("backbone", "hidden_dim"), f"model section in {train_config_path}")
    _require_keys(train_section, ("batch_size", "epochs", "learning_rate"), f"train section in {train_config_path}")
    feature_mode = str(data_payload["feature_mode"])
    if "feature_mode" in model_payload and str(model_payload["feature_mode"]) != feature_mode:
        raise ConfigError(
            "feature_mode mismatch between data config and train config: "
            f"{feature_mode!r} != {model_payload['feature_mode']!r}"
        )
    backbone_name = str(model_payload["backbone"])
    if backbone_name in {"kernel_operator", "deepsets"} and "num_layers" not in model_payload:
        raise ConfigError(f"{backbone_name} requires model.num_layers in the train config")
    if backbone_name not in {"kernel_operator", "deepsets", "mlp_baseline"}:
        raise ConfigError(f"Unsupported model.backbone: {backbone_name}")
    beta_range = data_payload["beta_range"]
    if not isinstance(beta_range, (list, tuple)) or len(beta_range) != 2:
        raise ConfigError("data config field 'beta_range' must contain exactly two numbers")
    patch_types = data_payload["patch_types"]
    if not isinstance(patch_types, list) or not patch_types:
        raise ConfigError("data config field 'patch_types' must be a non-empty list")
    loss_mode = str(loss_section.get("mode", "unsupervised_v1"))
    head_section.setdefault("basis_order", 1)
    head_section.setdefault("fallback_mode", "hard")
    train_section.setdefault("prior_type", "gaussian")
    eval_section.setdefault("compute_teacher_metrics", False)
    if head_section["basis_order"] not in (1, 2):
        raise ConfigError("basis_order must be 1 or 2")
    if head_section["fallback_mode"] != "hard":
        raise ConfigError("fallback_mode must be 'hard'")
    if train_section["prior_type"] != "gaussian":
        raise ConfigError("prior_type must be 'gaussian'")
    if head_section["basis_order"] == 2 and "kappa_max" not in head_section:
        raise ConfigError("basis_order=2 requires head.kappa_max")
    if loss_mode == "unsupervised_v1":
        _require_keys(loss_section, ("lambda_base_lin", "lambda_ent", "lambda_neg"), f"loss section in {train_config_path}")
    elif loss_mode == "legacy_teacher_baseline":
        _require_keys(loss_section, ("lambda_data", "lambda_cons", "lambda_neg"), f"loss section in {train_config_path}")
    else:
        raise ConfigError(f"Unsupported loss.mode: {loss_mode}")
    feature_dim = feature_dim_for_mode(feature_mode)
    seed = int(seed_override if seed_override is not None else data_payload["seed"])
    return ResolvedTrainConfig(
        data_path=data_config_path,
        train_path=train_config_path,
        data=dict(data_payload),
        model=model_payload,
        train=train_section,
        head=head_section,
        loss=loss_section,
        eval=eval_section,
        seed=seed,
        feature_mode=feature_mode,
        feature_dim=feature_dim,
        backbone_name=backbone_name,
        k_neighbors=int(data_payload["k_neighbors"]),
    )


def build_config_snapshot(
    resolved: ResolvedTrainConfig,
    *,
    run_name: str,
    device: str,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "data": resolved.data,
        "train": {
            "model": resolved.model,
            "train": resolved.train,
            "head": resolved.head,
            "loss": resolved.loss,
            "eval": resolved.eval,
        },
        "resolved": {
            "seed": resolved.seed,
            "feature_mode": resolved.feature_mode,
            "feature_dim": resolved.feature_dim,
            "backbone": resolved.backbone_name,
            "k_neighbors": resolved.k_neighbors,
            "prior_type": resolved.train["prior_type"],
            "train_seed": resolved.seed,
            "val_seed": resolved.seed + 1,
            "device": device,
            "run_name": run_name,
            "repo_root": str(repo_root),
            "loss_mode": resolved.loss["mode"],
            "basis_order": resolved.head["basis_order"],
        },
    }
