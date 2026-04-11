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
    _require_keys(model_payload, ("backbone", "hidden_dim"), f"model section in {train_config_path}")
    _require_keys(
        train_section,
        ("batch_size", "epochs", "learning_rate", "lambda_cons", "lambda_neg"),
        f"train section in {train_config_path}",
    )
    feature_mode = str(data_payload["feature_mode"])
    if "feature_mode" in model_payload and str(model_payload["feature_mode"]) != feature_mode:
        raise ConfigError(
            "feature_mode mismatch between data config and train config: "
            f"{feature_mode!r} != {model_payload['feature_mode']!r}"
        )
    backbone_name = str(model_payload["backbone"])
    if backbone_name == "kernel_operator" and "num_layers" not in model_payload:
        raise ConfigError("kernel_operator requires model.num_layers in the train config")
    beta_range = data_payload["beta_range"]
    if not isinstance(beta_range, (list, tuple)) or len(beta_range) != 2:
        raise ConfigError("data config field 'beta_range' must contain exactly two numbers")
    patch_types = data_payload["patch_types"]
    if not isinstance(patch_types, list) or not patch_types:
        raise ConfigError("data config field 'patch_types' must be a non-empty list")
    feature_dim = feature_dim_for_mode(feature_mode)
    seed = int(seed_override if seed_override is not None else data_payload["seed"])
    return ResolvedTrainConfig(
        data_path=data_config_path,
        train_path=train_config_path,
        data=dict(data_payload),
        model=model_payload,
        train=train_section,
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
        },
        "resolved": {
            "seed": resolved.seed,
            "feature_mode": resolved.feature_mode,
            "feature_dim": resolved.feature_dim,
            "backbone": resolved.backbone_name,
            "k_neighbors": resolved.k_neighbors,
            "prior_type": "gaussian",
            "train_seed": resolved.seed,
            "val_seed": resolved.seed + 1,
            "device": device,
            "run_name": run_name,
            "repo_root": str(repo_root),
        },
    }
