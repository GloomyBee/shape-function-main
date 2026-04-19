from __future__ import annotations

from pathlib import Path

import pytest

from shape_function.cli.config import ConfigError, resolve_train_config


def test_resolve_train_config_reads_repository_configs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    resolved = resolve_train_config(
        repo_root / "configs" / "data.yaml",
        repo_root / "configs" / "train_kernel_operator.yaml",
    )
    assert resolved.seed == 42
    assert resolved.feature_mode == "minimal"
    assert resolved.feature_dim == 4
    assert resolved.backbone_name == "kernel_operator"
    assert resolved.k_neighbors == 16
    assert resolved.train["epochs"] == 10
    assert resolved.loss["mode"] == "unsupervised_v1"
    assert resolved.head["basis_order"] == 2


def test_resolve_train_config_rejects_feature_mode_mismatch(tmp_path: Path) -> None:
    data_path = tmp_path / "data.yaml"
    train_path = tmp_path / "train.yaml"
    data_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "num_train: 4",
                "num_val: 2",
                "k_neighbors: 16",
                "beta_range: [0.5, 8.0]",
                "feature_mode: minimal",
                "patch_types:",
                "  - uniform",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_path.write_text(
        "\n".join(
            [
                "model:",
                "  backbone: kernel_operator",
                "  hidden_dim: 16",
                "  num_layers: 2",
                "  feature_mode: enhanced",
                "train:",
                "  batch_size: 2",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  prior_type: gaussian",
                "head:",
                "  basis_order: 2",
                "  kappa_max: 100000000.0",
                "  fallback_mode: hard",
                "loss:",
                "  mode: unsupervised_v1",
                "  lambda_base_lin: 1.0",
                "  lambda_ent: 0.01",
                "  lambda_neg: 0.00001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="feature_mode"):
        resolve_train_config(data_path, train_path)


def test_resolve_train_config_seed_override_wins(tmp_path: Path) -> None:
    data_path = tmp_path / "data.yaml"
    train_path = tmp_path / "train.yaml"
    data_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "num_train: 4",
                "num_val: 2",
                "k_neighbors: 16",
                "beta_range: [0.5, 8.0]",
                "feature_mode: enhanced",
                "patch_types:",
                "  - uniform",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_path.write_text(
        "\n".join(
            [
                "model:",
                "  backbone: mlp_baseline",
                "  hidden_dim: 32",
                "  feature_mode: enhanced",
                "train:",
                "  batch_size: 2",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  prior_type: gaussian",
                "head:",
                "  basis_order: 1",
                "  fallback_mode: hard",
                "loss:",
                "  mode: unsupervised_v1",
                "  lambda_base_lin: 1.0",
                "  lambda_ent: 0.01",
                "  lambda_neg: 0.00001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    resolved = resolve_train_config(data_path, train_path, seed_override=123)
    assert resolved.seed == 123
    assert resolved.feature_dim == 7


def test_resolve_train_config_reads_production_data_config() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    resolved = resolve_train_config(
        repo_root / "configs" / "data_production.yaml",
        repo_root / "configs" / "train_kernel_operator.yaml",
    )
    assert resolved.data["num_train"] == 20000
    assert resolved.data["num_val"] == 2000
    assert len(resolved.data["patch_types"]) == 7
    assert "anisotropic" in resolved.data["patch_types"]
    assert "sparse_dense_transition" in resolved.data["patch_types"]


def test_resolve_train_config_requires_kappa_max_for_second_order(tmp_path: Path) -> None:
    data_path = tmp_path / "data.yaml"
    train_path = tmp_path / "train.yaml"
    data_path.write_text(
        "seed: 1\nnum_train: 2\nnum_val: 1\nk_neighbors: 16\nbeta_range: [0.5, 1.0]\nfeature_mode: minimal\npatch_types:\n  - uniform\n",
        encoding="utf-8",
    )
    train_path.write_text(
        "model:\n  backbone: mlp_baseline\n  hidden_dim: 8\ntrain:\n  batch_size: 1\n  epochs: 1\n  learning_rate: 0.001\n  prior_type: gaussian\nhead:\n  basis_order: 2\n  fallback_mode: hard\nloss:\n  mode: unsupervised_v1\n  lambda_base_lin: 1.0\n  lambda_ent: 0.01\n  lambda_neg: 0.00001\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="kappa_max"):
        resolve_train_config(data_path, train_path)
