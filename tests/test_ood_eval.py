from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from shape_function.data.patch_sampler import sample_patch
from shape_function.eval.ood_eval import (
    OODProtocol,
    build_protocol_dataset,
    expand_protocols,
    rotate_patch_about_query,
    run_ood_eval,
)


def _snapshot(tmp_path: Path, *, backbone: str = "kernel_operator", k_neighbors: int = 16) -> dict:
    return {
        "data": {
            "seed": 3,
            "num_train": 8,
            "num_val": 4,
            "k_neighbors": k_neighbors,
            "beta_range": [0.5, 0.5],
            "feature_mode": "minimal",
            "patch_types": ["uniform", "boundary_truncated"],
        },
        "train": {
            "model": {"backbone": backbone, "hidden_dim": 8, "num_layers": 1, "feature_mode": "minimal"},
            "train": {"batch_size": 4, "epochs": 1, "learning_rate": 0.001, "prior_type": "gaussian"},
            "head": {"basis_order": 2, "kappa_max": 1.0e8, "fallback_mode": "hard"},
            "loss": {"mode": "unsupervised_v1", "lambda_base_lin": 1.0, "lambda_ent": 0.01, "lambda_neg": 1.0e-5},
            "eval": {"compute_teacher_metrics": False},
        },
        "resolved": {
            "seed": 3,
            "feature_mode": "minimal",
            "feature_dim": 4,
            "backbone": backbone,
            "k_neighbors": k_neighbors,
            "prior_type": "gaussian",
            "device": "cpu",
            "loss_mode": "unsupervised_v1",
            "basis_order": 2,
        },
    }


def _tiny_ood_config(tmp_path: Path) -> Path:
    path = tmp_path / "ood_eval.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 22,
                "defaults": {"num_patches": 2, "batch_size": 2, "compute_teacher_metrics": False},
                "protocols": {
                    "iid": {"enabled": True},
                    "loto": {"enabled": False},
                    "rotation": {"enabled": True, "angles": [90]},
                    "k_ood": {"enabled": False, "k_values": [12]},
                    "scale": {"enabled": True, "beta_ranges": [[8.0, 16.0]]},
                    "boundary": {"enabled": True},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _command_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath_entries = [str(repo_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    return env


def test_expand_protocols_has_stable_names() -> None:
    protocols = expand_protocols(
        {
            "protocols": {
                "iid": {"enabled": True},
                "loto": {"enabled": True, "held_out": ["uniform", "clustered"]},
                "rotation": {"enabled": True, "angles": [45, 90]},
                "k_ood": {"enabled": True, "k_values": [12, 24]},
                "scale": {"enabled": True, "beta_ranges": [[8.0, 16.0]]},
                "boundary": {"enabled": True},
            }
        }
    )
    assert [item.name for item in protocols] == [
        "iid",
        "loto_uniform",
        "loto_clustered",
        "rotation_45",
        "rotation_90",
        "k_12",
        "k_24",
        "scale_beta_8p0_16p0",
        "boundary",
    ]


def test_iid_and_boundary_protocols_build_dataset() -> None:
    snapshot = _snapshot(Path("."))
    for protocol in (
        OODProtocol("iid", "iid", {"num_patches": 2, "seed": 1}),
        OODProtocol("boundary", "boundary", {"num_patches": 2, "seed": 2}),
    ):
        dataset, params = build_protocol_dataset(protocol, snapshot, compute_teacher_metrics=False)
        assert len(dataset) == 2
        assert params["num_built"] == 2
        if protocol.kind == "boundary":
            assert {item["patch_type"] for item in dataset} == {"boundary_truncated"}


def test_rotation_recomputes_rmax_without_shape_change() -> None:
    import numpy as np

    patch = sample_patch(np.random.default_rng(7), "uniform", k_neighbors=16, beta_range=(0.5, 0.5))
    rotated = rotate_patch_about_query(patch, 90)
    assert rotated["X"].shape == patch["X"].shape
    assert abs(rotated["r_max"] - patch["r_max"]) < 1.0e-12


def test_scale_protocol_uses_beta_extrapolation_without_geometry_scaling() -> None:
    snapshot = _snapshot(Path("."))
    protocol = OODProtocol("scale_beta_8p0_16p0", "scale", {"num_patches": 2, "seed": 5, "beta_range": [8.0, 16.0]})
    dataset, params = build_protocol_dataset(protocol, snapshot, compute_teacher_metrics=False)
    assert params["scale_semantics"] == "beta_range_extrapolation"
    assert params["beta_range"] == [8.0, 16.0]
    assert all(8.0 <= item["beta"] <= 16.0 for item in dataset)


def test_loto_protocol_uses_held_out_patch_type() -> None:
    snapshot = _snapshot(Path("."))
    protocol = OODProtocol("loto_clustered", "loto", {"num_patches": 2, "seed": 6, "held_out": "clustered", "patch_types": ["clustered"]})
    dataset, params = build_protocol_dataset(protocol, snapshot, compute_teacher_metrics=False)
    assert params["held_out"] == "clustered"
    assert {item["patch_type"] for item in dataset} == {"clustered"}


def test_run_ood_eval_records_mlp_k_mismatch_as_skipped(tmp_path: Path) -> None:
    from shape_function.models.full_model import build_shape_function_model

    run_dir = tmp_path / "mlp_run"
    run_dir.mkdir()
    snapshot = _snapshot(tmp_path, backbone="mlp_baseline", k_neighbors=16)
    (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8")
    model = build_shape_function_model(
        backbone_name="mlp_baseline",
        feature_mode="minimal",
        hidden_dim=8,
        k_neighbors=16,
        head_kwargs=snapshot["train"]["head"],
    )
    torch.save({"model_state_dict": model.state_dict()}, run_dir / "checkpoint.pt")
    ood_path = tmp_path / "ood.yaml"
    ood_path.write_text(
        yaml.safe_dump(
            {
                "defaults": {"num_patches": 2, "batch_size": 2, "compute_teacher_metrics": False},
                "protocols": {"k_ood": {"enabled": True, "k_values": [12]}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = run_ood_eval(run_dir=run_dir, ood_config_path=ood_path, device="cpu", eval_name="case")
    payload = run_ood_eval(run_dir=run_dir, ood_config_path=ood_path, device="cpu", eval_name="case")
    assert "k_12" in payload["skipped"]
    metrics_path = run_dir / "ood_eval" / "case" / "ood_eval_metrics.json"
    assert metrics_path.is_file()
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "results" in saved
    assert "skipped" in saved


def test_ood_eval_cli_and_module_entrypoints(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    from tests.test_cli_train import _write_small_configs

    data_path, train_path = _write_small_configs(tmp_path, legacy=False)
    run_name = "ood_eval_cli_case"
    run_dir = repo_root / "runs" / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    train_command = [
        sys.executable,
        "-m",
        "shape_function.cli",
        "train",
        "--data-config",
        str(data_path),
        "--train-config",
        str(train_path),
        "--run-name",
        run_name,
        "--device",
        "cpu",
    ]
    train_completed = subprocess.run(
        train_command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=_command_env(repo_root),
    )
    assert train_completed.returncode == 0, train_completed.stderr
    ood_config = _tiny_ood_config(tmp_path)
    cli_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "shape_function.cli",
            "ood-eval",
            "--run-dir",
            str(run_dir),
            "--ood-config",
            str(ood_config),
            "--device",
            "cpu",
            "--eval-name",
            "cli",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=_command_env(repo_root),
    )
    assert cli_completed.returncode == 0, cli_completed.stderr
    cli_metrics = run_dir / "ood_eval" / "cli" / "ood_eval_metrics.json"
    assert cli_metrics.is_file()
    payload = json.loads(cli_metrics.read_text(encoding="utf-8"))
    assert "iid" in payload["results"]
    assert "boundary" in payload["results"]
    module_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "shape_function.eval.ood_eval",
            "--run-dir",
            str(run_dir),
            "--ood-config",
            str(ood_config),
            "--device",
            "cpu",
            "--eval-name",
            "module",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=_command_env(repo_root),
    )
    assert module_completed.returncode == 0, module_completed.stderr
    assert (run_dir / "ood_eval" / "module" / "ood_eval_metrics.json").is_file()
    shutil.rmtree(run_dir)
