from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _write_small_configs(tmp_path: Path) -> tuple[Path, Path]:
    data_path = tmp_path / "data.yaml"
    train_path = tmp_path / "train.yaml"
    data_path.write_text(
        "\n".join(
            [
                "seed: 9",
                "num_train: 8",
                "num_val: 4",
                "k_neighbors: 16",
                "beta_range: [0.5, 0.5]",
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
                "  hidden_dim: 8",
                "  num_layers: 1",
                "  feature_mode: minimal",
                "train:",
                "  batch_size: 4",
                "  epochs: 1",
                "  learning_rate: 0.001",
                "  lambda_cons: 0.0001",
                "  lambda_neg: 0.00001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return data_path, train_path


def _command_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath_entries = [str(repo_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])
    return env


def test_python_module_train_cli_creates_expected_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path, train_path = _write_small_configs(tmp_path)
    run_name = "cli_module_case"
    run_dir = repo_root / "runs" / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    command = [
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
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=_command_env(repo_root),
    )
    assert completed.returncode == 0, completed.stderr
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "summary.txt").is_file()
    assert (run_dir / "curves.npz").is_file()
    assert (run_dir / "checkpoint.pt").is_file()
    assert (run_dir / "best_model.pt").is_file()
    assert (run_dir / "config_snapshot.yaml").is_file()
    assert (run_dir / "eval_metrics.json").is_file()
    assert f"run_name: {run_name}" in completed.stdout
    assert "device: cpu" in completed.stdout
    assert "training complete" in completed.stdout.lower()
    shutil.rmtree(run_dir)


def test_console_script_train_cli_runs_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path, train_path = _write_small_configs(tmp_path)
    run_name = "cli_script_case"
    run_dir = repo_root / "runs" / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    script_name = "shape-function.exe" if os.name == "nt" else "shape-function"
    script_path = Path(sys.executable).parent / script_name
    assert script_path.exists(), "console script is missing; reinstall the package in editable mode"
    completed = subprocess.run(
        [
            str(script_path),
            "train",
            "--data-config",
            str(data_path),
            "--train-config",
            str(train_path),
            "--run-name",
            run_name,
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=_command_env(repo_root),
    )
    assert completed.returncode == 0, completed.stderr
    assert (run_dir / "checkpoint.pt").is_file()
    assert (run_dir / "best_model.pt").is_file()
    shutil.rmtree(run_dir)


def test_train_cli_fails_when_run_directory_exists(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_path, train_path = _write_small_configs(tmp_path)
    run_name = "cli_existing_case"
    run_dir = repo_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
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
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env=_command_env(repo_root),
    )
    assert completed.returncode != 0
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    assert run_name in combined_output
    assert "already exists" in combined_output.lower()
    shutil.rmtree(run_dir)
