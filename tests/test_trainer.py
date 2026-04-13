from __future__ import annotations

from pathlib import Path

import torch
import warnings

from shape_function.train import trainer as trainer_module


class _ScalarModel(torch.nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float64))

    def forward(self, *_args, **_kwargs):  # pragma: no cover - not used in patched epoch pass
        return {}


def test_train_model_saves_best_checkpoint_and_restores_best_weights(tmp_path: Path, monkeypatch) -> None:
    model = _ScalarModel()
    repo_root = tmp_path
    train_loader = []
    val_loader = []
    states = [
        ("train", 2.0, 0.2),
        ("val", 1.0, 0.1),
        ("train", 5.0, 0.5),
        ("val", 3.0, 0.3),
    ]
    call_index = {"value": 0}

    def fake_epoch_pass(*_args, **_kwargs):
        phase, total, marker = states[call_index["value"]]
        call_index["value"] += 1
        model.weight.data.fill_(marker)
        return {"total": total, "relative_l2": total}

    monkeypatch.setattr(trainer_module, "_epoch_pass", fake_epoch_pass)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
        result = trainer_module.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            repo_root=repo_root,
            run_name="trainer_case",
            epochs=2,
            learning_rate=1.0e-3,
            device="cpu",
        )
    curves_path = result["artifacts"].root_dir / "curves.npz"
    assert curves_path.is_file()
    assert not (result["artifacts"].root_dir / "best_model.pt").exists()
    assert not (result["artifacts"].root_dir / "metrics.json").exists()
    assert not (result["artifacts"].root_dir / "summary.txt").exists()
    assert abs(float(model.weight.item()) - 0.1) < 1.0e-12
    assert result["metrics"]["best_epoch"] == 1


def test_train_model_records_learning_rate_curve(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1, dtype=torch.float64)
    sample = {
        "X": torch.zeros((1, 2, 2), dtype=torch.float64),
        "x_q": torch.zeros((1, 2), dtype=torch.float64),
        "beta": torch.ones((1, 1), dtype=torch.float64),
        "rho_q": torch.zeros((1, 0), dtype=torch.float64),
        "phi_ref": torch.ones((1, 2), dtype=torch.float64) / 2.0,
    }

    def fake_epoch_pass(*_args, **_kwargs):
        return {"total": 1.0, "relative_l2": 1.0}

    original_epoch_pass = trainer_module._epoch_pass
    trainer_module._epoch_pass = fake_epoch_pass
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
            result = trainer_module.train_model(
                model=model,
                train_loader=[sample],
                val_loader=[sample],
                repo_root=tmp_path,
                run_name="scheduler_case",
                epochs=3,
                learning_rate=1.0e-3,
                device="cpu",
            )
    finally:
        trainer_module._epoch_pass = original_epoch_pass
    assert "lr" in result["history"]
    assert len(result["history"]["lr"]) == 3
    assert result["history"]["lr"][0] > result["history"]["lr"][-1]


def test_train_model_prints_epoch_progress(tmp_path: Path, capsys, monkeypatch) -> None:
    model = _ScalarModel()

    def fake_epoch_pass(*_args, **_kwargs):
        return {"total": 1.0, "relative_l2": 0.25}

    monkeypatch.setattr(trainer_module, "_epoch_pass", fake_epoch_pass)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
        trainer_module.train_model(
            model=model,
            train_loader=[],
            val_loader=[],
            repo_root=tmp_path,
            run_name="progress_case",
            epochs=2,
            learning_rate=1.0e-3,
            device="cpu",
        )
    captured = capsys.readouterr()
    assert "[epoch 1/2]" in captured.out
    assert "elapsed=" in captured.out
    assert "train_total=1.000000e+00" in captured.out
    assert "val_total=1.000000e+00" in captured.out
    assert "val_rel_l2=2.500000e-01" in captured.out
