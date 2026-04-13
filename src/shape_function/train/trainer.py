from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from shape_function.train.losses import compute_losses
from shape_function.train.metrics import compute_batch_metrics
from shape_function.utils.artifacts import ensure_run_artifacts, save_json, save_npz, save_summary


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _save_best_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    best_epoch: int,
    best_val_total: float,
) -> None:
    torch.save(
        {
            "model_state_dict": _clone_state_dict(model),
            "best_epoch": int(best_epoch),
            "best_val_total": float(best_val_total),
        },
        path,
    )


def _epoch_pass(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    lambda_cons: float,
    lambda_neg: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        X = batch["X"].to(device)
        x_q = batch["x_q"].to(device)
        beta = batch["beta"].to(device)
        rho_q = batch["rho_q"].to(device)
        phi_ref = batch["phi_ref"].to(device)
        outputs = model(X, x_q, beta, rho_q=rho_q if rho_q.shape[-1] > 0 else None)
        losses = compute_losses(outputs, phi_ref, lambda_cons=lambda_cons, lambda_neg=lambda_neg)
        metrics = compute_batch_metrics(outputs, phi_ref)
        if train_mode:
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
        merged = {key: float(value.detach().cpu()) for key, value in losses.items()}
        merged.update(metrics)
        for key, value in merged.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    repo_root: Path,
    run_name: str,
    epochs: int = 10,
    learning_rate: float = 1.0e-3,
    lambda_cons: float = 1.0e-4,
    lambda_neg: float = 1.0e-5,
    device: str = "cpu",
) -> dict[str, Any]:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(int(epochs), 1),
        eta_min=1.0e-5,
    )
    history: dict[str, list[float]] = {"epoch": [], "lr": []}
    artifacts = ensure_run_artifacts(repo_root, run_name)
    best_val = float("inf")
    best_epoch = 0
    best_state_dict = _clone_state_dict(model)
    best_checkpoint_path = artifacts.root_dir / "best_model.pt"
    for epoch in range(1, epochs + 1):
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        train_metrics = _epoch_pass(model, train_loader, optimizer, device, lambda_cons, lambda_neg)
        val_metrics = _epoch_pass(model, val_loader, None, device, lambda_cons, lambda_neg)
        history["epoch"].append(float(epoch))
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key, value in metrics.items():
                history.setdefault(f"{prefix}_{key}", []).append(float(value))
        if val_metrics["total"] < best_val:
            best_val = float(val_metrics["total"])
            best_epoch = epoch
            best_state_dict = _clone_state_dict(model)
            _save_best_checkpoint(
                best_checkpoint_path,
                model,
                best_epoch=best_epoch,
                best_val_total=best_val,
            )
        scheduler.step()
    model.load_state_dict(best_state_dict)
    metrics_payload = {
        "best_val_total": best_val,
        "best_epoch": best_epoch,
        "final_train_total": history["train_total"][-1],
        "final_val_total": history["val_total"][-1],
        "final_lr": history["lr"][-1],
    }
    summary_lines = [
        f"best_val_total: {best_val:.6e}",
        f"best_epoch: {best_epoch}",
        f"final_train_total: {history['train_total'][-1]:.6e}",
        f"final_val_total: {history['val_total'][-1]:.6e}",
        f"final_lr: {history['lr'][-1]:.6e}",
    ]
    curves = {key: np.asarray(values, dtype=np.float64) for key, values in history.items()}
    save_json(artifacts.root_dir / "metrics.json", metrics_payload)
    save_summary(artifacts.root_dir / "summary.txt", summary_lines)
    save_npz(artifacts.root_dir / "curves.npz", curves)
    return {"history": history, "metrics": metrics_payload, "artifacts": artifacts}
