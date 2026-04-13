from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from shape_function.train.losses import compute_losses
from shape_function.train.metrics import compute_batch_metrics
from shape_function.utils.artifacts import ensure_run_artifacts, save_npz


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _format_epoch_progress(
    *,
    epoch: int,
    epochs: int,
    elapsed_seconds: float,
    lr: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    is_best: bool,
) -> str:
    suffix = " *best" if is_best else ""
    return (
        f"[epoch {epoch}/{epochs}] "
        f"elapsed={elapsed_seconds:.2f}s "
        f"lr={lr:.6e} "
        f"train_total={train_metrics['total']:.6e} "
        f"val_total={val_metrics['total']:.6e} "
        f"val_rel_l2={val_metrics['relative_l2']:.6e}"
        f"{suffix}"
    )


def _move_batch_tensor(batch: dict[str, torch.Tensor], key: str, device: str) -> torch.Tensor:
    return batch[key].to(device, non_blocking=(device == "cuda"))


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
        X = _move_batch_tensor(batch, "X", device)
        x_q = _move_batch_tensor(batch, "x_q", device)
        beta = _move_batch_tensor(batch, "beta", device)
        rho_q = _move_batch_tensor(batch, "rho_q", device)
        phi_ref = _move_batch_tensor(batch, "phi_ref", device)
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
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["lr"].append(current_lr)
        train_metrics = _epoch_pass(model, train_loader, optimizer, device, lambda_cons, lambda_neg)
        val_metrics = _epoch_pass(model, val_loader, None, device, lambda_cons, lambda_neg)
        history["epoch"].append(float(epoch))
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key, value in metrics.items():
                history.setdefault(f"{prefix}_{key}", []).append(float(value))
        is_best = False
        if val_metrics["total"] < best_val:
            best_val = float(val_metrics["total"])
            best_epoch = epoch
            best_state_dict = _clone_state_dict(model)
            is_best = True
        print(
            _format_epoch_progress(
                epoch=epoch,
                epochs=epochs,
                elapsed_seconds=time.perf_counter() - epoch_start,
                lr=current_lr,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=is_best,
            )
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
    curves = {key: np.asarray(values, dtype=np.float64) for key, values in history.items()}
    save_npz(artifacts.root_dir / "curves.npz", curves)
    return {"history": history, "metrics": metrics_payload, "artifacts": artifacts}
