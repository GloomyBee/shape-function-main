from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from shape_function.train.losses import compute_losses
from shape_function.train.metrics import compute_batch_metrics
from shape_function.utils.artifacts import ensure_run_artifacts, save_json, save_npz, save_summary


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
    history: dict[str, list[float]] = {"epoch": []}
    artifacts = ensure_run_artifacts(repo_root, run_name)
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        train_metrics = _epoch_pass(model, train_loader, optimizer, device, lambda_cons, lambda_neg)
        val_metrics = _epoch_pass(model, val_loader, None, device, lambda_cons, lambda_neg)
        history["epoch"].append(float(epoch))
        for prefix, metrics in (("train", train_metrics), ("val", val_metrics)):
            for key, value in metrics.items():
                history.setdefault(f"{prefix}_{key}", []).append(float(value))
        best_val = min(best_val, val_metrics["total"])
    metrics_payload = {
        "best_val_total": best_val,
        "final_train_total": history["train_total"][-1],
        "final_val_total": history["val_total"][-1],
    }
    summary_lines = [
        f"best_val_total: {best_val:.6e}",
        f"final_train_total: {history['train_total'][-1]:.6e}",
        f"final_val_total: {history['val_total'][-1]:.6e}",
    ]
    curves = {key: np.asarray(values, dtype=np.float64) for key, values in history.items()}
    save_json(artifacts.root_dir / "metrics.json", metrics_payload)
    save_summary(artifacts.root_dir / "summary.txt", summary_lines)
    save_npz(artifacts.root_dir / "curves.npz", curves)
    return {"history": history, "metrics": metrics_payload, "artifacts": artifacts}
