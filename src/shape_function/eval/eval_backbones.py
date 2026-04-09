from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from shape_function.train.metrics import compute_batch_metrics


def evaluate_model(model, loader: DataLoader, device: str = "cpu") -> dict[str, Any]:
    model.eval()
    summary: dict[str, float] = {}
    count = 0
    for batch in loader:
        outputs = model(
            batch["X"].to(device),
            batch["x_q"].to(device),
            batch["beta"].to(device),
            rho_q=batch["rho_q"].to(device) if batch["rho_q"].shape[-1] > 0 else None,
        )
        metrics = compute_batch_metrics(outputs, batch["phi_ref"].to(device))
        for key, value in metrics.items():
            summary[key] = summary.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in summary.items()}
