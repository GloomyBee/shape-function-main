from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from shape_function.train.metrics import compute_batch_metrics


def evaluate_model(
    model,
    loader: DataLoader,
    device: str = "cpu",
    *,
    compute_teacher_metrics: bool = False,
) -> dict[str, Any]:
    model.eval()
    summary: dict[str, float] = {}
    count = 0
    for batch in loader:
        X = batch["X"].to(device)
        x_q = batch["x_q"].to(device)
        beta = batch["beta"].to(device)
        r_max = batch["r_max"].to(device)
        outputs = model(
            X,
            x_q,
            beta,
            rho_q=batch["rho_q"].to(device) if batch["rho_q"].shape[-1] > 0 else None,
        )
        phi_ref = batch["phi_ref"].to(device) if "phi_ref" in batch else None
        metrics = compute_batch_metrics(
            outputs,
            phi_ref,
            X=X,
            x_q=x_q,
            r_max=r_max,
            patch_type=batch.get("patch_type"),
            compute_teacher_metrics=compute_teacher_metrics and phi_ref is not None,
        )
        for key, value in metrics.items():
            summary[key] = summary.get(key, 0.0) + value
        count += 1
    return {key: value / max(count, 1) for key, value in summary.items()}
