from __future__ import annotations

from typing import Any

import numpy as np
import torch

from shape_function.models.heads.structure_head import StructurePreservingHead


def run_head_sanity(num_samples: int = 1000, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    head = StructurePreservingHead()
    X = rng.uniform(-1.0, 1.0, size=(num_samples, 16, 2))
    x_q = np.zeros((num_samples, 2), dtype=np.float64)
    logits = rng.normal(size=(num_samples, 16))
    distances = np.linalg.norm(X - x_q[:, None, :], axis=-1)
    r_max = distances.max(axis=-1, keepdims=True)
    with torch.no_grad():
        _, _, aux = head(
            torch.as_tensor(logits, dtype=torch.float64),
            torch.as_tensor(X, dtype=torch.float64),
            torch.as_tensor(x_q, dtype=torch.float64),
            torch.as_tensor(r_max, dtype=torch.float64),
        )
    return {
        "pou_residual": float(torch.mean(torch.abs(aux["sum_phi"] - 1.0))),
        "linear_residual": float(torch.mean(aux["linear_residual"])),
        "negative_fraction": float(torch.mean(aux["neg_fraction"])),
        "mean_cond_M": float(torch.mean(aux["cond_M"])),
    }
