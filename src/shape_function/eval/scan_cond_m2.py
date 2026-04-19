from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shape_function.data.patch_sampler import PATCH_TYPES, sample_patches
from shape_function.models.heads.reproducing_correction import apply_reproducing_correction


def scan_cond_m2(
    *,
    num_patches: int,
    seed: int,
    k_neighbors: int,
    beta_range: tuple[float, float],
    kappa_max: float,
    patch_types: tuple[str, ...] = PATCH_TYPES,
) -> dict[str, Any]:
    patches = sample_patches(
        num_patches,
        seed=seed,
        patch_types=patch_types,
        k_neighbors=k_neighbors,
        beta_range=beta_range,
    )
    grouped: dict[str, list[float]] = {name: [] for name in patch_types}
    grouped_fallback: dict[str, int] = {name: 0 for name in patch_types}
    for patch in patches:
        X = torch.as_tensor(patch["X"][None, ...], dtype=torch.float64)
        x_q = torch.as_tensor(patch["x_q"][None, ...], dtype=torch.float64)
        r_max = torch.as_tensor([[patch["r_max"]]], dtype=torch.float64)
        phi_base = torch.full((1, X.shape[1]), 1.0 / X.shape[1], dtype=torch.float64)
        result = apply_reproducing_correction(
            phi_base,
            X,
            x_q,
            r_max,
            basis_order=2,
            kappa_max=kappa_max,
            fallback_mode="hard",
        )
        cond_m2 = float(result["cond_M2"].item())
        grouped[patch["patch_type"]].append(cond_m2)
        grouped_fallback[patch["patch_type"]] += int(result["fallback_mask"].item())
    all_values = np.asarray([value for values in grouped.values() for value in values], dtype=np.float64)
    by_type = {}
    for patch_type, values in grouped.items():
        arr = np.asarray(values, dtype=np.float64)
        by_type[patch_type] = {
            "count": int(arr.size),
            "mean_cond_M2": float(arr.mean()) if arr.size else float("nan"),
            "p95_cond_M2": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
            "worst_cond_M2": float(arr.max()) if arr.size else float("nan"),
            "fallback_rate": float(grouped_fallback[patch_type] / max(arr.size, 1)),
        }
    return {
        "num_patches": num_patches,
        "seed": seed,
        "k_neighbors": k_neighbors,
        "beta_range": list(beta_range),
        "kappa_max": kappa_max,
        "overall": {
            "mean_cond_M2": float(all_values.mean()) if all_values.size else float("nan"),
            "p95_cond_M2": float(np.quantile(all_values, 0.95)) if all_values.size else float("nan"),
            "worst_cond_M2": float(all_values.max()) if all_values.size else float("nan"),
            "fallback_rate": float(sum(grouped_fallback.values()) / max(all_values.size, 1)),
        },
        "by_patch_type": by_type,
        "suggested_kappa_max": float(np.quantile(all_values, 0.95)) if all_values.size else float(kappa_max),
    }


def save_scan_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
