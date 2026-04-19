from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, patches: list[dict[str, Any]]):
        self.items = [self._to_tensor_patch(patch) for patch in patches]

    @staticmethod
    def _to_tensor_patch(patch: dict[str, Any]) -> dict[str, torch.Tensor | str]:
        item: dict[str, torch.Tensor | str] = {
            "X": torch.as_tensor(np.asarray(patch["X"]), dtype=torch.float64),
            "x_q": torch.as_tensor(np.asarray(patch["x_q"]), dtype=torch.float64),
            "beta": torch.as_tensor([patch["beta"]], dtype=torch.float64),
            "rho_q": torch.as_tensor(
                np.asarray(patch.get("rho_q", np.zeros((0,), dtype=np.float64))),
                dtype=torch.float64,
            ),
            "r_max": torch.as_tensor([patch.get("r_max", 0.0)], dtype=torch.float64),
            "patch_type": str(patch.get("patch_type", "unknown")),
        }
        if "phi_ref" in patch:
            item["phi_ref"] = torch.as_tensor(np.asarray(patch["phi_ref"]), dtype=torch.float64)
        return item

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return self.items[index]
