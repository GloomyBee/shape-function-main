from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, patches: list[dict[str, Any]]):
        self.patches = patches

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        patch = self.patches[index]
        return {
            "X": torch.as_tensor(np.asarray(patch["X"]), dtype=torch.float64),
            "x_q": torch.as_tensor(np.asarray(patch["x_q"]), dtype=torch.float64),
            "beta": torch.as_tensor([patch["beta"]], dtype=torch.float64),
            "rho_q": torch.as_tensor(np.asarray(patch.get("rho_q", np.zeros((0,), dtype=np.float64))), dtype=torch.float64),
            "phi_ref": torch.as_tensor(np.asarray(patch["phi_ref"]), dtype=torch.float64),
        }
