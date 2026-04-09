from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int = 10
    best_value: float = float("inf")
    bad_epochs: int = 0

    def update(self, value: float) -> bool:
        if value < self.best_value:
            self.best_value = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
