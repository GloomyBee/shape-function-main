from __future__ import annotations

import torch


def quartic_spline_window(s: torch.Tensor) -> torch.Tensor:
    inside = s < 1.0
    out = torch.zeros_like(s)
    out[inside] = (1.0 - s[inside] * s[inside]) ** 2
    return out


def wendland_c2_window(s: torch.Tensor) -> torch.Tensor:
    inside = s < 1.0
    out = torch.zeros_like(s)
    t = 1.0 - s[inside]
    out[inside] = t.pow(4) * (4.0 * s[inside] + 1.0)
    return out
