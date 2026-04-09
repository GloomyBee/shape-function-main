from __future__ import annotations

import torch
import torch.nn as nn

from shape_function.models.backbones.kernel_operator import KernelOperatorBackbone
from shape_function.models.backbones.mlp_baseline import MLPBaselineBackbone
from shape_function.models.heads.structure_head import StructurePreservingHead
from shape_function.utils.geometry import rel_coords_and_radius_torch


def build_node_features_torch(rel_hat: torch.Tensor, beta: torch.Tensor, rho_q: torch.Tensor | None = None) -> torch.Tensor:
    dist_hat = torch.linalg.norm(rel_hat, dim=-1, keepdim=True)
    beta_feat = beta[:, None, :].expand(-1, rel_hat.shape[1], -1)
    features = [rel_hat, dist_hat, beta_feat]
    if rho_q is not None and rho_q.numel() > 0:
        repeated = rho_q[:, None, :].expand(-1, rel_hat.shape[1], -1)
        features.append(repeated)
    return torch.cat(features, dim=-1)


class ShapeFunctionModel(nn.Module):
    def __init__(self, backbone_name: str = "kernel_operator", feature_dim: int = 4):
        super().__init__()
        if backbone_name == "kernel_operator":
            self.backbone = KernelOperatorBackbone(input_dim=feature_dim)
        elif backbone_name == "mlp_baseline":
            self.backbone = MLPBaselineBackbone(input_dim=feature_dim)
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")
        self.head = StructurePreservingHead()
        self.double()

    def forward(
        self,
        X: torch.Tensor,
        x_q: torch.Tensor,
        beta: torch.Tensor,
        rho_q: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, rel_hat, r_max = rel_coords_and_radius_torch(X, x_q)
        node_features = build_node_features_torch(rel_hat, beta, rho_q=rho_q)
        logits = self.backbone(node_features, rel_hat, mask=mask)
        phi_base, phi_corr, aux = self.head(logits, X, x_q, r_max, mask=mask)
        return {
            "logits": logits,
            "phi_base": phi_base,
            "phi_corr": phi_corr,
            "aux": aux,
            "r_max": r_max,
            "node_features": node_features,
        }
