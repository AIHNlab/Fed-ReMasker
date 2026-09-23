from __future__ import annotations
import torch
import torch.nn as nn
from FedHF.gnn import GraphSAGEWeighted


class FedHFImputer(nn.Module):
    """
    Graph-based federated imputer.

    Each feature is a node in a global correlation graph. The model learns
    to impute missing values by propagating information along graph edges
    (correlated features) via weighted GraphSAGE layers.

    Adapted from FedHFImpute (Hocine et al. 2025).
    """

    def __init__(self, n_features: int, d_model: int = 64,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_features = n_features

        self.feature_emb = nn.Embedding(n_features, d_model)
        self.in_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.gnn = nn.ModuleList([
            GraphSAGEWeighted(d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.out_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_val: torch.Tensor, obs_mask: torch.Tensor,
                avail_mask: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        x_val     : (B, F) feature values, NaN replaced with 0
        obs_mask  : (B, F) 1 = observed, 0 = missing
        avail_mask: (B, F) 1 = feature in schema (all 1s in our benchmark)
        edge_index: (2, E)
        edge_weight: (E,)
        Returns   : (B, F) imputed values for all features
        """
        B, F = x_val.shape
        token = torch.stack([x_val, obs_mask, avail_mask], dim=-1)  # (B, F, 3)
        h = self.in_proj(token)                                       # (B, F, d)

        feat_ids = torch.arange(F, device=x_val.device).unsqueeze(0).expand(B, F)
        h = h + self.feature_emb(feat_ids)

        for layer in self.gnn:
            h = h + layer(h, edge_index=edge_index,
                          edge_weight=edge_weight, node_mask=avail_mask)

        return self.out_head(h).squeeze(-1)  # (B, F)
