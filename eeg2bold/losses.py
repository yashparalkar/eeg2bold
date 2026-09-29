"""Training objective.

    L = l_mse * MSE + l_t * L_t + l_s * L_s + l_fc * L_fc

  L_t  = mean_p (1 - r_t(yhat_p, y_p))   Pearson along time, per ROI
  L_s  = mean_b (1 - r_s(yhat_b, y_b))   Pearson across ROIs, per time point
  L_fc = MSE between predicted and true FC edges (upper triangle of the P x P
         correlation matrix computed over the batch)

A batch is (B, P) with B consecutive time points from one scan (see
``data.ContiguousBatchSampler``), so correlations along B are temporal and the batch
FC is a genuine within-scan connectivity estimate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


def pearson_along(a: torch.Tensor, b: torch.Tensor, dim: int, eps: float = 1e-8) -> torch.Tensor:
    a = a - a.mean(dim=dim, keepdim=True)
    b = b - b.mean(dim=dim, keepdim=True)
    num = (a * b).sum(dim=dim)
    den = torch.sqrt((a * a).sum(dim=dim) * (b * b).sum(dim=dim) + eps)
    return num / (den + eps)


def fc_edges(y: torch.Tensor, iu: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(B, P) -> upper-triangular edges of the P x P correlation matrix."""
    yc = y - y.mean(0, keepdim=True)
    yc = yc / (yc.norm(dim=0, keepdim=True) + eps)
    fc = yc.T @ yc
    return fc[iu[0], iu[1]]


class CompositeLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        iu = torch.triu_indices(cfg.n_rois, cfg.n_rois, offset=1)
        self.register_buffer("iu", iu, persistent=False)

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> dict:
        cfg = self.cfg
        l_mse = F.mse_loss(yhat, y)
        l_t = (1 - pearson_along(yhat, y, dim=0, eps=cfg.corr_eps)).mean()
        l_s = (1 - pearson_along(yhat, y, dim=1, eps=cfg.corr_eps)).mean()
        total = cfg.lambda_mse * l_mse + cfg.lambda_t * l_t + cfg.lambda_s * l_s
        out = {"mse": l_mse, "t": l_t, "s": l_s}
        if cfg.lambda_fc > 0 and yhat.shape[0] >= cfg.fc_min_block:
            l_fc = F.mse_loss(fc_edges(yhat, self.iu, cfg.corr_eps),
                              fc_edges(y, self.iu, cfg.corr_eps))
            out["fc"] = l_fc
            total = total + cfg.lambda_fc * l_fc
        out["total"] = total
        return out
