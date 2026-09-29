"""End-to-end model: EEG window (B, C, T) -> BOLD at P ROIs (B, P)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .encoder import Encoder
from .frontend import build_frontend
from .readout import LeadfieldFIRReadout, RegressionHeads


class EEG2BOLD(nn.Module):
    """Spectral front-end -> Transformer encoder -> leadfield/FIR readout -> heads."""

    def __init__(self, cfg: Config, roi_xyz: Optional[np.ndarray] = None,
                 elec_xyz: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = cfg
        self.frontend = build_frontend(cfg)
        self.encoder = Encoder(cfg)
        self.readout = LeadfieldFIRReadout(cfg, roi_xyz, elec_xyz)
        self.heads = RegressionHeads(cfg.n_rois, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.frontend(x)          # (B, C, N, D)
        Z = self.encoder(e)           # (B, C*N, D)
        z = self.readout(Z)           # (B, P, D)
        return self.heads(z)          # (B, P)

    def regularisation(self) -> torch.Tensor:
        """Extra penalty added to the loss (FIR smoothness; 0 when disabled)."""
        if self.cfg.fir_smooth_lambda > 0:
            return self.cfg.fir_smooth_lambda * self.readout.hrf.smoothness_penalty()
        return torch.zeros((), device=self.heads.bias.device)

    # -- interpretability ----------------------------------------------------
    @torch.no_grad()
    def scalp_maps(self) -> torch.Tensor:
        """(P, C) learned topography per ROI (signed)."""
        return self.readout.spatial.detach().clone()

    @torch.no_grad()
    def hrf_kernels(self) -> torch.Tensor:
        """(P, N) learned FIR kernel per ROI, index n = N-1 is the most recent token."""
        return self.readout.hrf().detach().clone()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
