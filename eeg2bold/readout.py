"""Leadfield + FIR readout and per-ROI regression heads.

Each ROI p reads the encoder output through a SEPARABLE spatio-temporal filter

    kernel_p(c, n) = w_p(c) * h_p(n)

* ``w_p`` (C,)  a scalp topography, initialised from a geometric leadfield prior
  (which electrodes sit closest to the ROI) and then learned freely;
* ``h_p`` (N,)  a free FIR haemodynamic kernel over the N time tokens, initialised to
  the canonical double-gamma HRF and then learned freely.

    z_p = sum_{c,n} kernel_p(c, n) * Z[c, n, :]        (B, P, D)
    y_p = <head_w_p, LN(z_p)> + head_b_p               (B, P)

No cross-attention: the readout costs P*(C + N + D + 1) parameters.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .config import Config

_EPS = 1e-6


# ---------------------------------------------------------------------------
# HRF / FIR
# ---------------------------------------------------------------------------
def _gamma_pdf(t: torch.Tensor, shape: float, scale: float = 1.0) -> torch.Tensor:
    t = t.clamp_min(_EPS)
    k = torch.tensor(shape, dtype=t.dtype)
    log_pdf = (k - 1) * torch.log(t) - t / scale - torch.lgamma(k) - k * np.log(scale)
    return torch.exp(log_pdf)


def canonical_hrf(t: torch.Tensor) -> torch.Tensor:
    """SPM canonical double-gamma HRF (peak ~6 s, undershoot ~16 s, ratio 1/6)."""
    return _gamma_pdf(t, 6.0) - _gamma_pdf(t, 16.0) / 6.0


class FIRKernel(nn.Module):
    """One free FIR kernel per ROI over the N token positions.

    Token n runs forward in time (n = N-1 is the most recent), so its haemodynamic lag
    is tau(n) = (N - 1 - n) * dt. Kernels are L1-normalised; gain lives in the head.
    """

    def __init__(self, cfg: Config, n_rois: int):
        super().__init__()
        N = cfg.n_time
        lags = (N - 1 - torch.arange(N, dtype=torch.float32)) * cfg.token_dt
        self.register_buffer("lags", lags, persistent=False)
        h0 = canonical_hrf(lags).unsqueeze(0)
        h0 = h0 / (h0.abs().sum(-1, keepdim=True) + _EPS)
        self.fir = nn.Parameter(h0.repeat(n_rois, 1))               # (P, N)

    def forward(self) -> torch.Tensor:
        return self.fir / (self.fir.abs().sum(-1, keepdim=True) + _EPS)

    def smoothness_penalty(self) -> torch.Tensor:
        h = self()
        d2 = h[:, 2:] - 2.0 * h[:, 1:-1] + h[:, :-2]
        return (d2 ** 2).mean()


# ---------------------------------------------------------------------------
# Leadfield prior
# ---------------------------------------------------------------------------
def leadfield_prior(roi_xyz: np.ndarray, elec_xyz: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Electrode->ROI proximity prior, (P, C), mean-centred per ROI.

    ``roi_xyz`` (P, 3) ROI centroids and ``elec_xyz`` (C, 3) electrode positions may
    live in different coordinate frames (e.g. MNI vs head space); each set is rescaled
    per axis to [-1, 1] first so only relative geometry matters. Each row is centred so
    it is a contrast between near and far electrodes rather than a weighted average.
    """
    roi = np.asarray(roi_xyz, dtype=np.float64)
    ele = np.asarray(elec_xyz, dtype=np.float64)
    roi = roi / np.maximum(np.abs(roi).max(0, keepdims=True), 1e-6)
    ele = ele / np.maximum(np.abs(ele).max(0, keepdims=True), 1e-6)
    d2 = ((roi[:, None, :] - ele[None, :, :]) ** 2).sum(-1)
    w = np.exp(-d2 / (2.0 * sigma ** 2))
    return (w - w.mean(1, keepdims=True)).astype(np.float32)


# ---------------------------------------------------------------------------
# Readout + heads
# ---------------------------------------------------------------------------
class RegressionHeads(nn.Module):
    """P independent linear regressors, one per ROI: (B, P, D) -> (B, P)."""

    def __init__(self, n_rois: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_rois, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_rois))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return (z * self.weight).sum(-1) + self.bias


class LeadfieldFIRReadout(nn.Module):
    """Per-ROI separable (topography x FIR) filter over the encoder tokens."""

    def __init__(self, cfg: Config, roi_xyz: Optional[np.ndarray] = None,
                 elec_xyz: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = cfg
        P, C, D = cfg.n_rois, cfg.n_channels, cfg.d_model
        self.in_norm = nn.LayerNorm(D)
        self.spatial = nn.Parameter(torch.randn(P, C) * C ** -0.5)   # unit-norm rows
        if cfg.readout_init == "leadfield":
            if roi_xyz is None or elec_xyz is None:
                raise ValueError("readout_init='leadfield' needs roi_xyz (P,3) and elec_xyz (C,3)")
            prior = torch.from_numpy(leadfield_prior(roi_xyz, elec_xyz, cfg.leadfield_sigma))
            rms = prior.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(_EPS)
            with torch.no_grad():                    # same scale as the random init
                self.spatial.copy_(prior / rms * C ** -0.5)
        elif cfg.readout_init != "random":
            raise ValueError(f"unknown readout_init {cfg.readout_init!r} (leadfield|random)")
        self.hrf = FIRKernel(cfg, P)
        self.out_norm = nn.LayerNorm(D)

    def kernels(self) -> torch.Tensor:
        """(P, C, N) spatio-temporal filter for every ROI."""
        return self.spatial[:, :, None] * self.hrf()[:, None, :]

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        B = Z.shape[0]
        Zg = self.in_norm(Z).reshape(B, cfg.n_channels, cfg.n_time, cfg.d_model)
        z = torch.einsum("bcnd,pcn->bpd", Zg, self.kernels())            # (B, P, D)
        return self.out_norm(z)
