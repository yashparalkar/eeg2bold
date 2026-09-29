"""Spectral front-ends: raw EEG (B, C, T) -> tokens (B, C, N, D).

Two interchangeable options, selected by ``cfg.frontend``:

* ``MSSE`` - multi-scale spectral embedding. Per-channel STFT magnitude at several
  window lengths, each projected along frequency (-> D) and along time (-> N), then
  summed across scales.
* ``MorletCWT`` - continuous wavelet transform with a constant-Q Morlet filterbank.
  Bandwidth grows with centre frequency, so low frequencies get long analysis windows
  and high frequencies get short ones. Power is averaged onto the same N-step token
  grid as MSSE, so everything downstream is unchanged.

Both add a learnable channel embedding and a sinusoidal positional encoding along time.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .config import Config


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal PE added along the time axis (second-to-last dim)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[-2]]


class _FrontEnd(nn.Module):
    """Shared channel embedding + positional encoding."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.channel_emb = nn.Parameter(torch.randn(cfg.n_channels, cfg.d_model) * 0.02)
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=max(cfg.n_time, 64))

    def _finish(self, e: torch.Tensor) -> torch.Tensor:
        # e: (B, C, N, D)
        e = e + self.channel_emb[None, :, None, :]
        return self.pos_enc(e)

    def _check(self, x: torch.Tensor) -> None:
        _, C, T = x.shape
        if C != self.cfg.n_channels or T != self.cfg.n_samples:
            raise ValueError(f"expected (B, {self.cfg.n_channels}, {self.cfg.n_samples}), "
                             f"got {tuple(x.shape)}")


class MSSE(_FrontEnd):
    """Multi-scale STFT magnitude embedding."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        T, D, N = cfg.n_samples, cfg.d_model, cfg.n_time
        self.n_fft = [cfg.l_base * 2 ** l for l in range(cfg.n_scales)]
        self.hop = [nf // 2 for nf in self.n_fft]
        freq_bins = [nf // 2 + 1 for nf in self.n_fft]
        frames = [1 + T // h for h in self.hop]          # torch.stft(center=True)
        self.freq_proj = nn.ModuleList([nn.Linear(f, D) for f in freq_bins])
        self.time_proj = nn.ModuleList([nn.Linear(f, N) for f in frames])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        B, C, T = x.shape
        flat = x.reshape(B * C, T)
        e = 0
        for l, (nf, hop) in enumerate(zip(self.n_fft, self.hop)):
            win = torch.hann_window(nf, device=x.device, dtype=x.dtype)
            mag = torch.stft(flat, n_fft=nf, hop_length=hop, window=win,
                             center=True, return_complex=True).abs()   # (BC, F, frames)
            if self.cfg.spectral_log:
                mag = torch.log1p(mag)
            h = self.freq_proj[l](mag.transpose(1, 2))                 # (BC, frames, D)
            h = self.time_proj[l](h.transpose(1, 2)).transpose(1, 2)   # (BC, N, D)
            e = e + h
        return self._finish(e.reshape(B, C, self.cfg.n_time, self.cfg.d_model))


class MorletCWT(_FrontEnd):
    """Constant-Q Morlet CWT, pooled to the token grid, then Linear(n_freqs -> D)."""

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        T, N = cfg.n_samples, cfg.n_time
        if T % N:
            raise ValueError(f"n_samples={T} must be divisible by n_time={N}")
        if cfg.cwt_fmax > cfg.fs / 2:
            raise ValueError(f"cwt_fmax={cfg.cwt_fmax} exceeds Nyquist ({cfg.fs / 2})")
        if cfg.cwt_n_cycles / cfg.cwt_fmin * cfg.fs > T:
            raise ValueError("the lowest-frequency wavelet does not fit inside the window")
        self.hop = T // N

        centre = np.geomspace(cfg.cwt_fmin, cfg.cwt_fmax, cfg.cwt_n_freqs)
        f = np.fft.fftfreq(T, d=1.0 / cfg.fs)
        psi = np.zeros((cfg.cwt_n_freqs, T), dtype=np.float32)
        for i, fc in enumerate(centre):
            sigma_f = fc / cfg.cwt_n_cycles                  # bandwidth proportional to fc
            g = np.exp(-((f - fc) ** 2) / (2 * sigma_f ** 2))
            g[f <= 0] = 0.0                                  # analytic wavelet
            psi[i] = g
        self.register_buffer("psi", torch.from_numpy(psi), persistent=False)
        self.register_buffer("centre_freqs", torch.from_numpy(centre.astype(np.float32)),
                             persistent=False)
        self.freq_proj = nn.Linear(cfg.cwt_n_freqs, cfg.d_model)

    @torch.no_grad()
    def scalogram(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B*C, N, n_freqs) magnitude, pooled to the token grid.

        The wavelets are fixed and the input needs no gradient, so this runs without
        autograd and costs no activation memory.
        """
        B, C, T = x.shape
        xf = torch.fft.fft(x.reshape(B * C, T), dim=-1)
        cols = []
        for i in range(self.psi.shape[0]):
            w = torch.fft.ifft(xf * self.psi[i], dim=-1)
            pw = (w.real ** 2 + w.imag ** 2).reshape(B * C, self.cfg.n_time, self.hop)
            cols.append(pw.mean(-1))
        mag = torch.stack(cols, dim=-1).sqrt()
        return torch.log1p(mag) if self.cfg.spectral_log else mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check(x)
        B, C, _ = x.shape
        h = self.freq_proj(self.scalogram(x))                         # (BC, N, D)
        return self._finish(h.reshape(B, C, self.cfg.n_time, self.cfg.d_model))


def build_frontend(cfg: Config) -> nn.Module:
    if cfg.frontend == "msse":
        return MSSE(cfg)
    if cfg.frontend == "cwt":
        return MorletCWT(cfg)
    raise ValueError(f"unknown frontend {cfg.frontend!r} (msse|cwt)")
