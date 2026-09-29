"""Transformer encoder over channel-time tokens.

in : e (B, C, N, D)   front-end tokens
out: Z (B, C*N, D)    unpooled, so the readout can weight every (channel, time) token
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: x + MHSA(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, cfg: Config):
        super().__init__()
        D = cfg.d_model
        self.norm1 = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(D)
        hidden = int(D * cfg.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(D, hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(hidden, D), nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    """Stack of ``cfg.encoder_depth`` Transformer blocks over all C*N tokens."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.encoder_depth)])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        B, C, N, D = e.shape
        h = e.reshape(B, C * N, D)
        for blk in self.blocks:
            h = blk(h)
        return self.out_norm(h)
