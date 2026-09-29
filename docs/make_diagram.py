"""Render docs/architecture.png (run from the repo root: python docs/make_diagram.py)."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import gamma

INK = "#1f2933"
MUTED = "#52606d"
PALETTE = {                      # (fill, edge)
    "io": ("#f5f7fa", "#7b8794"),
    "front": ("#e3f2fd", "#1e88e5"),
    "enc": ("#ede7f6", "#5e35b1"),
    "read": ("#e8f5e9", "#2e7d32"),
    "head": ("#fff3e0", "#ef6c00"),
}

fig, ax = plt.subplots(figsize=(10, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis("off")


def box(x0, y0, w, h, kind, lw=1.6, alpha=1.0, z=1):
    fc, ec = PALETTE[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z))


def text(x, y, s, size=11, weight="normal", color=INK, ha="center", **kw):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, fontweight=weight, color=color,
            zorder=5, **kw)


def arrow(x0, y0, x1, y1, label=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 lw=1.6, color=MUTED, zorder=4))
    if label:
        text(x0 + 0.18, (y0 + y1) / 2, label, size=9.5, color=MUTED, ha="left", style="italic")


# ---- input -----------------------------------------------------------------
box(2.8, 12.95, 4.4, 0.8, "io")
text(5, 13.5, "Raw EEG window", 12.5, "bold")
text(5, 13.17, "C channels × 16 s  (B, C, T)", 10, color=MUTED)
arrow(5, 12.95, 5, 12.5)

# ---- 1. spectral front-end --------------------------------------------------
box(0.5, 9.95, 9.0, 2.5, "front", lw=2, alpha=0.45)
text(0.8, 12.15, "1   Spectral front-end  (choose one)", 12.5, "bold", PALETTE["front"][1], ha="left")
box(0.85, 10.15, 3.8, 1.7, "front")
text(2.75, 11.55, "MSSE", 12, "bold")
text(2.75, 10.95, "multi-scale STFT magnitude\nwindows 0.5 s / 1 s / 2 s\nLinear(freq→D), Linear(frames→N)\nsummed over scales",
     9.2, linespacing=1.35)
text(5, 11.0, "or", 11, "bold", MUTED)
box(5.35, 10.15, 3.8, 1.7, "front")
text(7.25, 11.55, "Morlet CWT", 12, "bold")
text(7.25, 10.95, "constant-Q wavelet filterbank\n24 bands, 1–45 Hz\npower pooled to N tokens\nLinear(bands→D)",
     9.2, linespacing=1.35)
arrow(5, 9.95, 5, 9.55)

box(2.2, 8.75, 5.6, 0.8, "front")
text(5, 9.3, "+ channel embedding  + sinusoidal time encoding", 10.5)
text(5, 8.97, "tokens (B, C, N, D)", 10, color=MUTED)
arrow(5, 8.75, 5, 8.35)

# ---- 2. transformer encoder -------------------------------------------------
box(0.5, 6.35, 9.0, 2.0, "enc", lw=2, alpha=0.45)
text(0.8, 8.05, "2   Transformer encoder", 12.5, "bold", PALETTE["enc"][1], ha="left")
text(9.2, 8.05, "× depth", 11.5, "bold", PALETTE["enc"][1], ha="right")
box(0.85, 6.55, 3.8, 1.15, "enc")
text(2.75, 7.3, "Multi-head self-attention", 11, "bold")
text(2.75, 6.92, "over all C·N channel-time tokens", 9.2, color=MUTED)
box(5.35, 6.55, 3.8, 1.15, "enc")
text(7.25, 7.3, "MLP (GELU)", 11, "bold")
text(7.25, 6.92, "pre-LayerNorm + residual", 9.2, color=MUTED)
arrow(4.65, 7.12, 5.35, 7.12)
arrow(5, 6.35, 5, 5.95, "Z  (B, C·N, D), unpooled")

# ---- 3. leadfield + FIR readout --------------------------------------------
box(0.5, 2.75, 9.0, 3.2, "read", lw=2, alpha=0.45)
text(0.8, 5.65, "3   Leadfield + FIR readout  (one filter per ROI p)", 12.5, "bold",
     PALETTE["read"][1], ha="left")
box(0.85, 4.3, 3.8, 1.05, "read")
text(2.75, 5.0, r"Scalp topography  $w_p \in \mathbb{R}^C$", 10.5, "bold")
text(2.75, 4.6, "init: electrode→ROI leadfield prior", 9.2, color=MUTED)
box(5.35, 4.3, 3.8, 1.05, "read")
text(6.55, 5.0, r"FIR HRF  $h_p \in \mathbb{R}^N$", 10.5, "bold")
text(6.55, 4.6, "init: canonical HRF", 9.2, color=MUTED)
# tiny HRF sketch
ins = ax.inset_axes([7.9, 4.4, 1.15, 0.85], transform=ax.transData, zorder=6)
t = np.linspace(0, 16, 200)
hrf = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
ins.plot(t, hrf, color=PALETTE["read"][1], lw=1.8)
ins.axhline(0, color=MUTED, lw=0.6)
ins.set_xticks([]); ins.set_yticks([])
for s in ins.spines.values():
    s.set_visible(False)
ins.patch.set_alpha(0)
arrow(2.75, 4.3, 4.3, 3.95)
arrow(7.25, 4.3, 5.7, 3.95)
box(2.3, 3.35, 5.4, 0.6, "read")
text(5, 3.65, r"$k_p(c,n) = w_p(c)\,h_p(n)$     $z_p = \sum_{c,n} k_p(c,n)\,Z_{c,n,:}$", 11)
text(5, 3.0, "(B, P, D)", 9.5, color=MUTED)
arrow(5, 2.75, 5, 2.35)

# ---- 4. regression heads ----------------------------------------------------
box(2.2, 1.55, 5.6, 0.8, "head")
text(5, 2.1, "4   Per-ROI regression heads", 11.5, "bold", PALETTE["head"][1])
text(5, 1.78, r"$\hat{y}_p = \langle a_p, \mathrm{LN}(z_p) \rangle + b_p$", 10.5)
arrow(5, 1.55, 5, 1.15)

# ---- output -----------------------------------------------------------------
box(2.8, 0.3, 4.4, 0.85, "io")
text(5, 0.87, "Predicted BOLD", 12.5, "bold")
text(5, 0.55, "P ROIs  (B, P)", 10, color=MUTED)

# ---- training objective (side note) -----------------------------------------
text(9.45, 0.72, "Loss:  MSE + temporal r\n+ spatial r + FC", 9, color=MUTED, ha="right",
     linespacing=1.4)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", out)
