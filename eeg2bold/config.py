"""All tunable settings in one dataclass."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # ---- data ---------------------------------------------------------------
    n_channels: int = 26          # C: EEG channels
    fs: int = 200                 # EEG sampling rate (Hz)
    window_sec: int = 16          # EEG history used to predict one fMRI volume
    n_rois: int = 64              # P: fMRI ROIs predicted jointly
    tr: float = 2.1               # fMRI repetition time (s)

    # ---- spectral front-end -------------------------------------------------
    frontend: str = "cwt"         # "cwt" | "msse"
    d_model: int = 200            # D: token width
    spectral_log: bool = True     # log1p-compress spectral magnitudes
    # MSSE: multi-scale STFT, n_fft(l) = l_base * 2**l
    l_base: int = 100
    n_scales: int = 3
    # CWT: constant-Q Morlet filterbank
    cwt_n_freqs: int = 24
    cwt_fmin: float = 1.0
    cwt_fmax: float = 45.0
    cwt_n_cycles: float = 7.0

    # ---- transformer encoder ------------------------------------------------
    encoder_depth: int = 2
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # ---- leadfield + FIR readout --------------------------------------------
    readout_init: str = "leadfield"   # "leadfield" | "random"
    leadfield_sigma: float = 0.5      # width of the electrode->ROI proximity prior
    fir_smooth_lambda: float = 0.0    # roughness penalty on the FIR kernels

    # ---- loss ---------------------------------------------------------------
    lambda_mse: float = 0.8
    lambda_t: float = 0.2             # temporal Pearson (per ROI)
    lambda_s: float = 0.0             # spatial Pearson (per time point)
    lambda_fc: float = 0.2            # functional-connectivity consistency
    fc_min_block: int = 8             # skip L_fc on shorter batches
    corr_eps: float = 1e-8

    # ---- optimisation -------------------------------------------------------
    lr: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 20
    batch_size: int = 64
    grad_clip: float = 1.0
    seed: int = 0

    # ---- derived ------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        """T: EEG samples per window."""
        return self.fs * self.window_sec

    @property
    def n_time(self) -> int:
        """N: tokens per channel along time."""
        return self.n_samples // self.l_base

    @property
    def token_dt(self) -> float:
        """Seconds per time token (the FIR kernel's tap spacing)."""
        return self.window_sec / self.n_time

    def to_dict(self) -> dict:
        return asdict(self)
