"""Windowed EEG -> BOLD data.

A *scan* is a pair of simultaneously recorded arrays:

    eeg  (C, S)  float, S samples at cfg.fs
    bold (T, P)  float, T volumes at cfg.tr, P ROIs

Sample k of a scan is the EEG window of ``cfg.window_sec`` seconds ending at the
onset of volume k, paired with that volume's P ROI values.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import Config

Scan = Tuple[np.ndarray, np.ndarray]


def zscore(x: np.ndarray, axis: int) -> np.ndarray:
    return ((x - x.mean(axis, keepdims=True)) / (x.std(axis, keepdims=True) + 1e-8)).astype(np.float32)


class WindowDataset(Dataset):
    def __init__(self, scans: Sequence[Scan], cfg: Config, normalise: bool = True):
        self.T = cfg.n_samples
        self.eeg: List[np.ndarray] = []
        self.bold: List[np.ndarray] = []
        index = []
        for s, (eeg, bold) in enumerate(scans):
            if normalise:                         # per-scan z-score
                eeg, bold = zscore(eeg, axis=1), zscore(bold, axis=0)
            self.eeg.append(np.asarray(eeg, dtype=np.float32))
            self.bold.append(np.asarray(bold, dtype=np.float32))
            for k in range(bold.shape[0]):
                end = int(round(k * cfg.tr * cfg.fs))
                if end >= self.T and end <= eeg.shape[1]:
                    index.append((s, k, end))
        self.index = index
        self.groups = np.array([s for s, _, _ in index])   # scan id per sample

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        s, k, end = self.index[i]
        x = self.eeg[s][:, end - self.T:end]
        y = self.bold[s][k]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)


class ContiguousBatchSampler(Sampler):
    """Batches of consecutive time points from a single scan.

    Needed by the temporal-Pearson and FC loss terms, which correlate along the batch.
    Block order is shuffled for training and kept in time order for evaluation.
    """

    def __init__(self, groups: np.ndarray, batch_size: int, shuffle: bool = True, seed: int = 0):
        self.blocks = []
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            self.blocks += [idx[i:i + batch_size].tolist() for i in range(0, len(idx), batch_size)]
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = self.rng.permutation(len(self.blocks)) if self.shuffle else range(len(self.blocks))
        for b in order:
            yield self.blocks[b]

    def __len__(self) -> int:
        return len(self.blocks)


def load_npz_dir(path: str) -> Tuple[List[Scan], List[str]]:
    """Load every ``*.npz`` in ``path`` (keys ``eeg`` (C,S), ``bold`` (T,P), optional ``subject``)."""
    scans, subjects = [], []
    for f in sorted(glob.glob(os.path.join(path, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        scans.append((d["eeg"], d["bold"]))
        subjects.append(str(d["subject"]) if "subject" in d else os.path.basename(f))
    if not scans:
        raise FileNotFoundError(f"no .npz scans found in {path}")
    return scans, subjects


def make_synthetic_scans(cfg: Config, n_scans: int = 8, n_tr: int = 200, n_sources: int = 4,
                         seed: int = 0) -> Tuple[List[Scan], List[str], np.ndarray, np.ndarray]:
    """Toy data with a real EEG-band-power -> HRF -> BOLD relationship.

    Latent sources modulate the amplitude of band-limited oscillations on the scalp;
    the same envelopes, convolved with a canonical HRF, drive the ROI BOLD signals.
    Returns (scans, subject ids, roi_xyz (P,3), elec_xyz (C,3)).
    """
    from .readout import canonical_hrf

    rng = np.random.default_rng(seed)
    C, P, K, fs = cfg.n_channels, cfg.n_rois, n_sources, cfg.fs
    elec = rng.normal(size=(C, 3)); elec /= np.linalg.norm(elec, axis=1, keepdims=True)
    elec[:, 2] = np.abs(elec[:, 2])                           # upper hemisphere
    roi = rng.uniform(-0.8, 0.8, size=(P, 3))
    src = rng.uniform(-0.8, 0.8, size=(K, 3))
    prox = lambda a, b: np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / 0.5)
    A, B = prox(elec, src), prox(roi, src)                    # (C, K), (P, K)
    bands = rng.choice([6.0, 10.0, 20.0, 30.0], size=K)

    hrf_t = np.arange(0, 32, 0.1)
    hrf = canonical_hrf(torch.tensor(hrf_t, dtype=torch.float32)).numpy()
    scans, subjects = [], []
    for s in range(n_scans):
        n_sec = int(np.ceil(n_tr * cfg.tr)) + 2
        S = n_sec * fs
        t = np.arange(S) / fs
        z = rng.standard_normal((K, n_sec * 10))              # slow envelope on a 10 Hz grid
        z = np.apply_along_axis(lambda v: np.convolve(v, np.hanning(60), "same"), 1, z)
        env10 = np.exp(0.8 * zscore(z, axis=1))                 # (K, n_sec*10)
        env = np.stack([np.interp(t, np.arange(env10.shape[1]) / 10, e) for e in env10])
        osc = env * np.sin(2 * np.pi * bands[:, None] * t + rng.uniform(0, 6.3, (K, 1)))
        eeg = A @ osc + 0.5 * rng.standard_normal((C, S))
        drive = np.stack([np.convolve(e, hrf)[: env10.shape[1]] for e in env10])  # (K, n_sec*10)
        vol_idx = (np.arange(n_tr) * cfg.tr * 10).astype(int)
        bold = (B @ drive[:, vol_idx]).T + 0.3 * rng.standard_normal((n_tr, P))
        scans.append((eeg.astype(np.float32), bold.astype(np.float32)))
        subjects.append(f"sub-{s // 2:02d}")                  # two scans per subject
    return scans, subjects, roi.astype(np.float32), elec.astype(np.float32)


def split_by_subject(subjects: Sequence[str], val_frac: float, seed: int = 0,
                     val_subjects: Optional[Sequence[str]] = None) -> Tuple[List[int], List[int]]:
    """Scan indices for train / validation with no subject in both."""
    uniq = sorted(set(subjects))
    if val_subjects is None:
        rng = np.random.default_rng(seed)
        n_val = max(1, int(round(len(uniq) * val_frac)))
        val_subjects = list(rng.choice(uniq, size=n_val, replace=False))
    val = set(val_subjects)
    tr = [i for i, s in enumerate(subjects) if s not in val]
    va = [i for i, s in enumerate(subjects) if s in val]
    return tr, va
