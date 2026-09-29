"""Evaluation metrics on full-scan predictions.

pred, true: (T, P) - T fMRI time points, P ROIs. Metrics are computed within each
scan and then averaged across scans (``evaluate_grouped``).
"""
from __future__ import annotations

import numpy as np


def _pearson(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + eps))


def _fc(x: np.ndarray) -> np.ndarray:
    return np.corrcoef(x.T)


def _upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices_from(m, k=1)]


def per_roi_corr(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Temporal Pearson per ROI -> (P,)."""
    return np.array([_pearson(pred[:, p], true[:, p]) for p in range(pred.shape[1])])


def spatial_corr(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8) -> float:
    """Pearson across ROIs at each time point, averaged over time."""
    a = pred - pred.mean(1, keepdims=True)
    b = true - true.mean(1, keepdims=True)
    r = (a * b).sum(1) / (np.sqrt((a * a).sum(1) * (b * b).sum(1)) + eps)
    return float(r.mean())


def fc_corr(pred: np.ndarray, true: np.ndarray) -> float:
    """Pearson between predicted and true FC edges."""
    return _pearson(_upper(_fc(pred)), _upper(_fc(true)))


def fc_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """MSE between predicted and true FC edges."""
    return float(np.mean((_upper(_fc(pred)) - _upper(_fc(true))) ** 2))


def fc_f1(pred: np.ndarray, true: np.ndarray, percentile: float = 75.0) -> float:
    """F1 of the strongest edges, each FC binarised at its own percentile."""
    pe, te = _upper(_fc(pred)), _upper(_fc(true))
    p = pe >= np.percentile(pe, percentile)
    t = te >= np.percentile(te, percentile)
    tp = np.sum(p & t)
    if tp == 0:
        return 0.0
    prec, rec = tp / p.sum(), tp / t.sum()
    return float(2 * prec * rec / (prec + rec))


def evaluate(pred: np.ndarray, true: np.ndarray) -> dict:
    return {
        "tcorr": float(per_roi_corr(pred, true).mean()),
        "mse": float(np.mean((pred - true) ** 2)),
        "spatial_corr": spatial_corr(pred, true),
        "fc_corr": fc_corr(pred, true),
        "fc_mse": fc_mse(pred, true),
        "fc_f1_top25": fc_f1(pred, true, 75.0),
        "fc_f1_top50": fc_f1(pred, true, 50.0),
    }


def evaluate_grouped(pred: np.ndarray, true: np.ndarray, groups: np.ndarray) -> dict:
    """Per-scan metrics (``groups`` = scan id per row), mean and std across scans."""
    groups = np.asarray(groups)
    per = [evaluate(pred[groups == g], true[groups == g]) for g in np.unique(groups)]
    out = {"n_scans": len(per)}
    for k in per[0]:
        vals = np.array([r[k] for r in per])
        out[k] = float(vals.mean())
        out[k + "_std"] = float(vals.std())
    return out
