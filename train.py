"""Train and evaluate eeg2bold.

    python train.py --synthetic --epochs 3                    # smoke run on toy data
    python train.py --data data/ --elec-pos elec.npy --roi-pos roi.npy
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg2bold import CompositeLoss, Config, EEG2BOLD, evaluate_grouped
from eeg2bold.data import (ContiguousBatchSampler, WindowDataset, load_npz_dir,
                           make_synthetic_scans, split_by_subject)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="directory of .npz scans (keys: eeg, bold, [subject])")
    src.add_argument("--synthetic", action="store_true", help="use generated toy data")
    p.add_argument("--elec-pos", help=".npy (C, 3) electrode positions (leadfield init)")
    p.add_argument("--roi-pos", help=".npy (P, 3) ROI centroids (leadfield init)")
    p.add_argument("--frontend", choices=["cwt", "msse"], default="cwt")
    p.add_argument("--readout-init", choices=["leadfield", "random"], default="leadfield")
    p.add_argument("--depth", type=int, default=Config.encoder_depth)
    p.add_argument("--d-model", type=int, default=Config.d_model)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--fs", type=int, default=Config.fs)
    p.add_argument("--tr", type=float, default=Config.tr)
    p.add_argument("--window-sec", type=int, default=Config.window_sec)
    p.add_argument("--lambda-mse", type=float, default=Config.lambda_mse)
    p.add_argument("--lambda-t", type=float, default=Config.lambda_t)
    p.add_argument("--lambda-s", type=float, default=Config.lambda_s)
    p.add_argument("--lambda-fc", type=float, default=Config.lambda_fc)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="runs/default")
    return p.parse_args()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg = Config(frontend=a.frontend, readout_init=a.readout_init, encoder_depth=a.depth,
                 d_model=a.d_model, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                 fs=a.fs, tr=a.tr, window_sec=a.window_sec, lambda_mse=a.lambda_mse,
                 lambda_t=a.lambda_t, lambda_s=a.lambda_s, lambda_fc=a.lambda_fc, seed=a.seed)

    roi_xyz = np.load(a.roi_pos) if a.roi_pos else None
    elec_xyz = np.load(a.elec_pos) if a.elec_pos else None
    if a.synthetic:
        scans, subjects, roi_xyz, elec_xyz = make_synthetic_scans(cfg, seed=a.seed)
    else:
        scans, subjects = load_npz_dir(a.data)
        cfg.n_channels, cfg.n_rois = scans[0][0].shape[0], scans[0][1].shape[1]
    if cfg.readout_init == "leadfield" and (roi_xyz is None or elec_xyz is None):
        print("[warn] no --elec-pos/--roi-pos given; falling back to readout_init='random'")
        cfg.readout_init = "random"

    tr_idx, va_idx = split_by_subject(subjects, a.val_frac, seed=a.seed)
    train_ds = WindowDataset([scans[i] for i in tr_idx], cfg)
    val_ds = WindowDataset([scans[i] for i in va_idx], cfg)
    train_dl = DataLoader(train_ds, batch_sampler=ContiguousBatchSampler(
        train_ds.groups, cfg.batch_size, shuffle=True, seed=a.seed))
    val_dl = DataLoader(val_ds, batch_sampler=ContiguousBatchSampler(
        val_ds.groups, cfg.batch_size, shuffle=False))

    model = EEG2BOLD(cfg, roi_xyz, elec_xyz).to(a.device)
    criterion = CompositeLoss(cfg).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * len(train_dl))
    print(f"model: {model.n_params():,} params | frontend={cfg.frontend} "
          f"readout_init={cfg.readout_init} | train {len(train_ds)} / val {len(val_ds)} windows")

    os.makedirs(a.out, exist_ok=True)
    best, history = -np.inf, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0, run = time.time(), []
        for x, y in train_dl:
            x, y = x.to(a.device), y.to(a.device)
            losses = criterion(model(x), y)
            loss = losses["total"] + model.regularisation()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            run.append(loss.item())
        pred, true = predict(model, val_dl, a.device)
        m = evaluate_grouped(pred, true, val_ds.groups)
        m.update(epoch=epoch, train_loss=float(np.mean(run)))
        history.append(m)
        print(f"epoch {epoch:3d} | loss {m['train_loss']:.4f} | tcorr {m['tcorr']:.4f} "
              f"| fc_corr {m['fc_corr']:.4f} | mse {m['mse']:.4f} | {time.time() - t0:.1f}s")
        if m["tcorr"] > best:
            best = m["tcorr"]
            torch.save({"model": model.state_dict(), "config": cfg.to_dict()},
                       os.path.join(a.out, "best.pt"))
    with open(os.path.join(a.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
