import numpy as np
import pytest
import torch

from eeg2bold import CompositeLoss, Config, EEG2BOLD, evaluate_grouped


@pytest.mark.parametrize("frontend", ["cwt", "msse"])
@pytest.mark.parametrize("init", ["leadfield", "random"])
def test_forward_backward(frontend, init):
    cfg = Config(frontend=frontend, readout_init=init, encoder_depth=1, d_model=32,
                 n_heads=4, n_channels=8, n_rois=10)
    rng = np.random.default_rng(0)
    model = EEG2BOLD(cfg, rng.normal(size=(10, 3)), rng.normal(size=(8, 3)))
    x = torch.randn(16, cfg.n_channels, cfg.n_samples)
    y = torch.randn(16, cfg.n_rois)
    yhat = model(x)
    assert yhat.shape == (16, cfg.n_rois)
    loss = CompositeLoss(cfg)(yhat, y)["total"] + model.regularisation()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.hrf_kernels().shape == (cfg.n_rois, cfg.n_time)
    assert model.scalp_maps().shape == (cfg.n_rois, cfg.n_channels)


def test_metrics():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(100, 10))
    m = evaluate_grouped(true, true, np.repeat([0, 1], 50))
    assert m["tcorr"] == pytest.approx(1.0, abs=1e-6)
    assert m["fc_corr"] == pytest.approx(1.0, abs=1e-6)
