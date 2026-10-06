import numpy as np
import pandas as pd
import torch

from scripts.run_pca_cnn_transformer import (
    CNNTransformer,
    PCAState,
    make_training_batch,
    stock_weights_from_residual_weights,
)


def test_stock_weight_mapping_and_l1_normalization():
    rw = torch.tensor([[1.0, -2.0, 0.5]])
    phi = torch.eye(3).unsqueeze(0)
    mask = torch.ones_like(rw)
    w = stock_weights_from_residual_weights(rw, phi, mask)
    assert torch.allclose(w.abs().sum(dim=1), torch.ones(1), atol=1e-7)
    assert torch.allclose(w, rw / rw.abs().sum(), atol=1e-7)


def test_cnn_transformer_shape_and_no_tanh_output_constraint():
    torch.manual_seed(1)
    net = CNNTransformer()
    x = torch.randn(2, 7, 30)
    y = net(x)
    assert y.shape == (2, 7)
    # The paper's final FFN is linear; weights are normalized after mapping,
    # so the raw allocator is not artificially clipped to [-1, 1].
    assert torch.isfinite(y).all()


def test_training_batch_uses_only_pre_target_residuals():
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    assets = [f"A{i}" for i in range(50)]
    rng = np.random.default_rng(42)
    residuals = pd.DataFrame(rng.normal(size=(40, 50)), index=dates, columns=assets)
    returns = pd.DataFrame(rng.normal(size=(40, 50)) * 0.01, index=dates, columns=assets)
    states = {d: PCAState(assets=assets, phi=np.eye(50)) for d in dates[30:]}

    batch = make_training_batch(list(dates[30:40]), residuals, returns, states)
    assert batch is not None
    x, y, mask, phi, out_assets = batch
    assert x.shape == (10, 50, 30)
    assert y.shape == (10, 50)
    assert mask.shape == (10, 50)
    assert phi.shape == (10, 50, 50)
    assert out_assets == assets

    # First target date is dates[30], so its 30-day signal must end at dates[29].
    expected = residuals.loc[dates[:30], assets].to_numpy().T.cumsum(axis=1)
    assert np.allclose(x[0].numpy(), expected, atol=1e-6)
