from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn

SEED = 42
LOOKBACK = 30
PCA_WINDOW = 252
BETA_WINDOW = 60
TRAIN_WINDOW = 1000
RETRAIN_EVERY = 125
BATCH_DAYS = 125
FACTORS = 5
FILTERS = 8
HEADS = 4
DROPOUT = 0.25
LR = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)


@dataclass
class PCAState:
    assets: list[str]
    phi: np.ndarray


def load_membership(path: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    m = pd.read_csv(path, parse_dates=["valid_from", "valid_to"])
    m = m[m["index_name"].str.lower().eq("nifty 100")].copy()
    return m[(m["valid_to"].isna() | (m["valid_to"] >= start)) & (m["valid_from"] <= end)].copy()


def universe(m: pd.DataFrame, d: pd.Timestamp, columns: set[str]) -> list[str]:
    q = m[(m.valid_from <= d) & (m.valid_to.isna() | (m.valid_to > d))]
    return sorted(set(q.symbol) & columns)


def fit_pca_state(
    returns: pd.DataFrame,
    membership: pd.DataFrame,
    dates: pd.DatetimeIndex,
    ti: int,
    k: int = FACTORS,
    pca_window: int = PCA_WINDOW,
    beta_window: int = BETA_WINDOW,
) -> tuple[pd.Timestamp, PCAState, pd.Series] | None:
    """Estimate Phi_{t-1} and epsilon_t strictly out of sample.

    The paper estimates PCA factors from the previous 252 trading days, then
    estimates each stock's loading on those factors over the previous 60 days,
    and only then computes the residual for day t.  Missing observations in the
    rolling estimation window exclude that stock for that day's factor model.
    """
    if ti < pca_window + beta_window:
        return None
    d = dates[ti]
    candidate = universe(membership, d, set(returns.columns))
    if len(candidate) <= k:
        return None

    hist_all = returns.iloc[ti - pca_window:ti][candidate]
    beta_all = returns.iloc[ti - beta_window:ti][candidate]
    current = returns.iloc[ti][candidate]

    # This is a trading-date eligibility check, not a model input: a stock with
    # no realized return on t simply cannot contribute to the residual on t.
    complete = hist_all.notna().all(axis=0) & beta_all.notna().all(axis=0) & current.notna()
    assets = list(complete[complete].index)
    if len(assets) <= k:
        return None

    hist = hist_all[assets]
    mu = hist.mean()
    sd = hist.std(ddof=0).replace(0, np.nan)
    assets = list(sd[sd.notna()].index)
    if len(assets) <= k:
        return None
    hist = hist[assets]
    mu, sd = mu[assets], sd[assets]

    z = ((hist - mu) / sd).values
    if not np.isfinite(z).all():
        return None

    # PCA on standardized returns is PCA on the rolling correlation matrix.
    pca = PCA(n_components=k, random_state=SEED)
    pca.fit(z)
    V = pca.components_.T  # N x K eigenvectors

    beta_r = beta_all[assets]
    beta_z = (beta_r - mu) / sd
    if not np.isfinite(beta_z.values).all():
        return None
    factor_returns = beta_z.values @ V  # 60 x K

    # Equation R_t = beta' F_t + epsilon_t: no look-ahead and no separate
    # intercept is introduced, matching the paper's factor-model formulation.
    beta = np.linalg.lstsq(factor_returns, beta_r.values, rcond=None)[0].T  # N x K

    # F_t = w_F' R_t for w_F = diag(1/sigma) V.
    w_f = V / sd.to_numpy()[:, None]
    phi = np.eye(len(assets)) - beta @ w_f.T

    cur = current[assets]
    residual = pd.Series(phi @ cur.to_numpy(), index=assets, name=d)
    return d, PCAState(assets=assets, phi=phi), residual


def rolling_pca_residuals(
    returns: pd.DataFrame,
    membership: pd.DataFrame,
    dates: pd.DatetimeIndex,
    k: int = FACTORS,
) -> tuple[pd.DataFrame, dict[pd.Timestamp, PCAState]]:
    residuals = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)
    states: dict[pd.Timestamp, PCAState] = {}
    for ti in range(PCA_WINDOW + BETA_WINDOW, len(dates)):
        result = fit_pca_state(returns, membership, dates, ti, k=k)
        if result is None:
            continue
        d, state, residual = result
        residuals.loc[d, residual.index] = residual.to_numpy()
        states[d] = state
    return residuals, states


class CNNTransformer(nn.Module):
    """Paper-aligned 2-layer causal CNN + 1-layer Transformer + FFN allocator."""

    def __init__(self, filters: int = FILTERS, heads: int = HEADS, dropout: float = DROPOUT):
        super().__init__()
        self.filters = filters
        self.conv1 = nn.Conv1d(1, filters, kernel_size=2, padding=0, bias=True)
        self.norm1 = nn.InstanceNorm1d(filters, affine=True)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=2, padding=0, bias=True)
        self.norm2 = nn.InstanceNorm1d(filters, affine=True)
        layer = nn.TransformerEncoderLayer(
            d_model=filters,
            nhead=heads,
            dim_feedforward=2 * filters,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.alloc = nn.Sequential(
            nn.LayerNorm(filters),
            nn.Linear(filters, 2 * filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * filters, 1),
        )

    def _causal_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (conv.kernel_size[0] - 1, 0))
        return conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch x assets x 30 cumulative residual returns
        b, a, l = x.shape
        z = x.reshape(b * a, 1, l)
        x0 = z
        z = self._causal_conv(self.conv1, z)
        z = torch.relu(self.norm1(z))
        z = self._causal_conv(self.conv2, z)
        z = torch.relu(self.norm2(z))
        z = z + x0.expand(-1, self.filters, -1)
        z = z.transpose(1, 2)
        z = self.transformer(z)
        signal = z[:, -1, :]
        raw = self.alloc(signal).squeeze(-1)
        return raw.reshape(b, a)


def stock_weights_from_residual_weights(
    residual_weights: torch.Tensor,
    phi: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    rw = residual_weights * mask
    stock = torch.bmm(rw.unsqueeze(1), phi).squeeze(1)
    denom = stock.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
    return stock / denom


def make_training_batch(
    target_dates: list[pd.Timestamp],
    residuals: pd.DataFrame,
    returns: pd.DataFrame,
    states: dict[pd.Timestamp, PCAState],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]] | None:
    if not target_dates:
        return None
    union: set[str] = set()
    for d in target_dates:
        if d in states:
            union.update(states[d].assets)
    assets = sorted(union)
    if len(assets) < 50:
        return None

    n, l, a = len(target_dates), LOOKBACK, len(assets)
    x = np.zeros((n, a, l), dtype=np.float32)
    y = np.zeros((n, a), dtype=np.float32)
    mask = np.zeros((n, a), dtype=np.float32)
    phi = np.zeros((n, a, a), dtype=np.float32)
    pos = {s: j for j, s in enumerate(assets)}

    for bi, d in enumerate(target_dates):
        state = states.get(d)
        if state is None:
            continue
        active = [s for s in state.assets if s in pos]
        prior = residuals.index[residuals.index < d]
        if len(active) == 0 or len(prior) < LOOKBACK:
            continue
        wdates = prior[-LOOKBACK:]
        sig = residuals.loc[wdates, active]
        target = returns.loc[d, active]
        valid = sig.notna().all(axis=0) & target.notna()
        valid_assets = [s for s, ok in valid.items() if bool(ok)]
        if len(valid_assets) < 50:
            continue
        vidx = [pos[s] for s in valid_assets]
        x[bi, vidx, :] = np.cumsum(sig[valid_assets].to_numpy().T, axis=1).astype(np.float32)
        y[bi, vidx] = target[valid_assets].to_numpy(dtype=np.float32)
        mask[bi, vidx] = 1.0
        state_pos = {s: j for j, s in enumerate(state.assets)}
        si = [state_pos[s] for s in valid_assets]
        phi[bi][np.ix_(vidx, vidx)] = state.phi[np.ix_(si, si)].astype(np.float32)

    if mask.sum() < n * 50:
        return None
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask), torch.from_numpy(phi), assets


def sharpe_loss(portfolio_returns: torch.Tensor) -> torch.Tensor:
    return -(portfolio_returns.mean() / (portfolio_returns.std(unbiased=True) + 1e-8)) * np.sqrt(252.0)


def train_model(
    residuals: pd.DataFrame,
    returns: pd.DataFrame,
    states: dict[pd.Timestamp, PCAState],
    train_dates: pd.DatetimeIndex,
    epochs: int = 100,
) -> CNNTransformer | None:
    usable_dates = [d for d in train_dates if d in states]
    if len(usable_dates) < 300:
        return None
    batches = []
    for start in range(0, len(usable_dates), BATCH_DAYS):
        batch = make_training_batch(usable_dates[start:start + BATCH_DAYS], residuals, returns, states)
        if batch is not None:
            batches.append(batch)
    if len(batches) < 3:
        return None

    net = CNNTransformer()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    net.train()
    for _ in range(epochs):
        for x, y, mask, phi, _assets in batches:
            raw = net(x)
            stock_w = stock_weights_from_residual_weights(raw, phi, mask)
            pr = (stock_w * y).sum(dim=1)
            loss = sharpe_loss(pr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return net


def evaluate_day(
    net: CNNTransformer,
    d: pd.Timestamp,
    residuals: pd.DataFrame,
    returns: pd.DataFrame,
    state: PCAState,
) -> tuple[float, np.ndarray, list[str]] | None:
    prior = residuals.index[residuals.index < d]
    if len(prior) < LOOKBACK:
        return None
    active = state.assets
    wdates = prior[-LOOKBACK:]
    sig = residuals.loc[wdates, active]
    realized = returns.loc[d, active]
    valid = sig.notna().all(axis=0) & realized.notna()
    valid_assets = [s for s, ok in valid.items() if bool(ok)]
    if len(valid_assets) < 50:
        return None

    x = torch.from_numpy(np.cumsum(sig[valid_assets].to_numpy().T, axis=1)[None].astype(np.float32))
    raw = net(x)[0]
    mask = torch.ones((1, len(valid_assets)), dtype=torch.float32)
    state_pos = {s: i for i, s in enumerate(active)}
    si = [state_pos[s] for s in valid_assets]
    phi_sub = torch.from_numpy(state.phi[np.ix_(si, si)].astype(np.float32))[None]
    stock_w = stock_weights_from_residual_weights(raw.unsqueeze(0), phi_sub, mask)[0]
    r = torch.from_numpy(realized[valid_assets].to_numpy(dtype=np.float32))
    pr = float(torch.dot(stock_w, r).item())
    return pr, stock_w.numpy(), valid_assets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eod", required=True)
    ap.add_argument("--membership", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--oos-start", default="2019-01-02")
    ap.add_argument("--end", default="2026-09-23")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    e = pd.read_parquet(args.eod)
    e["date"] = pd.to_datetime(e["date"])
    e = e[(e.date >= args.start) & (e.date <= args.end)]
    p = e.pivot_table(index="date", columns="symbol", values="close_adjusted", aggfunc="last").sort_index()
    # The paper evaluates daily adjusted returns and uses cumulative residual returns
    # as the 30-day CNN input.
    r = p.pct_change(fill_method=None)
    m = load_membership(args.membership, p.index.min(), p.index.max())
    dates = p.index

    residuals, states = rolling_pca_residuals(r, m, dates, k=FACTORS)
    residuals.to_parquet(out / "pca5_residuals.parquet")

    oos = dates[(dates >= pd.Timestamp(args.oos_start)) & (dates <= pd.Timestamp(args.end))]
    all_port: list[tuple[pd.Timestamp, float]] = []
    blocks: list[dict] = []

    for b0 in range(0, len(oos), RETRAIN_EVERY):
        block = list(oos[b0:b0 + RETRAIN_EVERY])
        if not block:
            break
        d0 = block[0]
        hist = dates[dates < d0]
        if len(hist) < TRAIN_WINDOW:
            continue
        train_dates = hist[-TRAIN_WINDOW:]
        net = train_model(residuals, r, states, train_dates, epochs=args.epochs)
        if net is None:
            raise RuntimeError(f"Could not construct a valid training set for OOS block starting {d0.date()}")

        net.eval()
        block_ret = []
        with torch.no_grad():
            for d in block:
                state = states.get(d)
                if state is None:
                    continue
                result = evaluate_day(net, d, residuals, r, state)
                if result is None:
                    continue
                pr, _w, _assets = result
                all_port.append((d, pr))
                block_ret.append(pr)
        if block_ret:
            a = np.asarray(block_ret)
            blocks.append({
                "start": str(block[0].date()),
                "end": str(block[-1].date()),
                "n_days": len(a),
                "mean_daily_return": float(a.mean()),
                "vol_daily": float(a.std(ddof=1)),
                "sharpe": float(np.sqrt(252) * a.mean() / (a.std(ddof=1) + 1e-12)),
            })

    port = pd.DataFrame(all_port, columns=["date", "return"]).drop_duplicates("date").sort_values("date")
    if len(port) < 2:
        raise RuntimeError("No out-of-sample observations were produced; refusing to report fabricated or empty performance metrics.")

    daily = port["return"]
    mu = float(daily.mean() * 252)
    vol = float(daily.std(ddof=1) * np.sqrt(252))
    wealth = (1.0 + daily).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    metrics = {
        "oos_start": str(port.date.min().date()),
        "oos_end": str(port.date.max().date()),
        "n_days": int(len(port)),
        "annualized_mean_return": mu,
        "annualized_volatility": vol,
        "sharpe": float(mu / vol) if vol else None,
        "max_drawdown": float(dd.min()),
        "total_compounded_return": float(wealth.iloc[-1] - 1.0),
        "n_retraining_blocks": len(blocks),
        "pca_factors": FACTORS,
        "pca_window_days": PCA_WINDOW,
        "beta_window_days": BETA_WINDOW,
        "signal_lookback_days": LOOKBACK,
        "training_window_days": TRAIN_WINDOW,
        "retraining_frequency_days": RETRAIN_EVERY,
        "batch_days": BATCH_DAYS,
        "epochs": args.epochs,
        "seed": SEED,
    }

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "blocks.json").write_text(json.dumps(blocks, indent=2))
    port.to_csv(out / "portfolio_returns.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
