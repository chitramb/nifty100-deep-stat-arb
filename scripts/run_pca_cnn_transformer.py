from __future__ import annotations
import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torch import nn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(4)


def load_membership(path: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    m = pd.read_csv(path, parse_dates=["valid_from", "valid_to"])
    m = m[m["index_name"].str.lower().eq("nifty 100")]
    return m[(m["valid_to"].isna() | (m["valid_to"] >= start)) & (m["valid_from"] <= end)].copy()


def universe(m: pd.DataFrame, d: pd.Timestamp, columns: set[str]) -> list[str]:
    q = m[(m.valid_from <= d) & (m.valid_to.isna() | (m.valid_to > d))]
    return sorted(set(q.symbol) & columns)


def rolling_pca_residuals(
    returns: pd.DataFrame,
    membership: pd.DataFrame,
    dates: pd.DatetimeIndex,
    k: int = 5,
    pca_window: int = 252,
    beta_window: int = 60,
) -> pd.DataFrame:
    """Paper-style PCA residuals, computed strictly out of sample.

    At t-1 the PCA factors are estimated from the previous 252 days of the
    correlation matrix. Factor loadings are then estimated from the previous
    60 days, and the residual for t is computed using only information through
    t-1. This follows the paper's PCA procedure rather than fitting PCA on the
    same 1000-day window used to train the neural network.
    """
    out = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)
    for ti in range(pca_window + beta_window, len(dates)):
        d = dates[ti]
        assets = universe(membership, d, set(returns.columns))
        if len(assets) <= k:
            continue
        hist = returns.iloc[ti - pca_window:ti][assets]
        good = hist.notna().mean() >= 0.98
        assets = list(good[good].index)
        if len(assets) <= k:
            continue
        hist = hist[assets]
        mu = hist.mean()
        sd = hist.std(ddof=0).replace(0, np.nan)
        valid = sd.notna()
        assets = list(valid[valid].index)
        hist = hist[assets]
        mu, sd = mu[assets], sd[assets]
        if len(assets) <= k:
            continue
        z = (hist - mu) / sd
        if z.isna().any().any():
            continue
        pca = PCA(n_components=k, random_state=SEED)
        pca.fit(z.values)
        eig = pca.components_.T

        beta_dates = dates[ti - beta_window:ti]
        beta_r = returns.loc[beta_dates, assets]
        if beta_r.isna().any().any():
            continue
        beta_z = (beta_r - mu) / sd
        factors = beta_z.values @ eig
        y = beta_r.values
        X = np.column_stack([np.ones(len(beta_dates)), factors])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]

        current = returns.loc[d, assets].fillna(0.0)
        current_z = (current - mu) / sd
        fcur = current_z.values @ eig
        fitted = coef[0] + fcur @ coef[1:]
        out.loc[d, assets] = current.values - fitted
    return out


class CNNTransformer(nn.Module):
    """Small CNN + one-layer Transformer matching the paper's benchmark settings."""
    def __init__(self, filters: int = 8, heads: int = 4, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(1, filters, kernel_size=2, padding="same")
        self.norm1 = nn.InstanceNorm1d(filters, affine=True)
        self.conv2 = nn.Conv1d(filters, filters, kernel_size=2, padding="same")
        self.norm2 = nn.InstanceNorm1d(filters, affine=True)
        layer = nn.TransformerEncoderLayer(
            d_model=filters,
            nhead=heads,
            dim_feedforward=2 * filters,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.alloc = nn.Sequential(
            nn.LayerNorm(filters),
            nn.Linear(filters, 2 * filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * filters, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch x assets x 30 cumulative residual returns
        b, a, l = x.shape
        z = x.reshape(b * a, 1, l)
        z0 = z
        z = torch.relu(self.norm1(self.conv1(z)))
        z = torch.relu(self.norm2(self.conv2(z)))
        z = z + z0.expand(-1, z.shape[1], -1)
        z = z.transpose(1, 2)
        z = self.transformer(z)
        signal = z[:, -1, :].reshape(b, a, -1)
        raw = self.alloc(signal).squeeze(-1)
        raw = raw - raw.mean(dim=1, keepdim=True)
        return raw / raw.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)


def sharpe_objective(portfolio_returns: torch.Tensor) -> torch.Tensor:
    return -(portfolio_returns.mean() / (portfolio_returns.std(unbiased=True) + 1e-8)) * np.sqrt(252.0)


def train_model(residuals: pd.DataFrame, returns: pd.DataFrame, epochs: int = 100):
    arr = residuals.values
    samples, targets = [], []
    for i in range(30, len(residuals) - 1):
        window = arr[i - 29:i + 1].T
        if not np.isfinite(window).all():
            continue
        target = returns.iloc[i + 1].reindex(residuals.columns).fillna(0.0).values
        samples.append(np.cumsum(window, axis=1))
        targets.append(target)
    if len(samples) < 250:
        return None
    xs = torch.tensor(np.asarray(samples), dtype=torch.float32)
    ys = torch.tensor(np.asarray(targets), dtype=torch.float32)
    net = CNNTransformer()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    # The paper trains in consecutive, non-overlapping 125-day temporal batches.
    for _ in range(epochs):
        for start in range(0, len(xs), 125):
            xb, yb = xs[start:start + 125], ys[start:start + 125]
            if len(xb) < 2:
                continue
            w = net(xb)
            pr = (w * yb).sum(dim=1)
            loss = sharpe_objective(pr)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net


def main():
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
    r = np.log(p).diff()
    m = load_membership(args.membership, p.index.min(), p.index.max())
    dates = p.index

    residuals = rolling_pca_residuals(r, m, dates, k=5, pca_window=252, beta_window=60)
    residuals.to_parquet(out / "pca5_residuals.parquet")

    oos = dates[(dates >= pd.Timestamp(args.oos_start)) & (dates <= pd.Timestamp(args.end))]
    all_port, blocks = [], []
    for b0 in range(0, len(oos), 125):
        block = oos[b0:b0 + 125]
        if len(block) == 0:
            break
        d0 = block[0]
        hist = dates[dates < d0]
        if len(hist) < 1030:
            continue
        train_dates = hist[-1000:]
        assets = universe(m, d0, set(residuals.columns))
        if len(assets) < 50:
            continue
        rr = residuals.reindex(index=train_dates, columns=assets)
        yy = r.reindex(index=train_dates, columns=assets)
        keep = rr.notna().mean() >= 0.95
        assets = list(keep[keep].index)
        rr, yy = rr[assets], yy[assets]
        rr = rr.dropna(axis=1, how="any")
        assets = list(rr.columns)
        yy = yy[assets]
        if len(assets) < 50:
            continue
        net = train_model(rr, yy, epochs=args.epochs)
        if net is None:
            continue

        net.eval()
        block_ret = []
        for d in block:
            prior = dates[dates <= d]
            wdates = prior[-30:]
            sig = residuals.reindex(index=wdates, columns=assets)
            if len(wdates) < 30 or sig.isna().any().any():
                continue
            x = torch.tensor(np.cumsum(sig.values.T, axis=1)[None], dtype=torch.float32)
            with torch.no_grad():
                w = net(x).numpy()[0]
            realized = r.loc[d, assets].fillna(0.0).values
            pr = float(np.dot(w, realized))
            all_port.append((d, pr))
            block_ret.append(pr)
        if block_ret:
            a = np.asarray(block_ret)
            blocks.append({
                "start": str(block[0].date()),
                "end": str(block[-1].date()),
                "assets": len(assets),
                "n_days": len(a),
                "sharpe": float(np.sqrt(252) * a.mean() / (a.std(ddof=1) + 1e-12)),
            })

    port = pd.DataFrame(all_port, columns=["date", "return"]).drop_duplicates("date").sort_values("date")
    if len(port):
        vol = float(port["return"].std(ddof=1) * np.sqrt(252))
        mu = float(port["return"].mean() * 252)
        wealth = np.exp(port["return"].cumsum())
        dd = wealth / wealth.cummax() - 1
        metrics = {
            "oos_start": str(port.date.min().date()),
            "oos_end": str(port.date.max().date()),
            "n_days": len(port),
            "annualized_log_return": mu,
            "annualized_volatility": vol,
            "sharpe": float(mu / vol) if vol else None,
            "max_drawdown": float(dd.min()),
            "total_compounded_return": float(np.exp(port["return"].sum()) - 1),
        }
    else:
        metrics = {"error": "No out-of-sample observations were produced."}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "blocks.json").write_text(json.dumps(blocks, indent=2))
    port.to_csv(out / "portfolio_returns.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
