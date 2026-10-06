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
    m = m[m["index_name"].str.lower().eq("nifty 100")].copy()
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
    """Leakage-safe PCA factor residuals following the paper's rolling design.

    PCA is estimated from the previous 252 trading days of standardized returns.
    Factor loadings are estimated from the previous 60 trading days.  Both are
    therefore known before the residual at date t is formed.

    Missing observations are handled asset-by-asset rather than requiring every
    constituent to have an identical complete history.  This is important for a
    point-in-time index whose membership changes through time.
    """
    out = pd.DataFrame(index=dates, columns=returns.columns, dtype=float)
    for ti in range(pca_window + beta_window, len(dates)):
        d = dates[ti]
        assets = universe(membership, d, set(returns.columns))
        if len(assets) <= k:
            continue

        # PCA estimation window. Keep assets with sufficiently complete histories,
        # then use complete rows for the correlation/PCA calculation.
        hist = returns.iloc[ti - pca_window:ti][assets]
        completeness = hist.notna().mean()
        assets = list(completeness[completeness >= 0.98].index)
        if len(assets) <= k:
            continue
        hist = hist[assets]
        mu = hist.mean()
        sd = hist.std(ddof=0).replace(0, np.nan)
        valid = sd.notna()
        assets = list(valid[valid].index)
        if len(assets) <= k:
            continue
        hist = hist[assets]
        mu, sd = mu[assets], sd[assets]
        z = (hist - mu) / sd
        z = z.dropna(axis=0, how="any")
        if len(z) < max(60, k + 5):
            continue

        pca = PCA(n_components=k, random_state=SEED)
        pca.fit(z.values)
        eig = pca.components_.T

        # Estimate each asset's factor loading from the preceding 60 days.
        beta_dates = dates[ti - beta_window:ti]
        beta_r = returns.loc[beta_dates, assets]
        beta_z = (beta_r - mu) / sd
        complete_factor_rows = beta_z.dropna(axis=0, how="any")
        if len(complete_factor_rows) < max(30, k + 5):
            continue
        factors = complete_factor_rows.values @ eig

        coefs: dict[str, np.ndarray] = {}
        for j, asset in enumerate(assets):
            y = beta_r.loc[complete_factor_rows.index, asset].values
            if not np.isfinite(y).all():
                continue
            X = np.column_stack([np.ones(len(factors)), factors])
            coefs[asset] = np.linalg.lstsq(X, y, rcond=None)[0]
        if len(coefs) <= k:
            continue

        # Form today's residual using only today's return and yesterday's model.
        current = returns.loc[d, list(coefs.keys())]
        current_z = (current - mu[list(coefs.keys())]) / sd[list(coefs.keys())]
        for asset in coefs:
            value = current.get(asset, np.nan)
            if not np.isfinite(value):
                continue
            fcur = current_z[asset] * eig[assets.index(asset)]
            fitted = coefs[asset][0] + float(np.dot(fcur, coefs[asset][1:]))
            out.loc[d, asset] = float(value - fitted)
    return out


class CNNTransformer(nn.Module):
    """Small CNN + one-layer Transformer matching the paper benchmark settings."""
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

        # Require a usable 30-day signal history, not 100% availability over the
        # entire 1000-day training period. Membership changes naturally create NaNs.
        usable = rr.notna().rolling(30, min_periods=30).sum().max()
        keep = usable[usable >= 30].index
        rr, yy = rr[keep], yy[keep]
        if len(keep) < 50:
            continue
        # Training samples require complete 30-day windows; train_model filters them.
        net = train_model(rr, yy, epochs=args.epochs)
        if net is None:
            continue

        net.eval()
        block_ret = []
        for d in block:
            prior = dates[dates <= d]
            wdates = prior[-30:]
            sig = residuals.reindex(index=wdates, columns=keep)
            valid_assets = sig.columns[sig.notna().all()].tolist()
            if len(wdates) < 30 or len(valid_assets) < 50:
                continue
            x = torch.tensor(np.cumsum(sig[valid_assets].values.T, axis=1)[None], dtype=torch.float32)
            with torch.no_grad():
                w = net(x).numpy()[0]
            realized = r.loc[d, valid_assets].fillna(0.0).values
            pr = float(np.dot(w, realized))
            all_port.append((d, pr))
            block_ret.append(pr)
        if block_ret:
            a = np.asarray(block_ret)
            blocks.append({
                "start": str(block[0].date()),
                "end": str(block[-1].date()),
                "assets": len(keep),
                "n_days": len(a),
                "sharpe": float(np.sqrt(252) * a.mean() / (a.std(ddof=1) + 1e-12)),
            })

    port = pd.DataFrame(all_port, columns=["date", "return"]).drop_duplicates("date").sort_values("date")
    if len(port) >= 2:
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
            "n_blocks": len(blocks),
        }
    else:
        raise RuntimeError("No out-of-sample observations were produced; refusing to report fabricated or empty performance metrics.")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "blocks.json").write_text(json.dumps(blocks, indent=2))
    port.to_csv(out / "portfolio_returns.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
