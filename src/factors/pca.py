from __future__ import annotations
import pandas as pd
from sklearn.decomposition import PCA

def fit_pca_residuals(returns: pd.DataFrame, n_factors: int = 5) -> pd.DataFrame:
    if returns.empty:
        raise ValueError("returns is empty")
    if n_factors < 0:
        raise ValueError("n_factors must be non-negative")
    if n_factors == 0:
        return returns.copy()
    x = returns.dropna(axis=1, how="all").copy()
    x = x.fillna(x.mean())
    if x.shape[1] <= n_factors:
        raise ValueError("n_factors must be smaller than number of assets")
    pca = PCA(n_components=n_factors)
    scores = pca.fit_transform(x)
    reconstruction = pca.inverse_transform(scores)
    return pd.DataFrame(x - reconstruction, index=x.index, columns=x.columns)
