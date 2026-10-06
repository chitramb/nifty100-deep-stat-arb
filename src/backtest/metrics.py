from __future__ import annotations
import numpy as np
import pandas as pd

def annualized_sharpe(daily_returns: pd.Series, periods: int = 252) -> float:
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(periods) * r.mean() / r.std(ddof=1))

def max_drawdown(daily_returns: pd.Series) -> float:
    wealth = (1.0 + pd.Series(daily_returns).fillna(0.0)).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())
