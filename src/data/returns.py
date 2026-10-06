from __future__ import annotations
import numpy as np
import pandas as pd

def compute_log_returns(prices: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    if not {"date", "symbol", field}.issubset(prices.columns):
        raise ValueError("prices must contain date, symbol and requested price field")
    x = prices.copy()
    x["date"] = pd.to_datetime(x["date"])
    x["symbol"] = x["symbol"].astype(str).str.upper()
    x = x.sort_values(["symbol", "date"])
    x["log_return"] = x.groupby("symbol")[field].transform(lambda s: np.log(s).diff())
    return x[["date", "symbol", "log_return"]]
