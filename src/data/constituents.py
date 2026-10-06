from __future__ import annotations
import pandas as pd

def validate_constituent_history(df: pd.DataFrame) -> pd.DataFrame:
    required = {"effective_date", "symbol", "index_name", "in_index"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing constituent columns: {sorted(missing)}")
    out = df.copy()
    out["effective_date"] = pd.to_datetime(out["effective_date"])
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["index_name"] = out["index_name"].astype(str)
    out["in_index"] = out["in_index"].astype(bool)
    return out.sort_values(["effective_date", "symbol"]).reset_index(drop=True)

def membership_on_date(history: pd.DataFrame, date: str | pd.Timestamp) -> set[str]:
    d = pd.Timestamp(date)
    h = validate_constituent_history(history)
    eligible = h[h["effective_date"] <= d]
    if eligible.empty:
        return set()
    latest = eligible["effective_date"].max()
    row = eligible[eligible["effective_date"] == latest]
    return set(row.loc[row["in_index"], "symbol"])
